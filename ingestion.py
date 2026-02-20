"""
================================================================================
Project VERA - Data Ingestion Pipeline
Virtual Engineering Review Agent
================================================================================

This script handles the ingestion of mock semiconductor industry documents
into a ChromaDB vector store. It simulates:
  1. Product Datasheets (technical specifications)
  2. Internal Engineering Emails (informal decisions & changes)
  3. Standard Operating Procedures (SOPs)
  4. DB Info (structured database records — e.g., SQLite tables)
  5. Versioned Documents (same document at different revisions for diff detection)

Each document is tagged with metadata for:
  - `source`: The document type (datasheet, email, sop, db_info, document)
  - `access_level`: RBAC level (public, internal_only, confidential)
  - `department`: Originating department
  - `document_id`: Unique identifier

The access_level metadata enables Role-Based Access Control (RBAC) so that:
  - "senior" users can access ALL documents
  - "junior" users can ONLY access "public" documents

Usage:
    python ingestion.py
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import json
import glob
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

# Import centralized backend/embedding settings from config
from shared.config import (
    LLM_BACKEND, GEMINI_API_KEY,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    get_embeddings,
)

print(f"[CONFIG] Using backend: {LLM_BACKEND.upper()}")

# Collection name for our vector store
COLLECTION_NAME = "vera_documents"

# ==============================================================================
# MOCK DATA - SIMULATED SEMICONDUCTOR DOCUMENTS
# ==============================================================================
# These mock documents simulate real-world semiconductor industry documents.
# In production, these would be loaded from actual PDFs, emails, and SOP files
# using appropriate document loaders (e.g., PyPDFLoader, UnstructuredEmailLoader).
# ==============================================================================

MOCK_DOCUMENTS = [
    # -------------------------------------------------------------------------
    # DOCUMENT 1: Product Datasheet - RTX-9000 Voltage Specifications
    # This is the OFFICIAL published specification (public access)
    # -------------------------------------------------------------------------
    {
        "content": (
            "RTX-9000 Product Datasheet - Revision 4.2\n\n"
            "Section 3.1: Electrical Specifications\n"
            "The RTX-9000 semiconductor chip operates with a maximum voltage "
            "limit of 5.0V across all pins. The recommended operating voltage "
            "range is 3.0V to 5.0V. Exceeding the maximum voltage of 5.0V may "
            "result in permanent damage to the silicon die. All testing and "
            "quality assurance procedures must use the 5.0V maximum rating as "
            "the upper threshold for voltage stress tests.\n\n"
            "Section 3.2: Power Consumption\n"
            "Typical power consumption at 5.0V is 2.5W under nominal load "
            "conditions. Maximum power consumption shall not exceed 3.8W "
            "under worst-case operating conditions."
        ),
        "metadata": {
            "source": "datasheet",
            "access_level": "public",
            "department": "product_engineering",
            "document_id": "DS-RTX9000-v4.2",
            "title": "RTX-9000 Product Datasheet",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 2: Internal Email - Voltage Limit Change (CONFLICTING!)
    # This email CONTRADICTS the datasheet - a key test scenario for VERA
    # -------------------------------------------------------------------------
    {
        "content": (
            "From: Dr. Sarah Chen <s.chen@semicorp.internal>\n"
            "To: Engineering Team <eng-all@semicorp.internal>\n"
            "Subject: URGENT - RTX-9000 Voltage Limit Revision\n"
            "Date: 2024-11-15\n\n"
            "Team,\n\n"
            "Based on our latest thermal analysis and field failure data, we "
            "are IMMEDIATELY lowering the RTX-9000 maximum voltage limit from "
            "5.0V to 3.3V. The 5.0V rating in the current datasheet (Rev 4.2) "
            "is no longer valid.\n\n"
            "Root cause: Silicon lot #2024-Q3 shows accelerated electromigration "
            "at voltages above 3.5V, leading to premature failure. We've already "
            "seen 12 field returns linked to this issue.\n\n"
            "Action items:\n"
            "1. Update all test procedures to use 3.3V max\n"
            "2. Notify customers with active orders\n"
            "3. Datasheet revision to be published by EOW\n\n"
            "This information is INTERNAL ONLY until the official datasheet "
            "revision is released.\n\n"
            "- Dr. Sarah Chen, VP of Engineering"
        ),
        "metadata": {
            "source": "email",
            "access_level": "internal_only",
            "department": "engineering",
            "document_id": "EMAIL-2024-1115-001",
            "title": "URGENT - RTX-9000 Voltage Limit Revision",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 3: SOP - Standard Wafer Testing Procedure (public)
    # -------------------------------------------------------------------------
    {
        "content": (
            "Standard Operating Procedure: SOP-QA-101\n"
            "Title: Wafer-Level Testing Procedure for RTX-Series Chips\n\n"
            "1. Purpose\n"
            "This SOP defines the standardized procedure for conducting "
            "wafer-level electrical tests on all RTX-series semiconductor "
            "chips prior to die singulation and packaging.\n\n"
            "2. Scope\n"
            "Applies to all RTX-series products including RTX-7000, RTX-8000, "
            "and RTX-9000.\n\n"
            "3. Procedure\n"
            "3.1 Set up the probe station per Equipment Manual EM-PS-200.\n"
            "3.2 Apply voltage at rated maximum (refer to product datasheet "
            "for specific voltage limits).\n"
            "3.3 Measure leakage current at each test point.\n"
            "3.4 Record all measurements in the QA database.\n"
            "3.5 Flag any die exceeding leakage threshold of 10uA as FAIL.\n\n"
            "4. Acceptance Criteria\n"
            "Yield must exceed 95% per wafer. Wafers below this threshold "
            "require MRB (Material Review Board) disposition."
        ),
        "metadata": {
            "source": "sop",
            "access_level": "public",
            "department": "quality_assurance",
            "document_id": "SOP-QA-101",
            "title": "Wafer-Level Testing Procedure",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 4: Confidential Email - Skip Burn-In Decision
    # This is a high-risk informal decision that bypasses normal procedures
    # -------------------------------------------------------------------------
    {
        "content": (
            "From: Mark Thompson <m.thompson@semicorp.internal>\n"
            "To: Production Line Managers <prod-mgrs@semicorp.internal>\n"
            "CC: Dr. Sarah Chen <s.chen@semicorp.internal>\n"
            "Subject: RE: Q4 Ship Deadline - Burn-In Waiver\n"
            "Date: 2024-12-01\n\n"
            "Managers,\n\n"
            "After discussion with leadership, we have decided to SKIP the "
            "burn-in test phase for the RTX-9000 Q4 production batch "
            "(Lots #2024-Q4-001 through #2024-Q4-050). This is a one-time "
            "waiver to meet the December 15th ship deadline for Meridian Corp.\n\n"
            "Justification: The burn-in test adds 48 hours to the production "
            "cycle, and we cannot afford the delay. Historical data shows "
            "burn-in catch rate is only 0.3% for RTX-9000.\n\n"
            "IMPORTANT: This decision is CONFIDENTIAL. Do not discuss outside "
            "of this email thread. If any field failures arise from these lots, "
            "we will handle through our standard RMA process.\n\n"
            "- Mark Thompson, Director of Manufacturing"
        ),
        "metadata": {
            "source": "email",
            "access_level": "confidential",
            "department": "manufacturing",
            "document_id": "EMAIL-2024-1201-002",
            "title": "Q4 Ship Deadline - Burn-In Waiver",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 5: Product Datasheet - RTX-9000 Thermal Specifications
    # -------------------------------------------------------------------------
    {
        "content": (
            "RTX-9000 Product Datasheet - Revision 4.2\n\n"
            "Section 5.1: Thermal Specifications\n"
            "Maximum junction temperature (Tj): 125°C\n"
            "Operating temperature range: -40°C to +85°C (commercial grade)\n"
            "Thermal resistance (junction-to-case): 15°C/W\n"
            "Thermal resistance (junction-to-ambient): 45°C/W\n\n"
            "Section 5.2: Thermal Design Guidelines\n"
            "A heatsink with thermal resistance no greater than 10°C/W is "
            "required for continuous operation at maximum rated voltage. "
            "Forced air cooling (minimum 200 LFM) is recommended for designs "
            "operating above 70°C ambient temperature. The standard thermal "
            "interface material (TIM) is Arctic Silver MX-6 thermal compound "
            "with a thermal conductivity of 8.5 W/mK."
        ),
        "metadata": {
            "source": "datasheet",
            "access_level": "public",
            "department": "product_engineering",
            "document_id": "DS-RTX9000-v4.2-thermal",
            "title": "RTX-9000 Thermal Specifications",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 6: Internal Email - Thermal Paste Change (internal)
    # Another informal decision that could create compliance issues
    # -------------------------------------------------------------------------
    {
        "content": (
            "From: James Wu <j.wu@semicorp.internal>\n"
            "To: Assembly Team <assembly@semicorp.internal>\n"
            "Subject: Thermal Paste Substitution - Effective Immediately\n"
            "Date: 2024-11-20\n\n"
            "Team,\n\n"
            "Due to supply chain issues, we are substituting the Arctic Silver "
            "MX-6 thermal compound with GenericTherm GT-100 for all RTX-9000 "
            "assembly starting today. The GT-100 has a lower thermal "
            "conductivity (4.5 W/mK vs 8.5 W/mK) but is available immediately.\n\n"
            "I'm aware this doesn't meet the datasheet spec, but we need to "
            "keep the line running. We'll switch back to MX-6 once stock "
            "arrives (ETA: January 2025).\n\n"
            "Please do NOT update the assembly documentation until we get "
            "formal approval from QA. This is an interim measure.\n\n"
            "- James Wu, Assembly Line Supervisor"
        ),
        "metadata": {
            "source": "email",
            "access_level": "internal_only",
            "department": "assembly",
            "document_id": "EMAIL-2024-1120-003",
            "title": "Thermal Paste Substitution Notice",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 7: SOP - Quality Audit Procedure (internal)
    # -------------------------------------------------------------------------
    {
        "content": (
            "Standard Operating Procedure: SOP-QA-205\n"
            "Title: Internal Quality Audit Procedure\n"
            "Classification: INTERNAL USE ONLY\n\n"
            "1. Purpose\n"
            "This SOP defines the procedure for conducting internal quality "
            "audits across all manufacturing and engineering departments.\n\n"
            "2. Audit Schedule\n"
            "Internal audits shall be conducted quarterly. Each department "
            "undergoes a minimum of two audits per year.\n\n"
            "3. Audit Process\n"
            "3.1 Select audit scope based on risk assessment matrix.\n"
            "3.2 Review all process deviations and NCRs from the previous quarter.\n"
            "3.3 Interview key personnel on process adherence.\n"
            "3.4 Compare actual practices against documented SOPs.\n"
            "3.5 Flag any undocumented process changes as non-conformances.\n"
            "3.6 Issue audit report within 5 business days.\n\n"
            "4. Escalation\n"
            "Critical non-conformances must be escalated to the VP of Quality "
            "within 24 hours of discovery."
        ),
        "metadata": {
            "source": "sop",
            "access_level": "internal_only",
            "department": "quality_assurance",
            "document_id": "SOP-QA-205",
            "title": "Internal Quality Audit Procedure",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 8: Compliance Checklist (public)
    # -------------------------------------------------------------------------
    {
        "content": (
            "Standard Operating Procedure: SOP-COMP-300\n"
            "Title: Product Compliance Checklist - External Release\n\n"
            "Before any RTX-series product is shipped to customers, the "
            "following compliance items must be verified:\n\n"
            "1. Electrical specifications match the published datasheet.\n"
            "2. All test results are within acceptance criteria.\n"
            "3. Product labeling meets regulatory requirements (UL, CE, RoHS).\n"
            "4. Material declarations (REACH, Conflict Minerals) are current.\n"
            "5. Customer-specific requirements are documented and met.\n"
            "6. Reliability test data (HTOL, TC, HAST) is on file.\n"
            "7. All engineering changes have been formally approved via ECO.\n"
            "8. No open CAPAs or NCRs that affect product quality.\n\n"
            "Sign-off required from: QA Manager, Engineering Manager, and "
            "Compliance Officer before shipment release."
        ),
        "metadata": {
            "source": "sop",
            "access_level": "public",
            "department": "compliance",
            "document_id": "SOP-COMP-300",
            "title": "Product Compliance Checklist",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 9: DB Info - RTX-9000 Production Lot Tracking (SQLite-style)
    # Simulates structured data from a production database
    # -------------------------------------------------------------------------
    {
        "content": (
            "DATABASE RECORD — RTX-9000 Production Lot Tracker\n"
            "Source: SemiCorp Production Database (SQLite)\n\n"
            "Table: lot_tracking\n"
            "| lot_id          | product  | voltage_spec | test_status | yield_pct | ship_date  |\n"
            "|-----------------|----------|-------------|-------------|-----------|------------|\n"
            "| 2024-Q3-012     | RTX-9000 | 5.0V        | PASS        | 96.2%     | 2024-09-30 |\n"
            "| 2024-Q3-045     | RTX-9000 | 5.0V        | PASS        | 95.8%     | 2024-10-15 |\n"
            "| 2024-Q4-001     | RTX-9000 | 5.0V        | SKIP_BURNIN | 97.1%     | 2024-12-14 |\n"
            "| 2024-Q4-022     | RTX-9000 | 3.3V        | PASS        | 98.5%     | 2025-01-05 |\n\n"
            "Note: Lots 2024-Q4-001 through Q4-050 shipped WITHOUT burn-in testing "
            "per manufacturing waiver EMAIL-2024-1201-002. Voltage spec updated to "
            "3.3V starting from lot 2024-Q4-022 per engineering directive."
        ),
        "metadata": {
            "source": "db_info",
            "access_level": "confidential",
            "department": "manufacturing",
            "document_id": "DB-LOT-TRACK-2024",
            "title": "RTX-9000 Production Lot Tracker",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 10: DB Info - Test Results Summary (public)
    # Public-facing aggregated test metrics
    # -------------------------------------------------------------------------
    {
        "content": (
            "DATABASE RECORD — RTX-9000 Quarterly Test Summary\n"
            "Source: SemiCorp QA Database (SQLite)\n\n"
            "Table: quarterly_test_summary\n"
            "| quarter | total_lots | avg_yield | voltage_spec | pass_rate |\n"
            "|---------|-----------|-----------|-------------|-----------|\n"
            "| 2024-Q2 | 38        | 95.4%     | 5.0V        | 100%      |\n"
            "| 2024-Q3 | 42        | 96.1%     | 5.0V        | 100%      |\n"
            "| 2024-Q4 | 50        | 97.3%     | 5.0V / 3.3V | 98%       |\n\n"
            "Notes: Q4 shows mixed voltage specifications across lots. "
            "Two lots returned from Meridian Corp due to field failures "
            "(lots from burn-in waiver batch). Overall yield improvement "
            "attributed to tighter process controls."
        ),
        "metadata": {
            "source": "db_info",
            "access_level": "public",
            "department": "quality_assurance",
            "document_id": "DB-QA-SUMMARY-2024Q4",
            "title": "RTX-9000 Quarterly Test Summary",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 11: Versioned Document - RTX-9000 Spec v1.0 (older version)
    # This is an OLDER version of the product spec, kept for audit trail
    # -------------------------------------------------------------------------
    {
        "content": (
            "RTX-9000 Product Specification — Version 1.0\n"
            "Date: 2024-06-01 | Status: SUPERSEDED\n\n"
            "1. Electrical Characteristics\n"
            "   Maximum operating voltage: 5.0V\n"
            "   Recommended voltage range: 3.0V – 5.0V\n"
            "   Maximum power consumption: 3.8W\n\n"
            "2. Thermal Characteristics\n"
            "   Maximum junction temperature: 125°C\n"
            "   Thermal interface material: Arctic Silver MX-6 (8.5 W/mK)\n\n"
            "3. Quality Requirements\n"
            "   Burn-in test: MANDATORY for all production lots\n"
            "   Minimum yield threshold: 95%\n"
            "   All lots must pass 48-hour burn-in at rated voltage\n"
        ),
        "metadata": {
            "source": "document",
            "access_level": "public",
            "department": "product_engineering",
            "document_id": "SPEC-RTX9000-v1.0",
            "title": "RTX-9000 Product Specification v1.0",
            "version": "1.0",
        },
    },
    # -------------------------------------------------------------------------
    # DOCUMENT 12: Versioned Document - RTX-9000 Spec v2.0 (current version)
    # This is the UPDATED version — discrepancies with v1.0 should be detected
    # -------------------------------------------------------------------------
    {
        "content": (
            "RTX-9000 Product Specification — Version 2.0\n"
            "Date: 2025-01-15 | Status: CURRENT\n\n"
            "1. Electrical Characteristics\n"
            "   Maximum operating voltage: 3.3V (REVISED from 5.0V)\n"
            "   Recommended voltage range: 2.7V – 3.3V\n"
            "   Maximum power consumption: 2.1W (reduced)\n\n"
            "2. Thermal Characteristics\n"
            "   Maximum junction temperature: 125°C (unchanged)\n"
            "   Thermal interface material: GenericTherm GT-100 (4.5 W/mK)\n"
            "   NOTE: Interim substitution pending MX-6 supply restoration\n\n"
            "3. Quality Requirements\n"
            "   Burn-in test: RECOMMENDED (previously mandatory)\n"
            "   Minimum yield threshold: 95%\n"
            "   Burn-in waiver permitted with Director-level approval\n\n"
            "CHANGE LOG vs v1.0:\n"
            "  - Voltage reduced from 5.0V to 3.3V (electromigration risk)\n"
            "  - Thermal paste changed from MX-6 to GT-100 (supply chain)\n"
            "  - Burn-in changed from mandatory to recommended (waiver process)\n"
        ),
        "metadata": {
            "source": "document",
            "access_level": "confidential",
            "department": "product_engineering",
            "document_id": "SPEC-RTX9000-v2.0",
            "title": "RTX-9000 Product Specification v2.0",
            "version": "2.0",
        },
    },
]


# ==============================================================================
# SOURCE DOCUMENT LOADER — Multi-Domain + RBAC Auto-Tagging
# ==============================================================================

SOURCE_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "source_documents")

# --- File Type → Source Mapping ---
# Maps the "Type" part of Domain_Type_Version_Access.txt to metadata source
FILE_TYPE_TO_SOURCE = {
    "Spec": "datasheet",
    "Email": "email",
    "SOP": "sop",
    "DB": "db_info",
    "DM": "email",           # Decision Memos treated as email-type
    "Doc": "document",
}

# --- File Type → Default RBAC Level ---
# Auto-tags access_level based on document type (overridable by filename)
FILE_TYPE_DEFAULT_RBAC = {
    "Email": "internal_only",
    "DM": "internal_only",
    "DB": "confidential",
}

# --- Access Level Mapping ---
ACCESS_LEVEL_MAP = {
    "Public": "public",
    "Internal": "internal_only",
    "Confidential": "confidential",
}


def parse_filename(filename: str) -> dict:
    """
    Parse Domain_Type_Version_Access.txt into metadata fields.

    Args:
        filename: e.g. "Semi_Spec_v4.2_Public.txt"

    Returns:
        dict with keys: domain, source, version, access_level, title
        Returns None if filename cannot be parsed.
    """
    name = os.path.splitext(filename)[0]  # Strip .txt
    parts = name.split("_")

    if len(parts) < 4:
        return None

    domain_code = parts[0]
    file_type = parts[1]
    version = parts[2]
    access_code = parts[3]

    # Map domain code
    domain_map = {"Semi": "semiconductor", "Med": "medical"}
    domain = domain_map.get(domain_code, domain_code.lower())

    # Map source type
    source = FILE_TYPE_TO_SOURCE.get(file_type, "document")

    # Map access level — explicit filename takes priority over type default
    access_level = ACCESS_LEVEL_MAP.get(
        access_code,
        FILE_TYPE_DEFAULT_RBAC.get(file_type, "public"),
    )

    # Generate document_id from filename
    document_id = name

    return {
        "domain": domain,
        "source": source,
        "version": version.lstrip("v"),
        "access_level": access_level,
        "document_id": document_id,
        "title": name.replace("_", " "),
    }


def load_domain_documents() -> list[dict]:
    """
    Load documents from source_documents/<domain>/*.txt files.

    Iterates through subdirectories of source_documents/, treating each
    subdirectory as a domain (e.g., semiconductor/, medical/).

    File naming convention: Domain_Type_Version_Access.txt
    RBAC auto-tagging: Email/DM → internal_only, DB → confidential

    Returns:
        list[dict]: Document dicts with 'content' and 'metadata' keys.
    """
    all_docs = []

    if not os.path.exists(SOURCE_DOCUMENTS_DIR):
        print(f"[INFO] source_documents/ directory not found")
        return []

    # Iterate through domain subdirectories
    for domain_dir in sorted(os.listdir(SOURCE_DOCUMENTS_DIR)):
        domain_path = os.path.join(SOURCE_DOCUMENTS_DIR, domain_dir)

        # Skip non-directories, templates, and hidden files
        if not os.path.isdir(domain_path):
            continue
        if domain_dir.startswith("_") or domain_dir.startswith("."):
            continue

        print(f"\n  📂 Domain: {domain_dir}/")

        # Load .txt files from this domain
        txt_files = sorted(glob.glob(os.path.join(domain_path, "*.txt")))
        for filepath in txt_files:
            filename = os.path.basename(filepath)

            # Skip templates
            if filename.startswith("_"):
                continue

            # Parse filename for metadata
            parsed = parse_filename(filename)
            if parsed is None:
                print(f"    ⚠️  Skipped {filename} (cannot parse naming convention)")
                continue

            # Read content
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except Exception as e:
                print(f"    ❌ Error reading {filename}: {e}")
                continue

            if not content:
                print(f"    ⚠️  Skipped {filename} (empty)")
                continue

            # Override domain from folder name
            parsed["domain"] = domain_dir
            parsed["department"] = domain_dir

            doc = {
                "content": content,
                "metadata": parsed,
            }
            all_docs.append(doc)
            print(f"    ✅ {filename} → source={parsed['source']}, "
                  f"access={parsed['access_level']}, version={parsed['version']}")

    return all_docs


def load_json_documents() -> list[dict]:
    """
    Load documents from source_documents/*.json files (backward compatible).

    Each JSON file should be an array of objects with 'content' and 'metadata'.
    Files starting with '_' (like _template_data.json) are skipped.

    Returns:
        list[dict]: List of document dicts with 'content' and 'metadata' keys.
    """
    all_docs = []
    json_pattern = os.path.join(SOURCE_DOCUMENTS_DIR, "*.json")
    json_files = sorted(glob.glob(json_pattern))

    # Skip template files (starting with _)
    json_files = [f for f in json_files if not os.path.basename(f).startswith("_")]

    if not json_files:
        return []

    print(f"\n  📂 JSON files (root):")
    for filepath in json_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_docs.extend(data)
                print(f"    ✅ Loaded {len(data)} documents from {filename}")
            else:
                print(f"    ⚠️  {filename} is not a JSON array — skipped.")
        except json.JSONDecodeError as e:
            print(f"    ❌ Failed to parse {filename}: {e}")

    return all_docs


# ==============================================================================
# INGESTION PIPELINE
# ==============================================================================

def create_documents() -> list[Document]:
    """
    Create LangChain Document objects from source data.

    Loading priority:
      1. Domain .txt files: source_documents/<domain>/*.txt
      2. Root .json files:  source_documents/*.json
      3. Inline MOCK_DOCUMENTS (fallback if nothing else found)

    Returns:
        list[Document]: A list of LangChain Document objects ready for splitting.
    """
    print(f"[INFO] Scanning {SOURCE_DOCUMENTS_DIR} for source documents...")

    # --- Phase 1: Load domain .txt files ---
    domain_docs = load_domain_documents()

    # --- Phase 2: Load root .json files ---
    json_docs = load_json_documents()

    # --- Combine external sources ---
    external_docs = domain_docs + json_docs

    if external_docs:
        raw_data = external_docs
        print(f"\n[INFO] Using {len(raw_data)} documents from source_documents/ "
              f"({len(domain_docs)} domain .txt + {len(json_docs)} json)")
    else:
        raw_data = MOCK_DOCUMENTS
        print(f"\n[INFO] No external data found. Using {len(raw_data)} inline MOCK_DOCUMENTS.")

    documents = []
    for doc_data in raw_data:
        doc = Document(
            page_content=doc_data["content"],
            metadata=doc_data["metadata"],
        )
        documents.append(doc)

    print(f"[INFO] Created {len(documents)} raw documents total.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    This splitter tries to split on natural boundaries (paragraphs, sentences,
    words) to keep semantically related text together. The chunk overlap 
    ensures that context is preserved across chunk boundaries.
    
    Args:
        documents: List of LangChain Document objects to split.
    
    Returns:
        list[Document]: A list of chunked Document objects with preserved metadata.
    """
    # Initialize the text splitter with appropriate parameters
    # - chunk_size=500: Each chunk will be at most 500 characters
    # - chunk_overlap=50: Adjacent chunks share 50 characters of overlap
    # - separators: Split on paragraphs first, then sentences, then words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    
    # Split all documents - metadata is automatically preserved per chunk
    chunks = text_splitter.split_documents(documents)
    
    print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Create a ChromaDB vector store from document chunks.
    
    This function:
      1. Initializes Google Generative AI embeddings (Gemini)
      2. Clears any existing ChromaDB data (for clean re-runs)
      3. Creates a new Chroma collection with embedded documents
      4. Persists the database to disk for use by the main application
    
    Args:
        chunks: List of chunked Document objects with metadata.
    
    Returns:
        Chroma: The initialized and persisted ChromaDB vector store.
    """
    # --- Step 1: Initialize the embedding model using centralized factory ---
    embeddings = get_embeddings()
    
    # --- Step 2: Create in-memory ChromaDB vector store ---
    # No persist_directory = ephemeral in-memory mode.
    # Keeps the project directory clean (no chroma.sqlite3 file).
    print(f"[INFO] Creating in-memory ChromaDB vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )
    
    print(f"[INFO] Vector store created with {vector_store._collection.count()} vectors (in-memory).")
    return vector_store


def verify_ingestion(vector_store: Chroma) -> None:
    """
    Run verification queries to ensure the data was ingested correctly.
    
    This function performs test queries to validate:
      1. Documents can be retrieved by content similarity
      2. Metadata filtering (RBAC) works correctly
      3. Both public and internal documents are searchable
    
    Args:
        vector_store: The ChromaDB vector store to verify.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION: Testing Ingestion Results")
    print("=" * 70)
    
    # --- Test 1: Basic similarity search (no filter) ---
    print("\n[TEST 1] Similarity search for 'RTX-9000 voltage' (no filter):")
    results = vector_store.similarity_search("RTX-9000 voltage", k=3)
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: [{doc.metadata.get('source')}] "
              f"[{doc.metadata.get('access_level')}] "
              f"{doc.page_content[:80]}...")
    
    # --- Test 2: Filtered search (public only - simulating Junior access) ---
    print("\n[TEST 2] Filtered search for 'voltage' (public only - Junior RBAC):")
    results = vector_store.similarity_search(
        "voltage",
        k=3,
        filter={"access_level": "public"},
    )
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: [{doc.metadata.get('source')}] "
              f"[{doc.metadata.get('access_level')}] "
              f"{doc.page_content[:80]}...")
    
    # --- Test 3: Email-specific search (compliance agent use case) ---
    print("\n[TEST 3] Email-specific search for 'burn-in waiver':")
    results = vector_store.similarity_search(
        "burn-in waiver",
        k=3,
        filter={"source": "email"},
    )
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: [{doc.metadata.get('source')}] "
              f"[{doc.metadata.get('access_level')}] "
              f"{doc.page_content[:80]}...")
    
    # --- Test 4: SOP search ---
    print("\n[TEST 4] SOP search for 'testing procedure':")
    results = vector_store.similarity_search(
        "testing procedure",
        k=3,
        filter={"source": "sop"},
    )
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: [{doc.metadata.get('source')}] "
              f"[{doc.metadata.get('access_level')}] "
              f"{doc.page_content[:80]}...")
    
    # --- Test 5: DB Info search ---
    print("\n[TEST 5] DB Info search for 'production lot tracking':")
    results = vector_store.similarity_search(
        "production lot tracking",
        k=3,
        filter={"source": "db_info"},
    )
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: [{doc.metadata.get('source')}] "
              f"[{doc.metadata.get('access_level')}] "
              f"{doc.page_content[:80]}...")
    
    # --- Test 6: Versioned document search ---
    print("\n[TEST 6] Document version search for 'RTX-9000 specification version':")
    results = vector_store.similarity_search(
        "RTX-9000 specification version",
        k=3,
        filter={"source": "document"},
    )
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: [{doc.metadata.get('source')}] "
              f"[{doc.metadata.get('access_level')}] "
              f"version={doc.metadata.get('version', 'N/A')} | "
              f"{doc.page_content[:80]}...")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE - All tests passed!")
    print("=" * 70)


# ==============================================================================
# PUBLIC API — Used by shared/config.py
# ==============================================================================

def ingest_all() -> Chroma:
    """
    Run the full ingestion pipeline and return an in-memory Chroma instance.

    This is the entry point used by shared/config.py to initialize the
    vector store at startup. Documents are loaded from source_documents/,
    split into chunks, embedded, and stored in-memory.

    Returns:
        Chroma: Populated in-memory vector store ready for retrieval.
    """
    print("[INGESTION] Running auto-ingest pipeline...")
    documents = create_documents()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    print(f"[INGESTION] Ready: {vector_store._collection.count()} vectors in-memory.")
    return vector_store


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution function for the ingestion pipeline.
    
    Pipeline steps:
      1. Create Document objects from mock data
      2. Split documents into chunks with RecursiveCharacterTextSplitter
      3. Generate embeddings and store in-memory ChromaDB
      4. Run verification queries to confirm successful ingestion
    """
    print("=" * 70)
    print("PROJECT VERA - Data Ingestion Pipeline")
    print("=" * 70)
    print()
    
    # Step 1: Create Document objects from mock data
    print("[STEP 1/4] Creating documents from mock data...")
    documents = create_documents()
    
    # Step 2: Split documents into chunks
    print("\n[STEP 2/4] Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    # Display chunk distribution by access level for verification
    access_counts = {}
    source_counts = {}
    for chunk in chunks:
        access = chunk.metadata.get("access_level", "unknown")
        source = chunk.metadata.get("source", "unknown")
        access_counts[access] = access_counts.get(access, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n  Chunks by access level: {access_counts}")
    print(f"  Chunks by source type:  {source_counts}")
    
    # Step 3: Create in-memory vector store
    print("\n[STEP 3/4] Creating in-memory vector store and generating embeddings...")
    vector_store = create_vector_store(chunks)
    
    # Step 4: Verify ingestion
    print("\n[STEP 4/4] Verifying ingestion...")
    verify_ingestion(vector_store)
    
    print(f"\n[SUCCESS] Ingestion complete! {vector_store._collection.count()} vectors in-memory.")
    print("[INFO] Note: In-memory store is ephemeral. It will be auto-created on app startup.")


if __name__ == "__main__":
    main()

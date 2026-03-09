import os
import re
import glob
import json
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- CONFIGURATION ---
load_dotenv()
from shared.config import get_embeddings

COLLECTION_NAME = "vera_documents"
# Define a persistent path in your project folder
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db") 
SOURCE_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "source_documents")

def sanitize_text(text: str) -> str:
    if not text: return ""
    return re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

# --- FILE TYPE MAPPINGS ---


FILE_TYPE_TO_SOURCE = {
    "Spec": "datasheet",
    "Email": "email",
    "SOP": "sop",
    "DB": "db_info",
    "Data": "dataset",  
}


def _sanitize_clause_label(label: str) -> str:
    """Normalize CUAD labels into compact attribute-style tokens."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (label or "").strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_") or "unknown_clause"


def _extract_cuad_documents(filepath: str, domain: str) -> list[dict]:
    """
    Convert CUAD JSON into synthetic retrieval documents.

    CUAD is contract-annotation data; we transform each contract into a compact
    text artifact containing labeled clause evidence so VERA can retrieve it
    through the normal RAG path.
    """
    docs: list[dict] = []
    filename = os.path.basename(filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"    ❌ Error parsing CUAD JSON: {filename} -> {e}")
        return docs

    contracts = payload.get("data", [])
    if not isinstance(contracts, list):
        print(f"    ❌ Invalid CUAD format: 'data' is not a list in {filename}")
        return docs

    for contract in contracts:
        title = sanitize_text(contract.get("title", "unknown_contract")).strip() or "unknown_contract"
        label_map: dict[str, set[str]] = {}

        for paragraph in contract.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id", "")
                label = qa_id.split("__", 1)[1] if "__" in qa_id else "Unknown"
                label = sanitize_text(label).strip() or "Unknown"
                answers = qa.get("answers", []) or []

                snippets = []
                for ans in answers[:3]:
                    text = sanitize_text(ans.get("text", "")).strip()
                    if text:
                        snippets.append(text[:220])

                if snippets:
                    label_map.setdefault(label, set()).update(snippets)

        if not label_map:
            continue

        clause_lines = []
        for label in sorted(label_map.keys()):
            samples = list(label_map[label])[:3]
            evidence = " | ".join(samples) if samples else "(label present, no answer text)"
            clause_lines.append(
                f"CUAD_LABEL: {label}\n"
                f"CUAD_ATTRIBUTE: {_sanitize_clause_label(label)}\n"
                f"EVIDENCE: {evidence}"
            )

        content = sanitize_text(
            f"LEGAL CONTRACT: {title}\n"
            f"DATASET: CUAD_v1\n"
            "LABELED CLAUSES:\n\n"
            + "\n\n".join(clause_lines)
        )

        metadata = {
            "source": "dataset",
            "version": "1.0",
            "access_level": "public",
            "document_id": title,
            "title": title,
            "domain": domain,
            "dataset_name": "CUAD_v1",
        }
        docs.append({"content": content, "metadata": metadata})

    return docs

def parse_filename(filename: str) -> dict:
    """Parse Domain_Type_Version_Access.ext or provide defaults for CSVs."""
    name, ext = os.path.splitext(filename)
    parts = name.split("_")

    # Special handling for JSON datasets like CUAD
    if ext.lower() == ".json":
        return {
            "source": "dataset",
            "version": "1.0",
            "access_level": "public",
            "document_id": name,
            "title": name.replace("_", " "),
        }

    # Fallback for simple names
    if len(parts) < 3:
        return {
            "source": "dataset" if ext.lower() == ".csv" else "document",
            "version": "1.0",
            "access_level": "public",
            "document_id": name,
            "title": name.replace("_", " "),
        }

    # Standard VERA naming logic: Domain_Type_Version_Access
    source_type = FILE_TYPE_TO_SOURCE.get(parts[1], "document")
    version = parts[2].lstrip("v") if len(parts) > 2 else "1.0"
    access = parts[3].lower() if len(parts) > 3 else "public"
    
    return {
        "source": source_type,
        "version": version,
        "access_level": access,
        "document_id": name,
        "title": name.replace("_", " "),
    }

def load_domain_documents() -> list[dict]:
    all_docs = []
    if not os.path.exists(SOURCE_DOCUMENTS_DIR): return []

    entries = sorted(os.listdir(SOURCE_DOCUMENTS_DIR))
    print(f"[DEBUG] Found entries in {SOURCE_DOCUMENTS_DIR}: {entries}")
    for domain_dir in entries:
        domain_path = os.path.join(SOURCE_DOCUMENTS_DIR, domain_dir)
        if not os.path.isdir(domain_path) or domain_dir.startswith(("_", ".")): continue

        actual_domain = domain_dir.lower() 
        print(f"\n 📂 Processing Domain: {actual_domain}")
        
        # Ingest CSV, PDF, TXT, JSON
        patterns = ["*.csv", "*.pdf", "*.txt", "*.json"]
        all_filepaths = []
        for p in patterns:
            all_filepaths.extend(glob.glob(os.path.join(domain_path, p)))

        for filepath in sorted(all_filepaths):
            filename = os.path.basename(filepath)
            if ":Zone.Identifier" in filename: continue
            
            metadata = parse_filename(filename)
            metadata["domain"] = actual_domain

            try:
                if filename.lower() == "cuad_v1.json" and actual_domain == "legal":
                    cuad_docs = _extract_cuad_documents(filepath, actual_domain)
                    all_docs.extend(cuad_docs)
                    print(f"    ✅ {filename} converted into {len(cuad_docs)} CUAD contract documents")
                    continue

                if filename.endswith(".csv"):
                    df = pd.read_csv(filepath).head(100)
                    content = sanitize_text(df.to_string(index=False))
                else:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = sanitize_text(f.read())
                
                if content:
                    all_docs.append({"content": content, "metadata": metadata})
                    print(f"    ✅ {filename} ingested as {metadata['source']} ({metadata['access_level']})")
            except Exception as e:
                print(f"    ❌ Error: {filename} -> {e}")
    return all_docs


def ingest_all():
    raw_docs = load_domain_documents()
    docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in raw_docs if d["content"].strip()]
    
    if not docs:
        print("[WARNING] No documents found to ingest.")
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    
    if not chunks:
        print("[WARNING] No chunks created from documents.")
        return None

    print(f"\n[INFO] Saving {len(chunks)} chunks to {CHROMA_PATH}...")
    # CRITICAL: persist_directory ensures data stays on disk
    return Chroma.from_documents(
        documents=chunks, 
        embedding=get_embeddings(), 
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )



if __name__ == "__main__":
    ingest_all()

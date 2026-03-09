import os
import re
import glob
import json
import pandas as pd
import mimetypes
from typing import Optional
from dotenv import load_dotenv
import requests

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pypdf import PdfReader

# --- CONFIGURATION ---
load_dotenv()
from shared.config import get_embeddings

COLLECTION_NAME = "vera_documents"
# Define a persistent path in your project folder
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db") 
SOURCE_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "source_documents")
REDUCTO_API_KEY = (
    os.getenv("REDUCTO_API_KEY")
    or os.getenv("REDUCTO_SECRET_KEY")
    or os.getenv("REDUCTO_KEY")
    or ""
).strip()
REDUCTO_PARSE_URL = (
    os.getenv("REDUCTO_PARSE_URL", "https://platform.reducto.ai/api/v1/parse")
).strip()
REDUCTO_ENABLED = bool(REDUCTO_API_KEY and REDUCTO_PARSE_URL)

REDUCTO_SUPPORTED_EXTENSIONS = {".txt", ".png"}
LOCAL_SUPPORTED_EXTENSIONS = {".txt", ".json"}

def sanitize_text(text: str) -> str:
    if not text: return ""
    return re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def _extract_text_from_reducto_response(payload: dict) -> str:
    """
    Best-effort extraction from Reducto parse payload shapes.
    """
    if not isinstance(payload, dict):
        return ""

    for key in ("text", "content", "markdown", "md"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        parts = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            for key in ("text", "content", "markdown", "md"):
                value = chunk.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
        if parts:
            return "\n\n".join(parts)

    pages = payload.get("pages")
    if isinstance(pages, list):
        parts = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            for key in ("text", "content", "markdown", "md"):
                value = page.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
        if parts:
            return "\n\n".join(parts)

    data = payload.get("data")
    if isinstance(data, dict):
        return _extract_text_from_reducto_response(data)

    return ""


def _parse_with_reducto(filepath: str) -> Optional[str]:
    """
    Parse a file with Reducto and return extracted text or None on failure.
    """
    if not REDUCTO_ENABLED:
        return None

    filename = os.path.basename(filepath)
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        with open(filepath, "rb") as f:
            response = requests.post(
                REDUCTO_PARSE_URL,
                headers={"Authorization": f"Bearer {REDUCTO_API_KEY}"},
                files={"file": (filename, f, mime_type)},
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        print(f"    ⚠️ Reducto parse failed for {filename}: {e}")
        return None

    extracted = sanitize_text(_extract_text_from_reducto_response(payload))
    return extracted.strip() or None


def _parse_pdf_locally(filepath: str) -> str:
    """
    Local fallback for PDFs when Reducto is unavailable.
    """
    try:
        reader = PdfReader(filepath)
        pages = [sanitize_text((page.extract_text() or "")) for page in reader.pages]
        return "\n\n".join([p for p in pages if p.strip()])
    except Exception as e:
        print(f"    ⚠️ Local PDF parse failed for {os.path.basename(filepath)}: {e}")
        return ""


def _parse_file_content(filepath: str) -> str:
    """
    Parse supported files for ingestion with selective Reducto usage.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath).head(100)
        return sanitize_text(df.to_string(index=False))

    use_reducto = ext in REDUCTO_SUPPORTED_EXTENSIONS
    if use_reducto:
        parsed = _parse_with_reducto(filepath)
        if parsed:
            return parsed
        if ext == ".png":
            if REDUCTO_ENABLED:
                print(
                    f"    ⚠️ Skipping {os.path.basename(filepath)}: unable to parse PNG via Reducto."
                )
            else:
                print(
                    f"    ⚠️ Skipping {os.path.basename(filepath)}: .png parsing requires Reducto. "
                    "Set REDUCTO_API_KEY to enable PNG processing."
                )
            return ""

    if ext == ".pdf":
        return _parse_pdf_locally(filepath)

    if ext in {".doc", ".docx"}:
        print(
            f"    ⚠️ Skipping {os.path.basename(filepath)}: "
            f"{ext} requires Reducto parsing. Set REDUCTO_API_KEY to enable."
        )
        return ""

    if ext in LOCAL_SUPPORTED_EXTENSIONS:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sanitize_text(f.read())

    return ""

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
        
        # Ingest CSV, PDF, TXT/JSON, Word docs, PNG (Reducto-only)
        patterns = ["*.csv", "*.pdf", "*.txt", "*.json", "*.doc", "*.docx", "*.png"]
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

                content = _parse_file_content(filepath)
                
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

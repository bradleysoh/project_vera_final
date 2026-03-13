import os
import re
import glob
import json
import hashlib
import time
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
from shared.advanced_rag import perform_llm_fact_extraction

COLLECTION_NAME = "vera_documents"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db") 
SOURCE_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "source_documents")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "ingestion_manifest.json")

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

# ============================================================================
# UTILITIES: Sanitization & Parsing
# ============================================================================

def sanitize_text(text: str) -> str:
    if not text: return ""
    return re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def _extract_text_from_reducto_response(payload: dict) -> str:
    if not isinstance(payload, dict): return ""
    for key in ("text", "content", "markdown", "md"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip(): return value
    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        parts = []
        for chunk in chunks:
            if not isinstance(chunk, dict): continue
            for key in ("text", "content", "markdown", "md"):
                value = chunk.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
        if parts: return "\n\n".join(parts)
    return ""

def _parse_with_reducto(filepath: str) -> Optional[str]:
    if not REDUCTO_ENABLED: return None
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
    try:
        reader = PdfReader(filepath)
        pages = [sanitize_text((page.extract_text() or "")) for page in reader.pages]
        return "\n\n".join([p for p in pages if p.strip()])
    except Exception as e:
        print(f"    ⚠️ Local PDF parse failed for {os.path.basename(filepath)}: {e}")
        return ""

def _parse_file_content(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath).head(100)
        return sanitize_text(df.to_string(index=False))
    
    use_reducto = ext in REDUCTO_SUPPORTED_EXTENSIONS
    if use_reducto:
        parsed = _parse_with_reducto(filepath)
        if parsed: return parsed
        if ext == ".png": return ""
        
    if ext == ".pdf": return _parse_pdf_locally(filepath)
    
    if ext in LOCAL_SUPPORTED_EXTENSIONS:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sanitize_text(f.read())
    return ""

# ============================================================================
# FILE LOGIC
# ============================================================================

FILE_TYPE_TO_SOURCE = {
    "Spec": "datasheet", "Email": "email", "SOP": "sop", 
    "DB": "db_info", "Data": "dataset",  
}

def _sanitize_clause_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (label or "").strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_") or "unknown_clause"

def _extract_cuad_documents(filepath: str, domain: str) -> list[dict]:
    # (保留原有逻辑，精简省略展示)
    docs: list[dict] = []
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            payload = json.load(f)
    except: return docs
    
    contracts = payload.get("data", [])
    for contract in contracts:
        title = sanitize_text(contract.get("title", "unknown_contract")).strip() or "unknown_contract"
        label_map: dict[str, set[str]] = {}
        for paragraph in contract.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id", "")
                label = sanitize_text(qa_id.split("__", 1)[1] if "__" in qa_id else "Unknown").strip() or "Unknown"
                for ans in (qa.get("answers", []) or [])[:3]:
                    text = sanitize_text(ans.get("text", "")).strip()
                    if text: label_map.setdefault(label, set()).update([text[:220]])
        
        if not label_map: continue
        clause_lines = []
        for label in sorted(label_map.keys()):
            samples = list(label_map[label])[:3]
            evidence = " | ".join(samples) if samples else "(label present, no answer text)"
            clause_lines.append(f"CUAD_LABEL: {label}\nCUAD_ATTRIBUTE: {_sanitize_clause_label(label)}\nEVIDENCE: {evidence}")
        
        content = sanitize_text(f"LEGAL CONTRACT: {title}\nDATASET: CUAD_v1\nLABELED CLAUSES:\n\n" + "\n\n".join(clause_lines))
        metadata = {
            "source": "dataset", "version": "1.0", "access_level": "public",
            "document_id": title, "title": title, "domain": domain, "dataset_name": "CUAD_v1"
        }
        docs.append({"content": content, "metadata": metadata})
    return docs

def parse_filename(filename: str) -> dict:
    name, ext = os.path.splitext(filename)
    parts = name.split("_")
    if ext.lower() == ".json": return {"source": "dataset", "version": "1.0", "access_level": "public", "document_id": name, "title": name.replace("_", " ")}
    if len(parts) < 3: return {"source": "dataset" if ext.lower() == ".csv" else "document", "version": "1.0", "access_level": "public", "document_id": name, "title": name.replace("_", " ")}
    return {
        "source": FILE_TYPE_TO_SOURCE.get(parts[1], "document"),
        "version": parts[2].lstrip("v") if len(parts) > 2 else "1.0",
        "access_level": parts[3].lower() if len(parts) > 3 else "public",
        "document_id": name,
        "title": name.replace("_", " "),
    }

def load_domain_documents() -> list[dict]:
    all_docs = []
    if not os.path.exists(SOURCE_DOCUMENTS_DIR): return []
    entries = sorted(os.listdir(SOURCE_DOCUMENTS_DIR))
    for domain_dir in entries:
        domain_path = os.path.join(SOURCE_DOCUMENTS_DIR, domain_dir)
        if not os.path.isdir(domain_path) or domain_dir.startswith(("_", ".")): continue
        actual_domain = domain_dir.lower() 
        patterns = ["*.csv", "*.pdf", "*.txt", "*.json", "*.doc", "*.docx", "*.png"]
        all_filepaths = []
        for p in patterns: all_filepaths.extend(glob.glob(os.path.join(domain_path, p)))
        
        for filepath in sorted(all_filepaths):
            filename = os.path.basename(filepath)
            if ":Zone.Identifier" in filename: continue
            metadata = parse_filename(filename)
            metadata["domain"] = actual_domain
            try:
                if filename.lower() == "cuad_v1.json" and actual_domain == "legal":
                    cuad_docs = _extract_cuad_documents(filepath, actual_domain)
                    all_docs.extend(cuad_docs)
                    continue
                content = _parse_file_content(filepath)
                if content: all_docs.append({"content": content, "metadata": metadata})
            except Exception as e:
                print(f"    ❌ Error: {filename} -> {e}")
    return all_docs

# ============================================================================
# STATEFUL INGESTION ENGINE
# ============================================================================

def _compute_hash(content: str) -> str:
    """计算文本的 SHA-256 指纹"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f: return json.load(f)
        except: return {}
    return {}

def _save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w") as f: json.dump(manifest, f, indent=4)

def ingest_all():
    # 延迟导入以防止循环依赖
    try:
        from shared.fact_store import store
        from shared.schemas import ExtractedFact
    except ImportError:
        print("[ERROR] Failed to import fact_store. Did you implement Phase 1 Schema?")
        return

    raw_docs = load_domain_documents()
    if not raw_docs:
        print("[WARNING] No documents found to ingest.")
        return

    manifest = _load_manifest()
    docs_to_extract = []
    docs_for_chroma = []
    
    print("\n[PHASE 0] Scanning for modifications (Hash Fingerprinting)...")
    for d in raw_docs:
        doc_id = d["metadata"].get("document_id", "unknown_doc")
        content_hash = _compute_hash(d["content"])
        
        doc = Document(page_content=d["content"], metadata=d["metadata"])
        docs_for_chroma.append(doc)

        # Skip fact extraction if hash matches
        if manifest.get(doc_id) == content_hash:
            continue
            
        d["hash"] = content_hash
        docs_to_extract.append(d)

    if docs_to_extract:
        print(f"\n[PHASE 1] Shift-Left: Extracting facts for {len(docs_to_extract)} NEW/MODIFIED documents...")
        for d in docs_to_extract:
            doc_id = d["metadata"].get("document_id", "unknown")
            print(f"    🔍 [LLM Extraction] Processing: {doc_id}...")
            try:
                doc_obj = Document(page_content=d["content"], metadata=d["metadata"])
                facts_data = perform_llm_fact_extraction(documents=[doc_obj], target_entity="GENERAL", target_attribute="GENERAL")
                
                facts = []
                for f_dict in facts_data:
                    try: facts.append(ExtractedFact(**f_dict))
                    except Exception: continue
                
                store.save_facts(doc_id, facts)
                manifest[doc_id] = d["hash"]
                _save_manifest(manifest)
                print(f"    ✅ Stored {len(facts)} facts for {doc_id}")
            except Exception as e:
                print(f"    ❌ Fact extraction failed for {doc_id}: {str(e)}")
    else:
        print("    ✅ All facts are up to date. Skipping Phase 1.")

    print("\n[PHASE 2] Vector DB Upsert...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs_for_chroma)
    chunks = [c for c in chunks if c.page_content.strip()]
    
    if chunks:
        # 生成确定性 ID 触发 Upsert，防止重复污染
        chunk_ids = [f"{c.metadata.get('document_id', 'doc')}_chunk_{i}" for i, c in enumerate(chunks)]
        
        print(f"[INFO] Initializing ChromaDB for upsert ({CHROMA_PATH})...")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME, 
            persist_directory=CHROMA_PATH, 
            embedding_function=get_embeddings()
        )
        
        # ── Batching Upsert ────────────────────────────────────────────────
        batch_size = 500
        n_chunks = len(chunks)
        print(f"[INFO] Upserting {n_chunks} chunks in batches of {batch_size}...")
        
        for i in range(0, n_chunks, batch_size):
            batch_end = min(i + batch_size, n_chunks)
            batch_chunks = chunks[i:batch_end]
            batch_ids = chunk_ids[i:batch_end]
            
            vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
            print(f"    ✅ Upserted chunks {i} to {batch_end} ({batch_end/n_chunks*100:.1f}%)")
            
        print("    ✅ Vector DB updated successfully.")

if __name__ == "__main__":
    ingest_all()
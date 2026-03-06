import os
import re
import json
import glob
import pandas as pd  # Critical: Added for Kaggle CSV support
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# --- CONFIGURATION ---
load_dotenv()
from shared.config import LLM_BACKEND, get_embeddings

COLLECTION_NAME = "vera_documents"
SOURCE_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "source_documents")

# --- FILE TYPE MAPPINGS ---
FILE_TYPE_TO_SOURCE = {
    "Spec": "datasheet",
    "Email": "email",
    "SOP": "sop",
    "DB": "db_info",
    "Data": "dataset",  # New mapping for CSV data
}

FILE_TYPE_DEFAULT_RBAC = {
    "Email": "internal_only",
    "Data": "public",    # Default Kaggle CSVs to public
}

def sanitize_text(text: str) -> str:
    if not text: return ""
    return re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def parse_filename(filename: str) -> dict:
    """Parse Domain_Type_Version_Access.ext or provide defaults for CSVs."""
    name, ext = os.path.splitext(filename)
    parts = name.split("_")

    # Fallback for simple CSV names (e.g., "patient_data.csv")
    if len(parts) < 4:
        return {
            "domain": "general",
            "source": "dataset" if ext.lower() == ".csv" else "document",
            "version": "1.0",
            "access_level": "public",
            "document_id": name,
            "title": name.replace("_", " "),
        }

    # Standard VERA naming logic
    return {
        "domain": parts[0].lower(),
        "source": FILE_TYPE_TO_SOURCE.get(parts[1], "document"),
        "version": parts[2].lstrip("v"),
        "access_level": parts[3].lower() if len(parts) > 3 else "public",
        "document_id": name,
        "title": name.replace("_", " "),
    }

def load_domain_documents() -> list[dict]:
    all_docs = []
    if not os.path.exists(SOURCE_DOCUMENTS_DIR): return []

    for domain_dir in sorted(os.listdir(SOURCE_DOCUMENTS_DIR)):
        domain_path = os.path.join(SOURCE_DOCUMENTS_DIR, domain_dir)
        if not os.path.isdir(domain_path) or domain_dir.startswith(("_", ".")): continue

        print(f"\n  📂 Domain: {domain_dir}/")
        
        # Support for .txt, .pdf, and now .csv
        patterns = ["*.txt", "*.pdf", "*.csv"]
        all_filepaths = []
        for p in patterns:
            all_filepaths.extend(glob.glob(os.path.join(domain_path, p)))
        
        for filepath in sorted(all_filepaths):
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            parsed = parse_filename(filename)

            try:
                if ext == ".csv":
                    # Convert CSV rows to a readable string for the LLM
                    df = pd.read_csv(filepath).head(50) 
                    content = sanitize_text(df.to_string(index=False))
                elif ext == ".pdf":
                    loader = PyPDFLoader(filepath)
                    content = sanitize_text("\n".join([d.page_content for d in loader.load()]))
                else:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = sanitize_text(f.read())
                
                if content:
                    parsed["domain"] = domain_dir
                    all_docs.append({"content": content, "metadata": parsed})
                    print(f"    ✅ {filename} ingested")
            except Exception as e:
                print(f"    ❌ Error: {filename} -> {e}")

    return all_docs

# --- PIPELINE WRAPPERS ---
def ingest_all() -> Chroma:
    raw_docs = load_domain_documents()
    docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in raw_docs]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    print(f"[INFO] Vector store created with {len(chunks)} chunks.")
    return Chroma.from_documents(documents=chunks, embedding=get_embeddings(), collection_name=COLLECTION_NAME)

if __name__ == "__main__":
    ingest_all()
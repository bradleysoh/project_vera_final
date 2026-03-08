import os
import re
import glob
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

def load_domain_documents() -> list[dict]:
    all_docs = []
    if not os.path.exists(SOURCE_DOCUMENTS_DIR): return []

    for domain_dir in sorted(os.listdir(SOURCE_DOCUMENTS_DIR)):
        domain_path = os.path.join(SOURCE_DOCUMENTS_DIR, domain_dir)
        if not os.path.isdir(domain_path) or domain_dir.startswith(("_", ".")): continue

        actual_domain = domain_dir.lower() 
        print(f"\n 📂 Processing Domain: {actual_domain}")
        
        # Ingest CSV, PDF, TXT
        patterns = ["*.csv", "*.pdf", "*.txt"]
        all_filepaths = []
        for p in patterns:
            all_filepaths.extend(glob.glob(os.path.join(domain_path, p)))
        
        for filepath in sorted(all_filepaths):
            filename = os.path.basename(filepath)
            if ":Zone.Identifier" in filename: continue

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(filepath).head(100)
                    content = sanitize_text(df.to_string(index=False))
                else:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = sanitize_text(f.read())
                
                if content:
                    # FORCE correct domain metadata
                    metadata = {"domain": actual_domain, "source": filename}
                    all_docs.append({"content": content, "metadata": metadata})
                    print(f"    ✅ {filename} tagged as: {actual_domain}")
            except Exception as e:
                print(f"    ❌ Error: {filename} -> {e}")
    return all_docs

def ingest_all():
    raw_docs = load_domain_documents()
    docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in raw_docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
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
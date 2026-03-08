import os
from langchain_chroma import Chroma
from shared.config import get_embeddings

COLLECTION_NAME = "vera_documents"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

def inspect_chroma():
    print(f"--- Inspecting Chroma at {CHROMA_PATH} ---")
    if not os.path.exists(CHROMA_PATH):
        print("❌ Chroma path does not exist.")
        return

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

    # Get all documents
    all_docs = vs.get()
    metadatas = all_docs.get("metadatas", [])
    print(f"Total chunks in DB: {len(metadatas)}")

    if not metadatas:
        print("❌ No metadata found.")
        return

    # Count by domain
    domain_counts = {}
    for meta in metadatas:
        domain = meta.get("domain", "UNKNOWN")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print("\n--- Domain Distribution ---")
    for domain, count in sorted(domain_counts.items()):
        print(f"  - {domain}: {count} chunks")

    # Sample check for potential bleed
    print("\n--- Sample Metadata (first 5) ---")
    for i, meta in enumerate(metadatas[:5]):
        print(f"  {i+1}. {meta}")

if __name__ == "__main__":
    inspect_chroma()

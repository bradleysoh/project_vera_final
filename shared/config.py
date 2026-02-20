"""
================================================================================
Project VERA — Shared Configuration & Resources
================================================================================

Provides centralized LLM factory, embedding factory, RBAC retriever, and
retry logic. All agents use `config.llm` for the active LLM instance.

    import shared.config as config
    chain = prompt | config.llm | StrOutputParser()

Factory functions (for custom initialization):
    from shared.config import get_llm, get_embeddings

DO NOT re-initialize the singleton — use config.llm directly.
================================================================================
"""

import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

# ==============================================================================
# ENVIRONMENT VARIABLE READS
# ==============================================================================

# --- Backend selection ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini").strip().lower()

# --- API Keys ---
# GOOGLE_API_KEY is the canonical name; GEMINI_API_KEY is a legacy alias
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Model names ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "").strip()
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "").strip().lower()

# --- Ollama settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
)
OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL", "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
)

# --- Email alerting ---
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
_raw_recipients = os.getenv("EMAIL_RECIPIENTS", "") or os.getenv("SUPERVISOR_EMAIL", "")
EMAIL_RECIPIENTS = [r.strip() for r in _raw_recipients.split(",") if r.strip()]

# --- Retrieval mode ---
# "fast" = direct retrieval (3 LLM calls, best for Ollama)
# "deep" = multi-query + reranking (8-9 LLM calls, best for Gemini/Groq)
_retrieval_env = os.getenv("RETRIEVAL_MODE", "").strip().lower()
if _retrieval_env in ("fast", "deep"):
    RETRIEVAL_MODE = _retrieval_env
else:
    # Auto-detect: fast for Ollama, deep for cloud APIs
    RETRIEVAL_MODE = "fast" if LLM_BACKEND == "ollama" else "deep"

print(f"[CONFIG] Using LLM backend: {LLM_BACKEND.upper()}")
print(f"[CONFIG] Retrieval mode: {RETRIEVAL_MODE.upper()}")

# Retry settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 60  # seconds


# ==============================================================================
# LLM RETRY WRAPPER
# ==============================================================================

def llm_invoke_with_retry(chain, inputs: dict, retries: int = MAX_RETRIES) -> str:
    """
    Invoke an LLM chain with exponential backoff retry for rate limits.

    Args:
        chain: A LangChain chain (prompt | llm | parser) to invoke.
        inputs: Dictionary of inputs for the chain.
        retries: Maximum number of retry attempts.

    Returns:
        str: The chain output string.
    """
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(
                    f"  [RATE LIMIT] Hit quota limit. Waiting {delay}s "
                    f"before retry ({attempt + 1}/{retries})..."
                )
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Failed after {retries} retries due to rate limiting.")


# ==============================================================================
# LLM FACTORY — get_llm()
# ==============================================================================

# Default model names per backend
_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": OLLAMA_MODEL,
}


def get_llm(backend: str = ""):
    """
    Factory function: create and return a LangChain chat model.

    Args:
        backend: 'gemini', 'groq', or 'ollama'.
                 Defaults to LLM_BACKEND env var.

    Returns:
        A LangChain BaseChatModel instance.
    """
    backend = (backend or LLM_BACKEND).strip().lower()
    model_name = LLM_MODEL_NAME or _DEFAULT_MODELS.get(backend, "")

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        print(f"[LLM] Using Ollama: {model_name}")
        return ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            timeout=120,
        )

    elif backend == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in .env or use a different backend."
            )
        from langchain_groq import ChatGroq
        print(f"[LLM] Using Groq: {model_name}")
        return ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature=0.1,
        )

    else:  # gemini (default)
        if not GEMINI_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in .env or use a different backend."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        print(f"[LLM] Using Gemini: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True,
        )


# Legacy alias for backward compatibility
initialize_llm = get_llm


# ==============================================================================
# EMBEDDING FACTORY — get_embeddings()
# ==============================================================================

def get_embeddings(model_type: str = ""):
    """
    Factory function: create and return a LangChain embedding model.

    Args:
        model_type: 'gemini', 'huggingface', or 'ollama'.
                    Defaults to EMBEDDING_MODEL_TYPE env var, then falls back
                    to a sensible default based on LLM_BACKEND.

    Returns:
        A LangChain Embeddings instance.
    """
    # Resolve model type: explicit env var > LLM backend fallback
    mtype = (model_type or EMBEDDING_MODEL_TYPE).strip().lower()
    if not mtype:
        # Fallback: match LLM backend (groq has no embeddings → use ollama)
        mtype = "ollama" if LLM_BACKEND in ("ollama", "groq") else "gemini"

    if mtype == "ollama":
        from langchain_ollama import OllamaEmbeddings
        print(f"[EMBEDDINGS] Using Ollama: {OLLAMA_EMBED_MODEL}")
        return OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

    elif mtype == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        hf_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"[EMBEDDINGS] Using HuggingFace: {hf_model}")
        return HuggingFaceEmbeddings(model_name=hf_model)

    else:  # gemini (default)
        if not GEMINI_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in .env for Gemini embeddings."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("[EMBEDDINGS] Using Gemini: gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY,
        )


# ==============================================================================
# RUNTIME BACKEND SWITCHING
# ==============================================================================

_current_backend = LLM_BACKEND


def get_current_backend() -> str:
    """Return the currently active LLM backend name."""
    return _current_backend


def switch_backend(backend: str):
    """
    Switch the LLM backend at runtime.

    Updates the module-level `llm` variable so all agents using
    `config.llm` will immediately use the new backend.

    Args:
        backend: 'gemini', 'groq', or 'ollama'
    """
    global llm, _current_backend
    backend = backend.strip().lower()
    if backend == _current_backend:
        print(f"[CONFIG] Already using {backend.upper()}, no change needed.")
        return
    print(
        f"[CONFIG] 🔄 Switching LLM backend: "
        f"{_current_backend.upper()} → {backend.upper()}"
    )
    llm = get_llm(backend)
    _current_backend = backend
    print(f"[CONFIG] ✅ Backend switched to {backend.upper()}")


# ==============================================================================
# RBAC CONFIGURATION
# ==============================================================================

ROLE_ACCESS_MAP = {
    "senior": ["public", "internal_only", "confidential", "supervisor"],
    "junior": ["public"],
}


def get_available_roles() -> list[str]:
    """Return a list of available user roles for RBAC."""
    return list(ROLE_ACCESS_MAP.keys())


# ==============================================================================
# DOMAIN PROMPT CONFIGURATION (Dynamic)
# ==============================================================================

DOMAIN_KEYWORDS = {
    "semiconductor": "chips, voltage, thermal, RTX, silicon, wafer, die, yield, burn-in",
    "medical": "clinical, FDA, patient, trial, device approval, safety class",
}

# ==============================================================================
# RBAC-AWARE RETRIEVER
# ==============================================================================

def retrieve_with_rbac(
    query: str,
    user_role: str,
    user_domain: str = "",
    source_filter: Optional[list[str]] = None,
    k: int = 4,
) -> tuple[List[Document], str]:
    """
    Retrieve documents from ChromaDB with Role-Based Access Control (RBAC)
    and domain isolation.

    Double filter:
      - Domain filter: restricts results to the user's assigned domain
      - Access filter: restricts by role-based access level

    Args:
        query: The search query string.
        user_role: The user's role ("senior" or "junior").
        user_domain: The user's domain ("semiconductor", "medical", etc.).
        source_filter: Optional list of source types to filter by.
        k: Number of results to return (default: 4).

    Returns:
        tuple: (list of retrieved Documents, metadata log string)
    """
    metadata_log = f"[RBAC] User role: {user_role} | Domain: {user_domain} | Query: '{query}'\n"

    # --- Build access level filter based on role ---
    allowed_levels = ROLE_ACCESS_MAP.get(user_role, ["public"])
    metadata_log += f"[RBAC] Allowed access levels: {allowed_levels}\n"

    # --- Build composite filter ---
    conditions = []

    # Access level filter (always applied)
    if len(allowed_levels) == 1:
        conditions.append({"access_level": allowed_levels[0]})
    else:
        conditions.append({"access_level": {"$in": allowed_levels}})

    # Domain filter (applied when user_domain is specified)
    if user_domain:
        conditions.append({"domain": user_domain})
        metadata_log += f"[RBAC] Domain filter: {user_domain}\n"

    # Source type filter (applied when specified by the agent)
    if source_filter:
        conditions.append({"source": {"$in": source_filter}})
        metadata_log += f"[RBAC] Source filter: {source_filter}\n"

    # Combine conditions
    if len(conditions) == 1:
        filter_conditions = conditions[0]
    else:
        filter_conditions = {"$and": conditions}

    # --- Execute retrieval ---
    if filter_conditions:
        results = vector_store.similarity_search(query, k=k, filter=filter_conditions)
    else:
        results = vector_store.similarity_search(query, k=k)

    metadata_log += f"[RBAC] Retrieved {len(results)} documents\n"

    for i, doc in enumerate(results):
        metadata_log += (
            f"  Doc {i+1}: source={doc.metadata.get('source')}, "
            f"access={doc.metadata.get('access_level')}, "
            f"domain={doc.metadata.get('domain')}, "
            f"id={doc.metadata.get('document_id')}\n"
        )

    return results, metadata_log


# ==============================================================================
# MULTI-QUERY RETRIEVER  (Ensemble / Query-Rewriting Strategy)
# ==============================================================================

def multi_query_retrieve(
    query: str,
    user_role: str,
    user_domain: str = "",
    source_filter: Optional[list[str]] = None,
    k: int = 4,
    n_variations: int = 3,
) -> tuple[List[Document], str]:
    """
    Generate *n_variations* rewrites of *query* via the LLM, run
    ``retrieve_with_rbac`` for each, and merge/deduplicate the results.

    This ensures no relevant chunks are missed because of embedding-similarity
    bias toward a single phrasing.

    Returns:
        tuple: (deduplicated Document list, combined metadata log)
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import re as _re

    # --- Step 1: Generate query variations ---
    variation_prompt = ChatPromptTemplate.from_messages([
        ("human", (
            "Generate {n} different phrasings of this question. "
            "Each phrasing should emphasise different keywords or angles "
            "so that a vector search finds ALL relevant documents.\n\n"
            "Original question: {question}\n\n"
            "Return ONLY the numbered list (1. … 2. … 3. …), no commentary."
        ))
    ])

    chain = variation_prompt | llm | StrOutputParser()
    try:
        raw = llm_invoke_with_retry(chain, {
            "n": str(n_variations),
            "question": query,
        })
        # Parse numbered lines
        variations = _re.findall(r"\d+\.\s*(.+)", raw)
        if not variations:
            variations = [query]
    except Exception as e:
        print(f"[MULTI-QUERY] ⚠️ Variation generation failed ({e}), using original query.")
        variations = [query]

    # Always include the original query
    all_queries = [query] + [v.strip() for v in variations if v.strip() != query]
    all_queries = all_queries[: n_variations + 1]  # cap total

    metadata_log = f"[MULTI-QUERY] Generated {len(all_queries)} query variations:\n"
    for i, q in enumerate(all_queries):
        metadata_log += f"  Q{i+1}: {q}\n"

    # --- Step 2: Retrieve for each variation ---
    seen_contents: set[str] = set()
    merged_docs: List[Document] = []
    combined_log = metadata_log

    for q in all_queries:
        docs, log = retrieve_with_rbac(
            query=q,
            user_role=user_role,
            user_domain=user_domain,
            source_filter=source_filter,
            k=k,
        )
        combined_log += log
        for doc in docs:
            key = doc.page_content[:120]
            if key not in seen_contents:
                seen_contents.add(key)
                merged_docs.append(doc)

    combined_log += f"[MULTI-QUERY] Total unique docs after merge: {len(merged_docs)}\n"
    print(f"[MULTI-QUERY] {len(all_queries)} queries → {len(merged_docs)} unique docs")

    return merged_docs, combined_log


# ==============================================================================
# LLM RERANKER
# ==============================================================================

def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 6,
) -> List[Document]:
    """
    Use the LLM to score each document's relevance to *query* (0-10),
    then return the *top_n* highest-scored documents in descending order.
    """
    if len(documents) <= top_n:
        return documents  # nothing to prune

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import re as _re

    # Build a concise summary of each doc for the LLM
    doc_summaries = []
    for i, doc in enumerate(documents):
        src = doc.metadata.get("source", "unknown")
        doc_summaries.append(f"[{i}] ({src}) {doc.page_content[:300]}")

    docs_block = "\n".join(doc_summaries)

    rerank_prompt = ChatPromptTemplate.from_messages([
        ("human", (
            "Rate each document's relevance to the QUESTION on a 0-10 scale.\n\n"
            "QUESTION: {question}\n\n"
            "DOCUMENTS:\n{docs}\n\n"
            "Return ONLY lines like: [index] score\n"
            "Example:\n[0] 8\n[1] 3\n"
        ))
    ])

    chain = rerank_prompt | llm | StrOutputParser()
    try:
        raw = llm_invoke_with_retry(chain, {
            "question": query,
            "docs": docs_block,
        })
        scores = {}
        for m in _re.finditer(r"\[(\d+)\]\s*(\d+)", raw):
            idx, score = int(m.group(1)), int(m.group(2))
            if 0 <= idx < len(documents):
                scores[idx] = score

        # Sort by score descending, keep top_n
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        reranked = [documents[idx] for idx, _ in ranked[:top_n]]
        print(f"[RERANK] Scored {len(scores)} docs, returning top {len(reranked)}")
        return reranked

    except Exception as e:
        print(f"[RERANK] ⚠️ Reranking failed ({e}), returning original order.")
        return documents[:top_n]


# ==============================================================================
# MODULE-LEVEL SINGLETONS (initialized once on import)
# ==============================================================================

llm = get_llm()

# Vector store: In-memory ChromaDB populated at startup via ingestion pipeline
from ingestion import ingest_all  # noqa: E402
vector_store = ingest_all()

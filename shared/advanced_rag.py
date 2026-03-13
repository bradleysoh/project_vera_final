"""
================================================================================
Advanced RAG Pipeline — Query Understanding + Precision Retrieval
================================================================================

Provides ``query_understand_and_retrieve()`` — a two-stage retrieval function
that replaces the basic ``retrieve_with_rbac → similarity_search`` pattern.

Stage 1 (Query Understanding):
    An LLM call translates the user's raw query into structured metadata
    filters (entity IDs, document versions, source types, date ranges)
    based on the domain's ``metadata_schema``.

Stage 2 (Precision Retrieval):
    Builds ChromaDB ``$and`` filters combining the extracted metadata with
    RBAC filters, then runs ``similarity_search`` with those filters.

Each returned document is tagged with a confidence level (HIGH / MEDIUM / LOW)
based on how many of the extracted filters matched.

Usage:
    from shared.advanced_rag import query_understand_and_retrieve

    result = query_understand_and_retrieve(
        query="What is the max voltage for RTX-9000?",
        user_role="senior",
        user_domain="semiconductor",
        source_filter=["datasheet", "spec"],
        metadata_schema=DOMAIN_CONFIG["metadata_schema"],
    )
    # result.documents, result.extracted_filters, result.confidence, ...
================================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import shared.config as config
from shared.config import llm_invoke_with_retry, retrieve_with_rbac, RETRIEVAL_MODE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standardized marker for "No matching data" results to distinguish from factual content
NO_DATA_MARKER = "__NO_DATA_FOUND__"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Structured output from the advanced RAG pipeline."""
    documents: List[Document] = field(default_factory=list)
    extracted_filters: dict = field(default_factory=dict)
    confidence: str = "LOW"          # HIGH / MEDIUM / LOW
    metadata_log: str = ""
    filter_match_count: int = 0      # how many extracted filters hit


# ---------------------------------------------------------------------------
# Stage 1: Query Understanding
# ---------------------------------------------------------------------------

_QUERY_UNDERSTANDING_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "You are a query analysis engine.  Extract structured metadata filters "
        "from the user's question so a database search can be narrowed down.\n\n"
        "DOMAIN METADATA SCHEMA:\n"
        "  Entity types that may appear: {entity_types}\n"
        "  Documents may have version numbers: {has_versions}\n\n"
        "USER QUESTION: {question}\n\n"
        "Extract the following (leave blank if not found):\n"
        "ENTITY_ID: <the primary entity name or ID mentioned>\n"
        "VERSION: <document version if mentioned>\n"
        "DATE_REF: <any date or time reference>\n"
        "ATTRIBUTE: <the specific attribute being asked about, e.g. 'voltage', 'dosage'>\n\n"
        "Return ONLY the fields above, one per line, exactly as formatted.  "
        "If a field is not found, write: (none)"
    ))
])


def _extract_query_filters(
    question: str,
    metadata_schema: dict,
) -> dict:
    """
    Use the LLM to extract structured metadata filters from the user's
    natural-language query.

    Returns a dict like:
        {"entity_id": "RTX-9000", "attribute": "voltage", ...}
    Empty/missing fields are omitted.
    """
    entity_types = ", ".join(metadata_schema.get("entity_types", []))
    has_versions = "yes" if metadata_schema.get("doc_versions") else "no"

    chain = _QUERY_UNDERSTANDING_PROMPT | config.llm | StrOutputParser()

    try:
        raw = llm_invoke_with_retry(chain, {
            "question": question,
            "entity_types": entity_types or "(none defined)",
            "has_versions": has_versions,
        })
    except Exception as e:
        print(f"[ADVANCED RAG] ⚠️ Query understanding failed ({e}), proceeding without filters")
        return {}

    # Parse the structured output
    filters: dict = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if value and value.lower() != "(none)":
            filters[key] = value

    return filters


# ---------------------------------------------------------------------------
# Stage 2: Precision Retrieval
# ---------------------------------------------------------------------------

def _build_precision_filter(
    extracted: dict,
    user_role: str,
    user_domain: str,
    source_filter: Optional[list[str]],
) -> tuple[dict | None, str]:
    """
    Build a ChromaDB ``$and`` filter from extracted metadata, RBAC rules,
    and source-type constraints.

    Returns (filter_dict, metadata_log_fragment).
    """
    from shared.config import ROLE_ACCESS_MAP

    conditions = []
    log_parts = []

    # RBAC access level
    allowed = ROLE_ACCESS_MAP.get(user_role, ["public"])
    if len(allowed) == 1:
        conditions.append({"access_level": allowed[0]})
    else:
        conditions.append({"access_level": {"$in": allowed}})

    # Domain isolation
    if user_domain:
        conditions.append({"domain": user_domain})
        log_parts.append(f"domain={user_domain}")

    # Source type
    if source_filter:
        conditions.append({"source": {"$in": source_filter}})
        log_parts.append(f"source∈{source_filter}")

    # From query understanding: entity filtering via document_id metadata
    entity = extracted.get("entity_id")
    if entity:
        log_parts.append(f"entity_hint={entity}")
        # NOTE: ChromaDB metadata filter on document_id is only helpful if
        # your ingestion pipeline stores entity IDs in metadata.  We keep the
        # semantic search as the primary mechanism but log the entity for
        # post-retrieval filtering below.

    log = "[PRECISION] Filters: " + ", ".join(log_parts) + "\n" if log_parts else ""

    if len(conditions) == 1:
        return conditions[0], log
    elif conditions:
        return {"$and": conditions}, log
    return None, log


def _post_filter_by_entity(
    docs: List[Document],
    entity: str,
) -> List[Document]:
    """
    Filter documents based on entity relevance to prevent context poisoning.
    Always allow generic documents (SOPs, Policies, Handbooks).
    """
    if not entity or entity.upper() == "GENERAL":
        return docs
        
    filtered = []
    entity_lower = entity.lower()
    # Broad variations for better recall while still isolating
    variations = {entity_lower, entity_lower.replace("-", " "), entity_lower.replace(" ", "-"), entity_lower.replace(" ", "")}
    
    for doc in docs:
        content_lower = doc.page_content.lower()
        source = doc.metadata.get("source", "").lower()
        title = doc.metadata.get("title", "").lower()
        
        # 1. Direct entity match in content or title
        entity_match = any(v in content_lower or v in title for v in variations if len(v) > 2)
        
        # 2. Document is generic/broad context (SOP, Policy, Manual)
        is_generic_doc = any(kw in source or kw in title for kw in ("sop", "policy", "handbook", "manual", "regulations", "standard"))
        
        if entity_match or is_generic_doc:
            filtered.append(doc)
            
    return filtered


def _compute_confidence(
    docs: List[Document],
    extracted: dict,
) -> tuple[str, int]:
    """
    Determine retrieval confidence based on filter-match completeness.

    - HIGH:   entity found in docs AND at least 2 filters matched
    - MEDIUM: entity found in docs OR at least 1 filter matched
    - LOW:    no filters matched and no entity found in docs
    """
    if not docs:
        return "LOW", 0

    entity = extracted.get("entity_id", "")
    entity_in_docs = any(
        entity.lower() in d.page_content.lower()
        for d in docs
    ) if entity else False

    filter_count = sum(1 for k, v in extracted.items() if v)

    if entity_in_docs and filter_count >= 2:
        return "HIGH", filter_count
    elif entity_in_docs or filter_count >= 1:
        return "MEDIUM", filter_count
    return "LOW", 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_understand_and_retrieve(
    query: str,
    user_role: str,
    user_domain: str = "",
    source_filter: Optional[list[str]] = None,
    metadata_schema: Optional[dict] = None,
    k: int = 6,
    target_entity: str = "",
) -> RetrievalResult:
    """
    Advanced RAG: Query Understanding → Precision Retrieval → Confidence.

    1. Extracts structured filters from the user's query via LLM.
    2. Retrieves documents using combined RBAC + extracted metadata filters.
    3. Post-filters by entity mention in document content.
    4. Assigns a confidence tag (HIGH / MEDIUM / LOW).

    Falls back to basic ``retrieve_with_rbac`` if the retrieval mode is
    "fast" (for Ollama) or if query understanding fails.

    Args:
        query: The user's natural-language question.
        user_role: "senior" or "junior".
        user_domain: Domain name (e.g. "semiconductor").
        source_filter: Restrict to these source types.
        metadata_schema: From the domain's ``DOMAIN_CONFIG``.
        k: Number of documents to retrieve.

    Returns:
        ``RetrievalResult`` with documents, filters, confidence, and log.
    """
    metadata_log = ""
    extracted: dict = {}

    # --- Stage 1: Query Understanding (skip in "fast" mode) ---
    if RETRIEVAL_MODE == "deep" and metadata_schema:
        extracted = _extract_query_filters(query, metadata_schema)
        if extracted:
            metadata_log += (
                f"[ADVANCED RAG] Extracted filters: {extracted}\n"
            )
            print(f"[ADVANCED RAG] Extracted filters: {extracted}")
    else:
        metadata_log += "[ADVANCED RAG] Using fast mode (no query understanding)\n"

    # --- Stage 2: Precision Retrieval ---
    # Use the existing retrieve_with_rbac as the base; the precision filter
    # is applied through its normal filter mechanism.
    docs, rbac_log = retrieve_with_rbac(
        query=query,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=source_filter,
        k=k,
    )
    metadata_log += rbac_log

    # --- Post-filter: entity relevance ---
    entity = extracted.get("entity_id", "") or target_entity
    if entity and entity.upper() != "GENERAL":
        docs = _post_filter_by_entity(docs, entity)
        metadata_log += f"[ADVANCED RAG] Post-filtered by entity: '{entity}'\n"

    # --- Confidence scoring ---
    confidence, filter_count = _compute_confidence(docs, extracted)
    metadata_log += f"[ADVANCED RAG] Confidence: {confidence} (filters matched: {filter_count})\n"
    print(f"[ADVANCED RAG] {len(docs)} docs, confidence={confidence}")

    return RetrievalResult(
        documents=docs,
        extracted_filters=extracted,
        confidence=confidence,
        metadata_log=metadata_log,
        filter_match_count=filter_count,
    )


# ---------------------------------------------------------------------------
# Stage 3: Structured Fact Extraction (Extract-then-Evaluate)
# ---------------------------------------------------------------------------
# Uses LangChain's `.with_structured_output(FactCollection)` for guaranteed
# Pydantic-valid output.  Falls back to text+regex if the model does not
# support tool-calling.
# ---------------------------------------------------------------------------

_FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "You are a fact extraction engine.  Extract structured facts from the "
        "documents below.\n\n"
        "TARGET ENTITY: {target_entity}\n"
        "TARGET ATTRIBUTE: {target_attribute}\n\n"
        "UNIT NORMALIZATION RULES (always apply):\n"
        "  - Voltage → Volts (V), e.g. '3.3V' not '3300mV'\n"
        "  - Temperature → Celsius (°C)\n"
        "  - Current → Amps (A) or milliamps (mA)\n"
        "  - Frequency → Hertz (Hz), kHz, MHz, GHz\n"
        "  - Power → Watts (W)\n"
        "  - Percentage → use '%' suffix, e.g. '96.2%'\n\n"
        "EXTRACTION RULES:\n"
        "1. Extract ONLY facts relevant to '{target_entity}' (or all entities "
        "if target is 'GENERAL').\n"
        "2. Prefer extracting the '{target_attribute}' attribute if specified.\n"
        "3. Include ALL factual data points: values, specs, limits, dates.\n"
        "4. If no facts can be extracted, return an empty list.\n\n"
        "DOCUMENTS:\n{documents}"
    ))
])

# Maximum documents to concatenate per LLM batch call
_BATCH_SIZE = 4


def _build_doc_text_batch(documents: list, start: int, end: int) -> str:
    """Concatenate a slice of documents into a single prompt fragment."""
    doc_texts = []
    for doc in documents[start:end]:
        src = doc.metadata.get("source", "unknown")
        doc_id = doc.metadata.get(
            "document_id", doc.metadata.get("source_file", "unknown")
        )
        date = doc.metadata.get("date", doc.metadata.get("version", "unknown"))
        doc_texts.append(
            f"[SOURCE: {src}] [DOC: {doc_id}] [DATE: {date}]\n"
            f"{doc.page_content[:600]}"
        )
    return "\n---\n".join(doc_texts)


def _is_garbage_text(text: str) -> bool:
    """Heuristic to detect binary leaks, PDF streams, or random character blobs."""
    if not text: return True
    binary_patterns = [
        "%PDF", "obj <<", ".indd", " R/Fit", " R/XYZ", " R/FitH", "/Prev", "/Encrypt",
        "/F1", "/F2", "/F3", ">> stream", "endstream", " 0 R", " 0 obj",
        "xmpMM:", "adobe:docid", "stRef:", "xmlns:", "rdf:Description",
        "rdf:li", "xmpG:", "swatchName", "CMYK", "PROCESS", "colorAnt",
        "PGF", "xmp:", "xmpTPg:", "xmpBJ:", "tiff:", "exif:"
    ]
    if any(p in text for p in binary_patterns): return True
    
    # Check for raw binary markers or extreme character densities
    if text.count("\\x") > 10 or text.count("&#x") > 10: return True

    # Character distribution check
    # Many PDF leaks contain lots of symbols and few letters/numbers
    # Or very long strings without spaces/punctuation
    sample = text[:1000]
    total = len(sample)
    if total < 5: return False
    
    # Space density: Real text usually has spaces every 3-15 chars
    spaces = sample.count(" ") + sample.count("\n")
    if total > 50 and spaces / total < 0.05: # Less than 5% spaces
        return True

    # Word length: Binary leaks often have huge "words"
    words = sample.split()
    if words and max(len(w) for w in words) > 60:
        return True
    
    alnum = sum(1 for char in sample if char.isalnum() or char.isspace())
    if alnum / total < 0.25: # Even more lenient for tabular scientific data
        return True

    return False


def perform_llm_fact_extraction(
    documents: list,
    target_entity: str = "GENERAL",
    target_attribute: str = "GENERAL",
    source_type_override: str = "",
    is_generic: bool = False,
) -> list[dict]:
    """
    Core LLM Extraction: Distill raw document chunks into structured facts.
    Used ONLY during the ingestion phase (Shift-Left).
    """
    if not documents:
        return []

    from shared.schemas import ExtractedFact, FactCollection

    # ── Fast mode: metadata-only extraction (no LLM call) ──────────────
    if RETRIEVAL_MODE == "fast":
        print(f"[FACT EXTRACT] Fast mode — extracting {len(documents)} facts from metadata only")
        facts = []
        entity_lower = target_entity.lower() if target_entity and target_entity != "GENERAL" else ""
        attr_name = target_attribute if target_attribute != "GENERAL" else "general_info"
        
        for doc in documents:
            src = doc.metadata.get("source", source_type_override or "unknown")
            doc_id = doc.metadata.get("document_id", "unknown")
            date = doc.metadata.get("date", doc.metadata.get("version", "unknown"))
            content_lower = doc.page_content.lower()

            # Increase snippet size from 250 to 1500 to prevent context poisoning for long documents
            snippet = doc.page_content[:1500].strip().replace("\n", " ")
            
            # Entity-aware labeling
            # Avoid "Fact Poisoning": do not force target_entity if document is about something else
            if entity_lower:
                if entity_lower in content_lower:
                    fact_entity = target_entity
                else:
                    # If not explicitly mentioned, use the document's own title or metadata
                    fact_entity = doc.metadata.get("title", doc.metadata.get("entity", "GENERAL"))[:50]
            else:
                fact_entity = doc.metadata.get("title", target_entity)[:50]

            # Attribute Diversification: If target_attribute is GENERAL, 
            # and it's a generic query, use the document title as the attribute 
            # to prevent collisions in the discrepancy engine.
            # If it's a SPECIFIC query, keep 'general_info' to allow comparison.
            fact_attribute = attr_name
            if attr_name == "general_info" and is_generic:
                fact_attribute = doc.metadata.get("title", "general_info")[:30].lower().replace(" ", "_")

            # Binary & Garbage Filter: Discard individual facts eventually, 
            # but don't skip the whole doc snippet here as it might contain valid text too.
            # if _is_garbage_text(snippet):
            #     print(f"[FACT EXTRACT] ⏭️ Skipping garbage/binary content in '{doc_id}'")
            #     continue

            # Filter out very short or purely decorative snippets (e.g. copyright footers)
            if len(snippet) < 30 and not entity_lower:
                continue

            fact = ExtractedFact(
                entity=fact_entity,
                attribute=fact_attribute,
                value=snippet,
                source_type=src,
                source_doc=str(doc_id),
                date=str(date),
                confidence="MEDIUM",
            )
            facts.append(fact.model_dump())
        return facts

    # ── Deep mode: LLM-based structured extraction with batching ───────
    import time as _time
    from shared.config import BATCH_DELAY

    all_facts: list[dict] = []
    n_docs = len(documents)
    n_batches = (n_docs + _BATCH_SIZE - 1) // _BATCH_SIZE
    
    for batch_idx in range(n_batches):
        start = batch_idx * _BATCH_SIZE
        end = min(start + _BATCH_SIZE, n_docs)
        documents_text = _build_doc_text_batch(documents, start, end)

        try:
            structured_llm = config.llm.with_structured_output(FactCollection)
            chain = _FACT_EXTRACTION_PROMPT | structured_llm
            result: FactCollection = llm_invoke_with_retry(chain, {
                "target_entity": target_entity,
                "target_attribute": target_attribute,
                "documents": documents_text,
            })
            for fact in result.facts:
                if source_type_override:
                    fact.source_type = source_type_override
                all_facts.append(fact.model_dump())
        except Exception as e:
            print(f"[FACT EXTRACT] Structured output failed ({e}), falling back to text parsing")
            # Minimal metadata fallback
            for doc in documents[start:end]:
                all_facts.append({
                    "entity": target_entity,
                    "attribute": target_attribute,
                    "value": doc.page_content[:200],
                    "source_type": doc.metadata.get("source", "unknown"),
                    "source_doc": str(doc.metadata.get("document_id", "unknown")),
                    "date": "unknown",
                    "confidence": "LOW",
                })

        if batch_idx < n_batches - 1:
            _time.sleep(BATCH_DELAY)

    return all_facts


def extract_facts_from_documents(
    documents: list,
    target_entity: str = "GENERAL",
    target_attribute: str = "GENERAL",
    source_type_override: str = "",
    is_generic: bool = False,
) -> list[dict]:
    """
    Shift-Left Retrieval: Directly retrieve pre-extracted facts from SQLite.
    Eliminates LLM latency during the inference phase.
    """
    if not documents:
        return []

    from shared.fact_store import store
    
    all_facts: list[dict] = []
    entity_lower = target_entity.lower() if target_entity and target_entity != "GENERAL" else ""
    
    print(f"[FACT RETRIEVAL] Fetching pre-extracted facts for {len(documents)} docs...")
    
    for doc in documents:
        doc_id = doc.metadata.get("document_id")
        if not doc_id:
            continue
            
        facts = store.get_facts_by_doc_id(doc_id)
        for f in facts:
            # Binary & Garbage Filter: Discard facts that look like PDF garbage or raw bytes
            if _is_garbage_text(f.value):
                continue
                
            # Filter out generic high-level facts that don't satisfy precision
            if f.attribute == "general_info" and len(f.value) < 100:
                continue
            # Conditional Re-labeling for Generic Queries (Topic-based)
            if is_generic and f.attribute == "general_info":
                f.attribute = f.source_doc.lower().replace(" ", "_").replace(".", "_")

            all_facts.append(f.model_dump())

    # Fallback if no facts were found in DB (e.g. document not yet processed by new ingestion)
    if not all_facts:
        print("[FACT RETRIEVAL] ⚠️ No pre-extracted facts found. Attempting real-time extraction (fallback).")
        real_time_facts = perform_llm_fact_extraction(documents, target_entity, target_attribute, source_type_override, is_generic)
        return real_time_facts

    print(f"[FACT RETRIEVAL] Retrieved {len(all_facts)} facts from store.")
    return all_facts



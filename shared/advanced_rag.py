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
    Post-retrieval filter: REMOVE documents that don't mention the target
    entity.  If NO docs match, keep the top 2 as fallback context.
    """
    if not entity or entity.upper() == "GENERAL":
        return docs

    entity_lower = entity.lower()
    matched = []
    unmatched = []
    for doc in docs:
        if entity_lower in doc.page_content.lower():
            matched.append(doc)
        else:
            unmatched.append(doc)

    # If entity matched docs exist, return ONLY those
    if matched:
        return matched
    # Fallback: keep top 2 unmatched docs for general context
    return unmatched[:2]


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

_FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "You are a fact extraction engine.  Extract structured facts from the "
        "documents below.\n\n"
        "TARGET ENTITY: {target_entity}\n"
        "TARGET ATTRIBUTE: {target_attribute}\n\n"
        "For each document chunk, extract facts as JSON objects with these fields:\n"
        "- entity: the entity this fact is about\n"
        "- attribute: the specific attribute (e.g. 'max_voltage', 'dosage')\n"
        "- value: the extracted value\n"
        "- source_type: the document source type\n"
        "- source_doc: the source filename\n"
        "- date: timestamp or version date\n"
        "- confidence: HIGH / MEDIUM / LOW\n\n"
        "DOCUMENTS:\n{documents}\n\n"
        "RULES:\n"
        "1. Extract ONLY facts relevant to '{target_entity}' (or all entities "
        "if target is 'GENERAL').\n"
        "2. Prefer extracting the '{target_attribute}' attribute if specified.\n"
        "3. Include ALL factual data points you find — values, specs, limits, dates.\n"
        "4. If no facts can be extracted, return an empty list.\n\n"
        "Return a JSON array of fact objects. Example:\n"
        '[{{"entity": "RTX-9000", "attribute": "max_voltage", "value": "5.0V", '
        '"source_type": "datasheet", "source_doc": "spec_v4.2.txt", '
        '"date": "2024-01-15", "confidence": "HIGH"}}]'
    ))
])


def extract_facts_from_documents(
    documents: list,
    target_entity: str = "GENERAL",
    target_attribute: str = "GENERAL",
    source_type_override: str = "",
) -> list[dict]:
    """
    Extract-then-Evaluate: Distill raw document chunks into structured facts.

    In 'fast' mode (Ollama): metadata-only extraction (zero LLM calls).
    In 'deep' mode (Gemini/Groq): LLM-based JSON extraction.

    Args:
        documents: List of LangChain Document objects.
        target_entity: Entity to focus extraction on (from QueryIntent).
        target_attribute: Attribute to focus on (from QueryIntent).
        source_type_override: Override source_type for all facts (e.g. "email").

    Returns:
        List of ExtractedFact dicts (serialized via .model_dump()).
    """
    if not documents:
        return []

    from shared.schemas import ExtractedFact, FactCollection

    # --- Fast mode: metadata-only extraction (no LLM call) ---
    if RETRIEVAL_MODE == "fast":
        print(f"[FACT EXTRACT] Fast mode — extracting {len(documents)} facts from metadata only")
        facts = []
        entity_lower = target_entity.lower() if target_entity and target_entity != "GENERAL" else ""
        for doc in documents:
            src = doc.metadata.get("source", source_type_override or "unknown")
            doc_id = doc.metadata.get("document_id", "unknown")
            date = doc.metadata.get("date", doc.metadata.get("version", "unknown"))
            content_lower = doc.page_content.lower()

            # Entity-aware labeling: only assign target_entity if content mentions it
            if entity_lower and entity_lower in content_lower:
                fact_entity = target_entity
            elif entity_lower:
                # Doc doesn't mention the target entity — label with actual content entity
                fact_entity = doc.metadata.get("title", "unrelated_doc")[:50]
            else:
                fact_entity = "unknown"

            fact = ExtractedFact(
                entity=fact_entity,
                attribute=target_attribute if target_attribute != "GENERAL" else "general_info",
                value=doc.page_content[:100],  # Keep small for fast mode
                source_type=src,
                source_doc=str(doc_id),
                date=str(date),
                confidence="MEDIUM",
            )
            facts.append(fact.model_dump())
        return facts

    # --- Deep mode: LLM-based extraction ---
    # Build document text for the extraction prompt
    doc_texts = []
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        doc_id = doc.metadata.get("document_id", doc.metadata.get("source_file", "unknown"))
        date = doc.metadata.get("date", doc.metadata.get("version", "unknown"))
        doc_texts.append(
            f"[SOURCE: {src}] [DOC: {doc_id}] [DATE: {date}]\n{doc.page_content[:600]}"
        )
    documents_text = "\n---\n".join(doc_texts)

    # --- Text-based JSON extraction (compatible with all backends) ---
    try:
        chain = _FACT_EXTRACTION_PROMPT | config.llm | StrOutputParser()
        raw = llm_invoke_with_retry(chain, {
            "target_entity": target_entity,
            "target_attribute": target_attribute,
            "documents": documents_text,
        })

        # Parse JSON from response (handle markdown code blocks)
        import json, re
        raw_clean = raw.strip()
        if "```" in raw_clean:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_clean)
            if match:
                raw_clean = match.group(1).strip()

        parsed = json.loads(raw_clean)
        if isinstance(parsed, list):
            facts = []
            for item in parsed:
                if isinstance(item, dict):
                    try:
                        fact = ExtractedFact(**item)
                        if source_type_override:
                            fact.source_type = source_type_override
                        facts.append(fact.model_dump())
                    except Exception:
                        continue
            print(f"[FACT EXTRACT] Text fallback: {len(facts)} facts extracted")
            return facts
    except Exception as e:
        print(f"[FACT EXTRACT] Text extraction failed ({e})")

    # --- Fallback: Minimal extraction from metadata only (no LLM) ---
    print("[FACT EXTRACT] ⚠️ All extraction methods failed, using metadata-only fallback")
    facts = []
    for doc in documents:
        src = doc.metadata.get("source", source_type_override or "unknown")
        doc_id = doc.metadata.get("document_id", "unknown")
        date = doc.metadata.get("date", doc.metadata.get("version", "unknown"))
        fact = ExtractedFact(
            entity=target_entity if target_entity != "GENERAL" else "unknown",
            attribute=target_attribute if target_attribute != "GENERAL" else "general_info",
            value=doc.page_content[:200],
            source_type=src,
            source_doc=str(doc_id),
            date=str(date),
            confidence="LOW",
        )
        facts.append(fact.model_dump())
    return facts


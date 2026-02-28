"""
================================================================================
Official Documents Agent — Extract-then-Evaluate (Structured Fact Passing)
================================================================================
Retrieves OFFICIAL, HIGH-AUTHORITY documents (Datasheets, SOPs, Specs),
then EXTRACTS structured facts using the Advanced RAG pipeline.

Returns List[ExtractedFact] as serialized dicts — raw document text does
NOT flow downstream to the discrepancy agent.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.advanced_rag import query_understand_and_retrieve, extract_facts_from_documents

# Import the domain's metadata schema for query understanding
from agents_logic.semiconductor_agents.domain_config import DOMAIN_CONFIG

_METADATA_SCHEMA = DOMAIN_CONFIG.get("metadata_schema", {})


@vera_agent("Official Docs Agent")
def run(state: GraphState) -> dict:
    """
    OFFICIAL DOCS AGENT: Retrieve → Extract → Return Structured Facts.

    1. Advanced RAG retrieval with RBAC + query understanding filters.
    2. LLM-based fact extraction focused on target_entity/target_attribute.
    3. Returns serialized ExtractedFact dicts to GraphState (official_facts).
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "semiconductor")
    target_entity = state.get("target_entity", "GENERAL")
    target_attribute = state.get("target_attribute", "GENERAL")

    # --- Stage 1: Precision Retrieval ---
    result = query_understand_and_retrieve(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["datasheet", "sop", "spec", "document"],
        metadata_schema=_METADATA_SCHEMA,
        k=10,
        target_entity=target_entity,
    )

    # --- Stage 2: Structured Fact Extraction ---
    facts = extract_facts_from_documents(
        documents=result.documents,
        target_entity=target_entity,
        target_attribute=target_attribute,
        source_type_override="",  # preserve original source types
    )

    print(f"[Official Docs Agent] {len(result.documents)} docs → {len(facts)} structured facts")

    # Accumulate into existing facts
    existing_facts = state.get("official_facts") or []
    all_facts = existing_facts + facts

    # Keep raw docs for backward compatibility (response agent may still use them)
    existing_docs = state.get("documents", [])
    retrieved_docs = state.get("retrieved_docs") or {}
    retrieved_docs["official"] = result.documents

    return {
        "official_facts": all_facts,
        "official_data": result.documents,
        "documents": existing_docs + result.documents,
        "metadata_log": state.get("metadata_log", "") + result.metadata_log,
        "retrieved_docs": retrieved_docs,
        "retrieval_confidence": result.confidence,
        "_thinking": (
            f"Retrieve→Extract: {len(result.documents)} docs → {len(facts)} facts "
            f"(entity='{target_entity}', attr='{target_attribute}', "
            f"confidence={result.confidence})."
        ),
    }

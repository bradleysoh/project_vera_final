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
from agents_logic.medical_agents.domain_config import DOMAIN_CONFIG

_METADATA_SCHEMA = DOMAIN_CONFIG.get("metadata_schema", {})


@vera_agent("Medical Official Docs Agent")
def run(state: GraphState) -> dict:
    """
    OFFICIAL DOCS AGENT: Retrieve → Extract → Return Structured Facts.

    1. Advanced RAG retrieval with RBAC + query understanding filters.
    2. LLM-based fact extraction focused on target_entity/target_attribute.
    3. Returns serialized ExtractedFact dicts to GraphState (official_facts).
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "medical")
    target_entity = state.get("target_entity", "GENERAL")
    target_attribute = state.get("target_attribute", "GENERAL")

    # --- Guard Clause: Fast-fail if intent doesn't need specs ---
    intent = state.get("intent", "")
    if intent not in ("spec_retrieval", "cross_reference", ""):
        print(f"[Medical Official Docs Agent] ⏭️ Fast-fail: intent='{intent}' is not spec-related")
        return {}

    # --- Stage 1: Precision Retrieval ---
    result = query_understand_and_retrieve(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["datasheet", "sop", "spec", "document", "dataset", "db_info"],
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

    print(f"[Medical Official Docs Agent] {len(result.documents)} docs → {len(facts)} structured facts")

    # Per-step return: ONLY the tokens we added (reducers handle the merge)
    return {
        "official_facts": facts,
        "official_data": result.documents,
        "documents": result.documents,
        "metadata_log": result.metadata_log,
        "retrieval_confidence": result.confidence,
        "thought_process": [
            f"Retrieve→Extract: {len(result.documents)} docs → {len(facts)} facts "
            f"(entity='{target_entity}', attr='{target_attribute}', "
            f"confidence={result.confidence})."
        ],
        "_thinking": f"Refined {len(facts)} facts from specs.",
    }

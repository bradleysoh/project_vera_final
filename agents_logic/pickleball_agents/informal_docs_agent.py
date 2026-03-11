"""
================================================================================
Informal Documents Agent — Extract-then-Evaluate (Structured Fact Passing)
================================================================================
Retrieves INFORMAL, TIME-SENSITIVE documents (Emails, Memos, DMs),
then EXTRACTS structured facts with timeline metadata.

Informal facts can OVERRIDE Official Docs ONLY IF they are NEWER.
The discrepancy agent applies this rule deterministically on the dates.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.advanced_rag import query_understand_and_retrieve, extract_facts_from_documents

# Import the domain's metadata schema for query understanding
from agents_logic.pickleball_agents.domain_config import DOMAIN_CONFIG

_METADATA_SCHEMA = DOMAIN_CONFIG.get("metadata_schema", {})


@vera_agent("Informal Docs Agent")
def run(state: GraphState) -> dict:
    """
    INFORMAL DOCS AGENT: Retrieve → Extract → Return Structured Facts.

    1. Advanced RAG retrieval with RBAC for informal sources.
    2. LLM-based fact extraction focused on target_entity/target_attribute.
    3. Returns serialized ExtractedFact dicts to GraphState (informal_facts).
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "pickleball")
    target_entity = state.get("target_entity", "GENERAL")
    target_attribute = state.get("target_attribute", "GENERAL")

    # --- Guard Clause: Fast-fail if intent doesn't need informal docs ---
    intent = state.get("intent", "")
    if intent not in ("cross_reference", ""):
        print(f"[Informal Docs Agent] ⏭️ Fast-fail: intent='{intent}' is not cross-reference")
        return {}

    # --- Stage 1: Precision Retrieval ---
    result = query_understand_and_retrieve(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["email", "memo", "dm"],
        metadata_schema=_METADATA_SCHEMA,
        k=10,
        target_entity=target_entity,
    )

    # --- Stage 2: Structured Fact Extraction ---
    facts = extract_facts_from_documents(
        documents=result.documents,
        target_entity=target_entity,
        target_attribute=target_attribute,
        source_type_override="",  # preserve original source types (email/memo/dm)
    )

    print(f"[Informal Docs Agent] {len(result.documents)} docs → {len(facts)} structured facts")

    print(f"[Informal Docs Agent] {len(result.documents)} docs → {len(facts)} structured facts")

    # Per-step return: ONLY the tokens we added (reducers handle the merge)
    return {
        "informal_facts": facts,
        "informal_data": result.documents,
        "documents": result.documents,
        "metadata_log": result.metadata_log,
        "retrieval_confidence": result.confidence,
        "thought_process": [
            f"Retrieve→Extract: {len(result.documents)} docs → {len(facts)} facts "
            f"(entity='{target_entity}', attr='{target_attribute}', "
            f"confidence={result.confidence})."
        ],
        "_thinking": f"Refined {len(facts)} facts from informal sources.",
    }

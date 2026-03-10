"""
Legal Official Docs Agent.

Retrieves CUAD-derived legal reference documents and extracts structured facts.
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.advanced_rag import query_understand_and_retrieve, extract_facts_from_documents
from agents_logic.legal_agents.domain_config import DOMAIN_CONFIG

_METADATA_SCHEMA = DOMAIN_CONFIG.get("metadata_schema", {})


@vera_agent("Legal Official Docs Agent")
def run(state: GraphState) -> dict:
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "legal")
    target_entity = state.get("target_entity", "GENERAL")
    target_attribute = state.get("target_attribute", "GENERAL")
    intent = state.get("intent", "")

    if intent not in ("spec_retrieval", "cross_reference", "db_query", ""):
        print(f"[Legal Official Docs Agent] ⏭️ Fast-fail: intent='{intent}' is not legal retrieval related")
        return {}

    result = query_understand_and_retrieve(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["document", "dataset", "spec", "sop"],
        metadata_schema=_METADATA_SCHEMA,
        k=10,
        target_entity=target_entity,
    )

    facts = extract_facts_from_documents(
        documents=result.documents,
        target_entity=target_entity,
        target_attribute=target_attribute,
        source_type_override="document",
    )

    print(f"[Legal Official Docs Agent] {len(result.documents)} docs → {len(facts)} structured facts")

    return {
        "official_facts": facts,
        "official_data": result.documents,
        "documents": result.documents,
        "metadata_log": result.metadata_log,
        "retrieval_confidence": result.confidence,
        "thought_process": [
            f"Legal retrieval from CUAD references: {len(result.documents)} docs → {len(facts)} facts."
        ],
        "_thinking": f"Retrieved {len(result.documents)} legal reference docs and extracted {len(facts)} facts.",
    }


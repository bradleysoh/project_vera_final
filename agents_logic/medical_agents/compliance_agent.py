"""
================================================================================
Medical Compliance Agent — Placeholder Template (Medical Domain)
================================================================================
OWNER: (Assign team member)
DOMAIN: medical
RESPONSIBILITY: Retrieve medical compliance docs (FDA, clinical trials, etc.).
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.config import retrieve_with_rbac


@vera_agent("Medical Compliance Agent")
def run(state: GraphState) -> dict:
    """
    MEDICAL COMPLIANCE AGENT: Placeholder — retrieves medical compliance docs.

    Replace with your domain-specific logic for FDA docs, clinical SOPs, etc.

    Returns:
        dict: {"documents": [...], "metadata_log": str, "retrieved_docs": {...}}
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "medical")

    documents, metadata_log = retrieve_with_rbac(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["sop", "email", "db_info", "document"],
        k=3,
    )

    print(f"[Medical Compliance Agent] Retrieved {len(documents)} compliance documents")

    sources_found = set(doc.metadata.get("source") for doc in documents)
    retrieved_docs = state.get("retrieved_docs") or {}
    retrieved_docs["compliance"] = documents

    return {
        "documents": documents,
        "metadata_log": metadata_log,
        "retrieved_docs": retrieved_docs,
        "_thinking": (
            f"Retrieved {len(documents)} compliance docs ({', '.join(sources_found) if sources_found else 'none'}) "
            f"for '{user_domain}' domain with '{user_role}' access."
        ),
    }

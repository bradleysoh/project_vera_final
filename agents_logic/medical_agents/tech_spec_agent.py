"""
================================================================================
Medical Spec Agent — Placeholder Template (Medical Domain)
================================================================================
OWNER: (Assign team member)
DOMAIN: medical
RESPONSIBILITY: Retrieve medical device specifications with RBAC filtering.

Copy this file and customize for your medical domain agent.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.config import retrieve_with_rbac


@vera_agent("Medical Spec Agent")
def run(state: GraphState) -> dict:
    """
    MEDICAL SPEC AGENT: Placeholder — retrieves medical device specifications.

    Replace this with your domain-specific retrieval logic.
    Use source_filter values matching your data (e.g., ["datasheet", "sop"]).

    Returns:
        dict: {"documents": [...], "metadata_log": str, "retrieved_docs": {...}}
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "medical")

    # TODO: Customize source_filter for medical domain data
    documents, metadata_log = retrieve_with_rbac(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["datasheet", "db_info"],
        k=4,
    )

    print(f"[Medical Spec Agent] Retrieved {len(documents)} documents")

    retrieved_docs = state.get("retrieved_docs") or {}
    retrieved_docs["tech"] = documents

    return {
        "documents": documents,
        "metadata_log": metadata_log,
        "retrieved_docs": retrieved_docs,
        "_thinking": (
            f"Retrieved {len(documents)} tech docs for '{user_domain}' domain "
            f"with '{user_role}' access. Query: '{question[:60]}'."
        ),
    }

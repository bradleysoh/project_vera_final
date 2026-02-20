"""
================================================================================
Official Documents Agent (Baseline Authority)
================================================================================
Retrieves OFFICIAL, HIGH-AUTHORITY documents:
  - Datasheets
  - SOPs (Standard Operating Procedures)
  - Specifications

These documents form the "Baseline" for discrepancy checking.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.config import retrieve_with_rbac

@vera_agent("Official Docs Agent")
def run(state: GraphState) -> dict:
    """
    Retrieves official documentation (Specs, SOPs).
    Populates 'official_data' in the graph state.
    Accumulates 'documents' for the Response Agent.
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "semiconductor")
    existing_docs = state.get("documents", [])
    
    # Retrieve authoritative sources
    documents, metadata_log = retrieve_with_rbac(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["datasheet", "sop", "spec", "document"], 
        k=10, 
    )

    # Store in state
    retrieved_docs = state.get("retrieved_docs") or {}
    retrieved_docs["official"] = documents

    return {
        "official_data": documents,
        "documents": existing_docs + documents, # ACCUMULATE
        "metadata_log": state.get("metadata_log", "") + metadata_log,
        "retrieved_docs": retrieved_docs,
        "_thinking": f"Retrieved {len(documents)} OFFICIAL documents."
    }

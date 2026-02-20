"""
================================================================================
Informal Documents Agent (Timeline Exception)
================================================================================
Retrieves INFORMAL, TIME-SENSITIVE documents:
  - Emails
  - Memos
  - Direct Messages

These documents can OVERRIDE Official Docs ONLY IF they are NEWER.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.config import retrieve_with_rbac

@vera_agent("Informal Docs Agent")
def run(state: GraphState) -> dict:
    """
    Retrieves informal documentation (Emails, Memos).
    Populates 'informal_data' in the graph state.
    Accumulates 'documents' for the Response Agent.
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "semiconductor")
    existing_docs = state.get("documents", [])
    
    # Retrieve informal sources
    documents, metadata_log = retrieve_with_rbac(
        query=question,
        user_role=user_role,
        user_domain=user_domain,
        source_filter=["email", "memo", "dm"], 
        k=10, 
    )

    # Store in state
    retrieved_docs = state.get("retrieved_docs") or {}
    retrieved_docs["informal"] = documents

    return {
        "informal_data": documents,
        "documents": existing_docs + documents, # ACCUMULATE
        "metadata_log": state.get("metadata_log", "") + metadata_log,
        "retrieved_docs": retrieved_docs,
        "_thinking": f"Retrieved {len(documents)} INFORMAL documents."
    }

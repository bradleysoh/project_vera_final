"""
================================================================================
AGENT TEMPLATE — Copy this file to create your own agent
================================================================================

Instructions:
  1. Copy this file:  cp _template_agent.py my_agent.py
  2. Rename the @vera_agent("Your Agent Name") decorator
  3. Import shared resources you need from shared.config
  4. Return a dict containing ONLY the state fields you update

Rules:
  ✅ Your agent MUST have a function called `run(state: GraphState) -> dict`
  ✅ Use the `@vera_agent("Name")` decorator for automatic logging
  ✅ Import GraphState from `shared.graph_state`
  ✅ Import llm, vector_store, etc. from `shared.config`
  ❌ Do NOT create your own LLM or vector store instances
  ❌ Do NOT modify GraphState fields that aren't your responsibility

Example: This template retrieves documents for a specific source type.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.config import (
    llm,
    vector_store,
    retrieve_with_rbac,
    llm_invoke_with_retry,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# AGENT NAME: (your agent name here)
# OWNER: (your name)
# RESPONSIBILITY: (what this agent does)
# ==============================================================================

@vera_agent("My Agent Name")  # <- Change this to your agent's name
def run(state: GraphState) -> dict:
    """
    Main agent function — this is called by the LangGraph workflow.

    MUST accept `state: GraphState` and return a `dict` of updated fields.
    The @vera_agent decorator handles logging automatically.

    Args:
        state: The current graph state (read any field you need).

    Returns:
        dict: A dictionary of state fields to update.
              Only include the fields YOUR agent is responsible for.

    Example returns:
        return {"documents": [...], "metadata_log": "...", "retrieved_docs": {...}}
        return {"generation": "..."}
        return {"route": "...", "flagged": True, "next_agent": "engineering"}
        return {"generation": "...", "discrepancy_report": "..."}
    """
    question = state["question"]
    user_role = state["user_role"]

    # -----------------------------------------------------------------------
    # OPTION A: Retrieve documents from ChromaDB
    # -----------------------------------------------------------------------
    # documents, metadata_log = retrieve_with_rbac(
    #     query=question,
    #     user_role=user_role,
    #     source_filter=["datasheet"],  # Change to your source type
    #     k=4,
    # )
    # # Store in retrieved_docs for cross-agent comparison
    # retrieved_docs = state.get("retrieved_docs") or {}
    # retrieved_docs["my_agent"] = documents
    # return {
    #     "documents": documents,
    #     "metadata_log": metadata_log,
    #     "retrieved_docs": retrieved_docs,
    # }

    # -----------------------------------------------------------------------
    # OPTION B: Use LLM to analyze/generate text
    # -----------------------------------------------------------------------
    # prompt = ChatPromptTemplate.from_messages([
    #     ("human", "Analyze: {question}\\n\\nRESPONSE:")
    # ])
    # chain = prompt | llm | StrOutputParser()
    # result = llm_invoke_with_retry(chain, {"question": question})
    # return {"generation": result}

    # -----------------------------------------------------------------------
    # Placeholder — replace with your actual logic
    # -----------------------------------------------------------------------
    return {
        "generation": f"[MY AGENT] Placeholder response for: {question}",
        # DeepSeek-style thinking trace (auto-appended by @vera_agent decorator)
        "_thinking": f"Processed query '{question[:50]}' with role '{user_role}'.",
    }

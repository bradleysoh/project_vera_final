"""
================================================================================
Project VERA — Shared GraphState Definition
================================================================================

This is the SINGLE SOURCE OF TRUTH for the state schema used by all agents
in the VERA multi-agent system.

ALL agents MUST import GraphState from this file:
    from shared.graph_state import GraphState

DO NOT define your own state — use this one.
================================================================================
"""

from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Defines the state schema for the VERA LangGraph workflow.

    Every agent receives the full state and returns a dict containing
    ONLY the fields it wants to update.

    Attributes:
        question (str): The user's input query.
        generation (str): The LLM-generated response.
        user_role (str): The user's role — "senior" or "junior".
            Determines which documents the user can access (RBAC).
        user_domain (str): The user's assigned domain — "semiconductor"
            or "medical". Used for:
              1. Double-filter retrieval (domain + access_level)
              2. Out-of-domain detection by the router
        documents (List[Document]): Retrieved documents from ChromaDB.
        route (str): The routing decision — "technical", "compliance",
            or "escalate".
        flagged (bool): Security flag — True if the query is flagged for
            unauthorized access attempt OR out-of-domain query.
        metadata_log (str): Logging information about the retrieval process
            for transparency and auditability.
        retrieved_docs (dict): Per-agent retrieved document storage.
            Populated by retrieval agents:
              {"tech": [Document, ...], "compliance": [Document, ...]}
            Used by the discrepancy agent for cross-agent comparison.
        db_result (str): Structured result from NL-to-SQL database query.
            Populated by the DB Agent for cross-referencing with documents.
        discrepancy_report (str): Structured discrepancy report from the
            Case Agent (discrepancy_agent). Empty if no conflicts found.
        next_agent (str): Domain-based routing target set by the router.
            Examples: "semiconductor", "medical". Used for multi-domain routing.
        thought_process (List[str]): DeepSeek-style sequential reasoning trace.
            Each agent appends its internal monologue before returning.
        refinement_count: int     # Tracks discussion loop iterations (max 3)
        max_refinements: int      # Configurable limit for discussion loop (default: 3)
        critique: str             # Feedback from Discrepancy Agent to Response Agent.
    """
    question: str
    generation: str
    user_role: str
    user_domain: str
    documents: List[Document]
    route: str
    flagged: bool
    metadata_log: str
    retrieved_docs: dict
    db_data: str              # Step 1: Structured DB results (Operational Reality)
    official_data: list[Document]  # Step 2: Official Docs (SOPs, Specs) (Baseline)
    informal_data: list[Document]  # Step 3: Informal Emails (Timeline Exception)
    latest_timestamp: str     # Track effective timestamp of overriding data
    discrepancy_report: str
    next_agent: str
    thought_process: list[str]
    refinement_count: int
    max_refinements: int
    critique: str

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

    == CORE FIELDS ==
    question            — The user's input query.
    generation          — The LLM-generated response.
    user_role           — "senior" or "junior" (determines RBAC access).
    user_domain         — "semiconductor" or "medical" (domain isolation).

    == QUERY UNDERSTANDING (set by Router Agent) ==
    target_entity       — Primary entity extracted from the query (e.g. "RTX-9000").
    target_attribute    — Specific attribute being asked about (e.g. "max_voltage").
    time_context        — Temporal qualifier (e.g. "latest", "2024-Q3", or "").

    == ROUTING & SECURITY ==
    route               — "technical", "compliance", or "escalate".
    flagged             — True if query is flagged for security/out-of-domain.
    next_agent          — Domain routing target (e.g. "semiconductor").

    == RAW RETRIEVAL (kept for backward compat, but NOT passed to discrepancy) ==
    documents           — Raw retrieved documents (List[Document]).
    retrieved_docs      — Per-agent raw doc storage (dict).

    == STRUCTURED FACTS (Pydantic-serialized ExtractedFact dicts) ==
    official_facts      — Facts from official/baseline docs (List[dict]).
    informal_facts      — Facts from informal/timeline docs (List[dict]).
    db_facts            — Facts from database agent (List[dict]).

    == DISCREPANCY & RESPONSE ==
    discrepancy_verdict — Serialized DiscrepancyVerdict (dict).
    discrepancy_report  — Legacy text report (str).
    retrieval_confidence — HIGH / MEDIUM / LOW from Advanced RAG.

    == AGENT INFRASTRUCTURE ==
    metadata_log        — Retrieval process logging.
    thought_process     — Per-agent reasoning trace.
    refinement_count    — Tracks discussion loop iterations.
    max_refinements     — Configurable limit (default: 3).
    critique            — Feedback from Discrepancy Agent to Response Agent.
    """
    # --- Core ---
    question: str
    generation: str
    user_role: str
    user_domain: str

    # --- Query Understanding ---
    target_entity: str
    target_attribute: str
    time_context: str

    # --- Routing & Security ---
    route: str
    flagged: bool
    next_agent: str

    # --- Raw Retrieval (backward compat) ---
    documents: List[Document]
    retrieved_docs: dict
    db_data: str              # Raw DB results string (Operational Reality)
    official_data: list       # Raw official docs (legacy, being replaced by facts)
    informal_data: list       # Raw informal docs (legacy, being replaced by facts)
    latest_timestamp: str

    # --- Structured Facts ---
    official_facts: list      # List[dict] — serialized ExtractedFact objects
    informal_facts: list      # List[dict] — serialized ExtractedFact objects
    db_facts: list            # List[dict] — serialized ExtractedFact objects

    # --- Discrepancy & Response ---
    discrepancy_verdict: dict  # Serialized DiscrepancyVerdict
    discrepancy_report: str    # Legacy text report
    retrieval_confidence: str  # HIGH / MEDIUM / LOW

    # --- Agent Infrastructure ---
    metadata_log: str
    thought_process: list[str]
    refinement_count: int
    max_refinements: int
    critique: str

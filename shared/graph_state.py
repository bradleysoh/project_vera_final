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

from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union
from langchain_core.documents import Document

# --- Reducers for robust state merging ---
def _merge_metadata(old: str, new: str) -> str:
    """Appends new log entries to the existing log string."""
    if not old: return new
    if not new: return old
    # Ensure newline separation
    return old + "\n" + new if not old.endswith("\n") else old + new

def _merge_list(old: list, new: list) -> list:
    """Standard list addition reducer."""
    return (old or []) + (new or [])

class GraphState(TypedDict):
    """
    Defines the state schema for the VERA LangGraph workflow.
    Uses Annotated reducers to allow accumulating data (documents, facts, logs)
    cleanly across the graph flow, preventing 'InvalidUpdateError'.
    """
    # --- Core ---
    question: str
    generation: str
    user_role: str
    user_domain: str

    # --- Query Understanding (set by Router Agent) ---
    target_entity: str
    entity_type: str
    target_attribute: str
    time_context: str

    # --- Routing & Security ---
    route: str
    intent: str   # "db_query", "spec_retrieval", "cross_reference", "general_chat", "out_of_domain"
    flagged: bool
    next_agent: str

    # --- Raw Retrieval (Accumulated via Reducers) ---
    documents: Annotated[List[Document], _merge_list]
    retrieved_docs: dict
    db_data: str              # Raw DB results string
    official_data: Annotated[list, _merge_list]
    informal_data: Annotated[list, _merge_list]
    latest_timestamp: str

    # --- Structured Facts (Accumulated via Reducers) ---
    official_facts: Annotated[list, _merge_list]
    informal_facts: Annotated[list, _merge_list]
    db_facts: Annotated[list, _merge_list]

    # --- Discrepancy & Response ---
    discrepancy_verdict: dict  # Serialized DiscrepancyVerdict
    discrepancy_report: str    # Legacy text report (Discrepancy Agent)
    discrepancy_report_summary: str  # Summarized report (Response Agent)
    retrieval_confidence: str  # HIGH / MEDIUM / LOW

    # --- Agent Infrastructure (Accumulated via Reducers) ---
    metadata_log: Annotated[str, _merge_metadata]
    thought_process: Annotated[List[str], _merge_list]
    refinement_count: int
    max_refinements: int
    critique: str


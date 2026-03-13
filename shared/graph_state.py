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

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union
from langchain_core.documents import Document

# --- Reducers for robust state merging ---
def _merge_metadata(old: str, new: str) -> str:
    """Appends new log entries to the existing log string cleanly."""
    if not old: return new or ""
    if not new: return old or ""
    return old + "\n" + new if not old.endswith("\n") else old + new

def _merge_list(old: list, new: list) -> list:
    """Standard list addition reducer. Prevents NoneType errors."""
    return (old or []) + (new or [])

def _merge_dict(old: dict, new: dict) -> dict:
    """Standard dict merge reducer for plugin architectures."""
    res = (old or {}).copy()
    res.update(new or {})
    return res

class GraphState(TypedDict):
    """
    Defines the rigid state schema for the VERA LangGraph workflow.
    """
    # ==========================================
    # 1. CORE CONTEXT
    # ==========================================
    question: str
    generation: str
    user_role: str
    user_domain: str
    input_contract_text: str
    input_contract_name: str

    # ==========================================
    # 2. QUERY UNDERSTANDING (Router Output)
    # ==========================================
    target_entity: str
    entity_type: str
    target_attribute: str
    time_context: str
    route: str
    intent: str   # "db_query", "spec_retrieval", "cross_reference", "general_chat", "out_of_domain"
    flagged: bool
    is_generic_query: bool
    next_agent: str

    # ==========================================
    # 3. DYNAMIC STATE MACHINE (Early Exit)
    # ==========================================
    # CRITICAL: Using operator.or_ means if ANY agent sets this to True,
    # the kill-switch engages and cannot be overwritten to False by other agents.
    is_resolved: Annotated[bool, operator.or_] 
    
    # Phase 3 Plugin Sandbox: Domain-specific computed logic (e.g., Medical BMI, Legal CUAD)
    domain_computed_facts: Annotated[dict, _merge_dict]

    # ==========================================
    # 4. RAW RETRIEVAL PIPELINE
    # ==========================================
    documents: Annotated[List[Document], _merge_list]
    db_data: str              # Standard overwrite: Latest DB query takes precedence
    official_data: Annotated[list, _merge_list]
    informal_data: Annotated[list, _merge_list]
    latest_timestamp: str

    # ==========================================
    # 5. SHIFT-LEFT: STRUCTURED FACTS
    # ==========================================
    official_facts: Annotated[list, _merge_list]
    informal_facts: Annotated[list, _merge_list]
    db_facts: Annotated[list, _merge_list]

    # ==========================================
    # 6. DISCREPANCY AUDIT
    # ==========================================
    discrepancy_verdict: dict  # Serialized DiscrepancyVerdict
    discrepancy_report: str    # Detailed textual conflict report
    discrepancy_report_summary: str 
    retrieval_confidence: str  # HIGH / MEDIUM / LOW

    # ==========================================
    # 7. AGENT TELEMETRY & TRACEABILITY
    # ==========================================
    metadata_log: Annotated[str, _merge_metadata]
    thought_process: Annotated[List[str], _merge_list]
    refinement_count: int
    max_refinements: int
    critique: str
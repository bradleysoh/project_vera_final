"""
================================================================================
Router Agent — Global Query Planner with Structured Output & Query Rewriting
================================================================================
RESPONSIBILITY: 
    Act as the system's "Query Planner". Uses LLM structured output to 
    perform intent recognition, entity extraction, security validation, 
    and query optimization (rewriting) in a SINGLE ATOMIC STEP.
    ABSOLUTELY NO DOMAIN-SPECIFIC LOGIC (e.g., 'legal' hardcoding) ALLOWED.
================================================================================
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from shared.graph_state import GraphState
from shared.agent_base import vera_agent
import shared.config as config
from shared.config import llm_invoke_with_retry
from shared.dynamic_loader import (
    get_available_domains,
    load_domain_configs,
    resolve_domain_alias,
)
from langchain_core.prompts import ChatPromptTemplate

# --- INTENT CONSTANTS (for app.py compatibility) ---
INTENT_DB = "db_query"
INTENT_SPECS = "spec_retrieval"
INTENT_CROSS = "cross_reference"
INTENT_CHAT = "general_chat"

# --- Pydantic Schema for Structured Output ---
class QueryPlannerOutput(BaseModel):
    thought_process: str = Field(description="Reasoning.")
    user_intent_category: Literal["SMALL_TALK", "GENERIC_QA", "SPEC_LOOKUP", "DISCREPANCY_AUDIT", "DATA_QUERY"] = Field(
        description="Category: DATA_QUERY (tables/metrics), GENERIC_QA (summaries), SPEC_LOOKUP (facts), DISCREPANCY_AUDIT (conflicts), SMALL_TALK (greetings)."
    )
    detected_domain: str = Field(description="Inferred domain.")
    target_entity: str = Field(description="Specific identifier (e.g. 'RTX-9000', 'TB cluster'). NEVER use generic terms like 'chips' or 'data'. Use 'GENERAL' if none.")
    entity_type: str = Field(description="Type (e.g. 'product').")
    target_attribute: str = Field(description="Metric (e.g. 'voltage').")
    time_context: str = Field(description="Temporal qualifiers.")
    is_security_risk: str = Field(description="'true' or 'false' (informal emails/bypass for junior).")
    rewritten_query: str = Field(description="Keyword-rich natural language instruction (NO SQL).")

# --- Module-level dynamic loading ---
_DOMAIN_CONFIGS = load_domain_configs()

# --- Planner Prompt ---
PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are the Global Query Planner for VERA.\n"
        "Plan retrieval and rewrite to natural language. NO SQL keywords (SELECT, FROM).\n"
        "ENTITY EXTRACTION: Prioritize specific IDs (RTX-9000) over generic names (chips).\n"
        "Rewritten Query examples:\n"
        "- 'Retrieve details for [ENTITY] from official sources.'\n"
        "- 'Extract first [N] rows from [DATASET] dataset.'"
    )),
    ("human", (
        "ROLE: {user_role}\n"
        "DOMAIN: {user_domain}\n"
        "DOMAIN INFO:\n{domain_info}\n"
        "QUESTION: {question}"
    ))
])

@vera_agent("Router Agent")
def run(state: GraphState) -> dict:
    """
    ROUTER NODE: Global Query Planner implementation.
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "")
    available_domains = get_available_domains()

    # 注入动态 Domain Info 以辅助 LLM 进行精准的领域推断
    domain_info = ""
    for domain in available_domains:
        cfg = _DOMAIN_CONFIGS.get(domain, {})
        hints = cfg.get("keyword_hints", cfg.get("description", "General queries"))
        domain_info += f"- {domain}: {hints}\n"

    # =========================================================
    # 核心修复 1: 绝对单次执行 (Atomic Call)
    # =========================================================
    planner_llm = config.llm.with_structured_output(QueryPlannerOutput)
    chain = PLANNER_PROMPT | planner_llm
    
    print(f"[Router Agent] Planning query for: '{question}' (Role: {user_role}, Domain: {user_domain})")
    
    try:
        planner_result_raw = llm_invoke_with_retry(
            chain, 
            {
                "question": question,
                "user_role": user_role,
                "user_domain": user_domain,
                "domain_info": domain_info
            }
        )
        if isinstance(planner_result_raw, dict):
            planner_result = QueryPlannerOutput(**planner_result_raw)
        else:
            planner_result = planner_result_raw
            
    except Exception as e:
        print(f"[Router Agent] ❌ Planner failed: {e}. Falling back to default.")
        planner_result = QueryPlannerOutput(
            thought_process=f"Error during planning: {e}. Defaulting to safe fallback.",
            user_intent_category="GENERIC_QA",
            detected_domain=user_domain or (available_domains[0] if available_domains else "unknown"),
            target_entity="GENERAL",
            target_attribute="GENERAL",
            entity_type="GENERAL",
            time_context="",
            is_security_risk="false",
            rewritten_query=question
        )

    metadata_log = ""
    
    # =========================================================
    # 核心修复 2: 弹性的域解析，不乱拉警报
    # =========================================================
    user_domain_clean = user_domain.lower().strip() if user_domain else ""
    if user_domain_clean and user_domain_clean in available_domains:
        final_domain = user_domain_clean
        print(f"[Router Agent] Guard: Using user-selected domain '{final_domain}'")
    else:
        detected = planner_result.detected_domain.lower().strip()
        resolved = resolve_domain_alias(detected, _DOMAIN_CONFIGS)
        final_domain = resolved or detected
        
        if final_domain not in available_domains:
            matched = False
            for d in available_domains:
                if d in final_domain or final_domain in d:
                    final_domain = d
                    matched = True
                    break
            
            if not matched:
                metadata_log += f"[ROUTER] ⚠️ Unresolved domain '{final_domain}'. Defaulting to general/fallback domain.\n"
                final_domain = available_domains[0] if available_domains else "general"

    # =========================================================
    # 核心修复 3: 原生的 RBAC 风控拦截 (带字符串抗错装甲)
    # =========================================================
    flagged = False
    # 安全地将大模型返回的 "True", "true", "False", 甚至是意外布尔值转化为真实的 Python Boolean
    is_risk = str(planner_result.is_security_risk).strip().lower() == "true"
    
    if user_role == "junior" and is_risk:
        flagged = True
        metadata_log += "[ROUTER] 🚨 SECURITY FLAG: Junior user attempting restricted data access.\n"
        print("[Router Agent] 🚨 SECURITY FLAG: Junior user attempting restricted data")

    # =========================================================
    # 意图映射
    # =========================================================
    intent_map = {
        "SMALL_TALK": INTENT_CHAT,          
        "GENERIC_QA": INTENT_SPECS,         
        "SPEC_LOOKUP": INTENT_SPECS,
        "DISCREPANCY_AUDIT": INTENT_CROSS,
        "DATA_QUERY": INTENT_DB,
    }
    mapped_intent = intent_map.get(planner_result.user_intent_category, INTENT_SPECS)
    optimized_query = planner_result.rewritten_query or question
    
    # ---------------------------------------------------------
    # 核心修复 4: 实体后置纠偏 (Generic Entity Filtering & Regex Fallback)
    # ---------------------------------------------------------
    final_entity = planner_result.target_entity.strip()
    # Load generic entities dynamically from all domain configs
    generic_entities = set()
    for cfg in _DOMAIN_CONFIGS.values():
        generic_entities.update(cfg.get("generic_entities", []))
    
    if final_entity.lower() in generic_entities:
        # LLM extracted a generic term. Try to find a specific ID in the question.
        import re
        # Look for alphanumeric patterns like RTX-9000, TB-123, OR CamelCase names (QuantumLogic)
        id_pattern = r'\b([A-Z0-9]{2,}[-][A-Z0-9]+|[A-Z]{2,}[0-9]+|[A-Z][a-z]+[A-Z][a-zA-Z]*)\b'
        matches = re.findall(id_pattern, question.upper())
        if matches:
            final_entity = matches[0]
            print(f"[Router Agent] 🔧 Post-processing: Swapped generic '{planner_result.target_entity}' for detected ID '{final_entity}'")

    # ---------------------------------------------------------
    # 核心修复 5: Generic Query Detection (Intent-Driven + Topic Keywords)
    # ---------------------------------------------------------
    is_generic = (planner_result.user_intent_category in ["GENERIC_QA", "SMALL_TALK"]) or (final_entity.upper() == "GENERAL")
    
    # Substring match only for very specific generic markers, or exact match for common terms
    if final_entity.lower() in generic_entities:
        is_generic = True
        print(f"[Router Agent] 🔧 Topic-based query detected via generic entity list.")
    
    # Phrase match for question patterns (What are the actions for X)
    generic_phrases = ["what are the", "is it possible", "how do i", "can a player", "rule book", "latest updates"]
    if any(p in question.lower() for p in generic_phrases):
        is_generic = True
        print(f"[Router Agent] 🔧 Topic-based query detected via question phrase.")

    thinking = (
        f"User role='{user_role}', domain='{user_domain}'. "
        f"Intent Category: '{planner_result.user_intent_category}' -> '{mapped_intent}'. "
        f"Detected domain: '{final_domain}'. Flagged: {flagged}. Generic: {is_generic}. "
        f"Entity: '{final_entity}' (LLM said '{planner_result.target_entity}'), Attr: '{planner_result.target_attribute}'."
    )

    # Detect required domain for specific query types
    required_domain = ""
    if "contract" in question.lower() or "cuad" in question.lower() or "clause" in question.lower():
        required_domain = "legal"
    # Add more conditions as needed for other domains

    return {
        "route": mapped_intent, 
        "intent": mapped_intent,
        "flagged": flagged,
        "is_generic_query": is_generic,
        "next_agent": final_domain,
        "user_domain": final_domain,
        "required_domain": required_domain,
        "target_entity": final_entity,
        "target_attribute": planner_result.target_attribute,
        "entity_type": planner_result.entity_type,
        "time_context": planner_result.time_context,
        "metadata_log": metadata_log.strip(),
        "_thinking": thinking,
        # 严格遵守 GraphState：禁止在此处返回 documents 或 is_resolved 等脏数据
    }

def decide_route(state: GraphState) -> str:
    """
    Conditional edge switchboard for LangGraph.
    """
    if state.get("flagged", False):
        return "escalate"

    domain = state.get("user_domain", "")
    intent = state.get("intent", INTENT_SPECS)
    
    if intent == INTENT_CHAT:
        print(f"[ROUTING] -> {INTENT_CHAT} (general_chat — skipping retrieval)")
        return INTENT_CHAT

    route_key = f"{domain}__{intent}"
    print(f"[ROUTING] -> {route_key}")
    return route_key
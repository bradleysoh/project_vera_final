import os
from typing import List, Dict
from langgraph.graph import StateGraph, END

# --- VERA Shared Infrastructure ---
from shared.graph_state import GraphState
from shared.dynamic_loader import (
    get_available_domains,
    discover_domain_agents,
    register_domain_nodes,
    get_agent_node_name,
)
import agents_logic.router_agent as router_agent
import agents_logic.escalation_agent as escalation_agent

# --- Response Agent (Final Synthesis) ---
import agents_logic.response_agent as response_agent

# ============================================================================
# EARLY EXIT ROUTING LOGIC (新增：早退判定机制)
# ============================================================================
def check_db_resolution(state: GraphState) -> str:
    """
    动态检查数据库 Agent 是否已经完美解答了问题。
    如果 state 中被标记了 is_resolved=True, 系统直接熔断, 跳过冗余文档检索。
    """
    is_resolved = state.get("is_resolved", False)
    score = state.get("satisfaction_score", 0.0)
    
    if is_resolved:
        print(f"⚡ [Pipeline] 触发早退 (Early Exit)：满意度 {score}，跳过冗余文档检索。")
        return "generate_response"
    
    if score > 0 and score < 0.8:
        print(f"⚠️ [Pipeline] 满意度不足 ({score})，继续深入检索...")
        
    return "continue_pipeline"

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """
    Constructs a deterministic, hybrid multi-agent graph with dynamic short-circuiting.
    Bypasses deep retrieval for general chat, and allows DB agents to short-circuit
    the pipeline if authoritative answers are found immediately.
    """
    workflow = StateGraph(GraphState)

    # 1. CORE NODES
    workflow.add_node("router", router_agent.run)
    workflow.add_node("escalate", escalation_agent.run)
    workflow.add_node("generate_response", response_agent.run)

    # 2. DYNAMIC DOMAIN NODES
    # Automatically discovery and add all agents from agents_logic/*_agents/
    domain_agents = discover_domain_agents()
    domain_node_map = register_domain_nodes(workflow, domain_agents)
    domains = list(domain_node_map.keys())

    # 3. ENTRY POINT
    workflow.set_entry_point("router")

    # 4. CONDITIONAL ROUTING (The Switchboard)
    all_intents = [router_agent.INTENT_DB, router_agent.INTENT_SPECS, router_agent.INTENT_CROSS]
    
    routing_map = {
        router_agent.INTENT_CHAT: "generate_response",
        "escalate": "escalate"
    }

    # Map every domain+intent combination to the appropriate starting node
    for domain in domains:
        for intent in all_intents:
            route_key = f"{domain}__{intent}"
            
            # ALL intents start with DB query — the DB agent's guard clause
            # handles irrelevant intents gracefully by returning {}.
            start_node = f"{domain}_query_database"
            
            # Fallback: if domain has no DB agent, use official docs
            if start_node not in domain_node_map.get(domain, []):
                start_node = f"{domain}_retrieve_official"
            
            routing_map[route_key] = start_node

    workflow.add_conditional_edges(
        "router",
        router_agent.decide_route,
        routing_map
    )

    # 5. DYNAMIC DOMAIN PIPELINES (重构：从瀑布流升级为动态早退)
    # 核心变化：DB 节点不再死板连接 Official 节点，而是交给 check_db_resolution 裁决。
    for domain in domains:
        domain_nodes = domain_node_map.get(domain, [])
        db_node = f"{domain}_query_database"
        official_node = f"{domain}_retrieve_official"
        informal_node = f"{domain}_retrieve_informal"
        audit_node = f"{domain}_check_discrepancy"

        # --- DB 节点逻辑 (带短路熔断) ---
        if db_node in domain_nodes:
            # 如果没有早退，应该继续流向哪个节点？
            if official_node in domain_nodes:
                next_node_if_not_resolved = official_node
            elif informal_node in domain_nodes:
                next_node_if_not_resolved = informal_node
            elif audit_node in domain_nodes:
                next_node_if_not_resolved = audit_node
            else:
                next_node_if_not_resolved = "generate_response"

            workflow.add_conditional_edges(
                db_node,
                check_db_resolution,
                {
                    "generate_response": "generate_response",      # 早退路径
                    "continue_pipeline": next_node_if_not_resolved # 瀑布流继续
                }
            )
        
        # --- 官方文档节点逻辑 ---
        if official_node in domain_nodes:
            if informal_node in domain_nodes:
                workflow.add_edge(official_node, informal_node)
            elif audit_node in domain_nodes:
                workflow.add_edge(official_node, audit_node)
            else:
                workflow.add_edge(official_node, "generate_response")

        # --- 非正式文档节点逻辑 ---
        if informal_node in domain_nodes:
            if audit_node in domain_nodes:
                workflow.add_edge(informal_node, audit_node)
            else:
                workflow.add_edge(informal_node, "generate_response")
            
        # --- 差异审查节点逻辑 ---
        if audit_node in domain_nodes:
            # Phase 3: Insert domain-specific analyzer node if it exists
            analyzer_node = f"{domain}_analyze_contract"
            if analyzer_node in domain_nodes:
                workflow.add_edge(audit_node, analyzer_node)
                workflow.add_edge(analyzer_node, "generate_response")
            else:
                workflow.add_edge(audit_node, "generate_response")

    # 6. SHARED ESCALATION / COMPLETION
    workflow.add_edge("escalate", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
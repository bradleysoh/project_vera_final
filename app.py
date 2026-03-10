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
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """
    Constructs a deterministic, hybrid multi-agent graph.
    Bypasses deep retrieval for general chat, but enforces a strict
    DB -> Specs -> Audit pipeline for technical and cross-reference queries.
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
    # The router returns "general_chat", "escalate", or "domain__intent"
    all_intents = [router_agent.INTENT_DB, router_agent.INTENT_SPECS, router_agent.INTENT_CROSS]
    
    # Generate the mapping for conditional edges
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
            # This ensures CSV/SQLite data is always checked first.
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

    # 5. DETERMINISTIC DOMAIN PIPELINES
    # For each domain, wire the segments together into a rigid audit chain
    for domain in domains:
        # DB -> Official Docs (Hierarchical dependency)
        if f"{domain}_query_database" in domain_node_map[domain] and f"{domain}_retrieve_official" in domain_node_map[domain]:
            workflow.add_edge(f"{domain}_query_database", f"{domain}_retrieve_official")
        
        # Official Docs -> Discrepancy Audit
        if f"{domain}_retrieve_official" in domain_node_map[domain] and f"{domain}_check_discrepancy" in domain_node_map[domain]:
            workflow.add_edge(f"{domain}_retrieve_official", f"{domain}_check_discrepancy")
            
        # Discrepancy Audit -> Final Response
        if f"{domain}_check_discrepancy" in domain_node_map[domain]:
            workflow.add_edge(f"{domain}_check_discrepancy", "generate_response")

    # 6. SHARED ESCALATION / COMPLETION
    workflow.add_edge("escalate", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
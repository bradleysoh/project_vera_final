import os
import time
from datetime import datetime
from langgraph.graph import StateGraph, END

# --- Import shared state and config ---
from shared.graph_state import GraphState
from shared.config import LLM_BACKEND, OLLAMA_MODEL
from shared.dynamic_loader import (
    discover_domain_agents,
    get_agent_node_name,
    register_domain_nodes,
    EXPECTED_AGENT_ROLES,
)

# --- Import shared (non-domain-specific) agents ---
from agents_logic import router_agent
from agents_logic import response_agent
from agents_logic import escalation_agent
from shared.system_logging import log_vera_step

# ==============================================================================
# AGENT TRACING (For Streamlit Timeline)
# ==============================================================================
# This global list allows the Streamlit UI to display agent steps in real-time
AGENT_TRACE = []

def log_agent_action(agent, action):
    """Utility to track agent movements for the UI Timeline."""
    AGENT_TRACE.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "action": action
    })

# ==============================================================================
# BUILD THE LANGGRAPH WORKFLOW
# ==============================================================================

def build_graph():
    """
    Construct the VERA LangGraph workflow with dynamically loaded domain agents.
    Supports: semiconductor, medical, aerospace, energy, finance.
    """
    # Step 1: Discover domain agents (Auto-detects your new folders!)
    domain_agents = discover_domain_agents()

    if not domain_agents:
        print("[GRAPH] ⚠️ No domain agent folders found!")

    workflow = StateGraph(GraphState)

    # Step 2: Add shared nodes
    workflow.add_node("route_query", router_agent.run)
    workflow.add_node("generate_response", response_agent.run)
    workflow.add_node("escalate", escalation_agent.run)

    # Step 3: Dynamically register domain agent nodes
    # This adds the nodes like medical_official_docs_agent to the graph
    register_domain_nodes(workflow, domain_agents)

    # Step 4: Entry point
    workflow.set_entry_point("route_query")

    # Step 5: Intent-Based Conditional Routing
    # We explicitly map domain__intent to ensure KeyError is resolved
    route_map = {
        "escalate": "escalate",
        "general_chat": "generate_response",
    }

    for domain, agents in domain_agents.items():
        # Get node names for this domain
        db_node = get_agent_node_name(domain, "db_agent")
        spec_node = get_agent_node_name(domain, "official_docs_agent") or get_agent_node_name(domain, "tech_spec_agent")
        comp_node = get_agent_node_name(domain, "informal_docs_agent") or get_agent_node_name(domain, "compliance_agent")
        audit_node = get_agent_node_name(domain, "discrepancy_agent")

        # --- Map Intents to specific domain nodes ---
        
        # Routing for DB Agents
        if db_node:
            route_map[f"{domain}__db_query"] = db_node
            workflow.add_edge(db_node, "generate_response")

        # Routing for Specification Retrieval
        if spec_node:
            # Fixing the KeyError by adding this specific mapping for all domains
            route_map[f"{domain}__spec_retrieval"] = spec_node
            
            # Sub-routing: Spec -> Auditor (if cross-referencing) or Response
            def make_spec_router(d_node):
                def _router(state: GraphState):
                    if state.get("intent") == "cross_reference" and d_node:
                        return "audit"
                    return "respond"
                return _router
            
            workflow.add_conditional_edges(
                spec_node, 
                make_spec_router(audit_node), 
                {"audit": audit_node, "respond": "generate_response"} if audit_node else {"respond": "generate_response"}
            )

        # Routing for Compliance/Informal Docs
        if comp_node:
            route_map[f"{domain}__compliance_retrieval"] = comp_node
            workflow.add_edge(comp_node, "generate_response")

        # Routing for Discrepancy Audits
        if audit_node:
            workflow.add_edge(audit_node, "generate_response")

    # Final routing from the main Router Agent
    workflow.add_conditional_edges("route_query", router_agent.decide_route, route_map)

    # Final Connections to Responder or Escalation
    workflow.add_edge("generate_response", END)
    workflow.add_edge("escalate", END)

    compiled_graph = workflow.compile()
    print(f"\n[GRAPH] VERA compiled successfully for: {list(domain_agents.keys())}")
    return compiled_graph

# ==============================================================================
# EXECUTION
# ==============================================================================

def run_test_query(graph, question, user_role, user_domain):
    """Executes query and logs actions for the Streamlit Timeline"""
    AGENT_TRACE.clear()
    log_agent_action("Router Agent", f"Intent classification for {user_domain} domain")
    
    initial_state = {
        "question": question,
        "user_role": user_role,
        "user_domain": user_domain,
        "documents": [],
        "generation": "",
        "intent": "",
        "flagged": False,
        "thought_process": []
    }

    result = graph.invoke(initial_state)
    
    # Trace security and retrieval logic for the UI
    if result.get("flagged"):
        log_agent_action("Security Agent", "RBAC Violation Detected -> ESCALATED")
    else:
        log_agent_action("Retrieval Agent", f"Accessing {user_domain} documentation")
        if result.get("has_discrepancy"):
            log_agent_action("Discrepancy Agent", "Detected data conflict in sources")

    log_vera_step(result, user_domain)
    return result

if __name__ == "__main__":
    graph = build_graph()
    # Test call for Medical domain
    run_test_query(graph, "What is the readmission risk?", "senior", "medical")
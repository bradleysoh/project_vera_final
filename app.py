"""
================================================================================
Project VERA — Central Orchestrator (Dynamic Loader Version)
Virtual Engineering Review Agent
================================================================================

This is the CENTRAL app.py that dynamically discovers and imports agent modules
from domain subfolders under agents_logic/ (e.g., semiconductor_agents/,
public_health_agents/) and connects them using LangGraph's StateGraph.

Architecture Overview:
  START -> route_query -> (conditional)
    -> {domain}_retrieve_specs     -> generate_response -> {domain}_check_discrepancy -> END
    -> {domain}_retrieve_compliance -> generate_response -> {domain}_check_discrepancy -> END
    -> escalate -> END

Domain agents are auto-discovered from agents_logic/*_agents/ subfolders.
Shared agents (router, response, escalation) are imported directly.
The state schema (GraphState) is in shared/graph_state.py.

Usage:
    python app.py
================================================================================
"""

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


# ==============================================================================
# BUILD THE LANGGRAPH WORKFLOW
# ==============================================================================

def build_graph():
    """
    Construct the VERA LangGraph workflow with dynamically loaded domain agents.

    1. Discovers domain agent subfolders (semiconductor_agents/, medical_agents/)
    2. Registers each domain's agents as {domain}_{role} nodes
    3. Wires shared agents (router, response, escalation) at the top level
    4. Creates conditional routing from router -> domain-specific agents

    Returns:
        Compiled StateGraph ready for execution.
    """
    # --- Step 1: Discover domain agents ---
    domain_agents = discover_domain_agents()

    if not domain_agents:
        print("[GRAPH] ⚠️  No domain agent folders found! Using fallback.")

    workflow = StateGraph(GraphState)

    # --- Step 2: Add shared (non-domain) nodes ---
    workflow.add_node("route_query", router_agent.run)
    workflow.add_node("generate_response", response_agent.run)
    workflow.add_node("escalate", escalation_agent.run)

    # --- Step 3: Dynamically register domain agent nodes ---
    domain_nodes = register_domain_nodes(workflow, domain_agents)

    # --- Step 4: Entry point ---
    workflow.set_entry_point("route_query")

    # --- Step 5: Intent-Based Conditional Routing (Surgical) ---
    # Instead of broadcasting to ALL agents, route to specific subchains
    # based on fine-grained intent from the router.
    #
    # Intent → Subchain:
    #   general_chat    → generate_response (skip all retrieval)
    #   db_query        → DB → generate_response
    #   spec_retrieval  → Official → generate_response
    #   cross_reference → DB → Official → Informal → generate_response → Discrepancy
    #   escalate        → escalate → END

    route_map = {
        "escalate": "escalate",
        "general_chat": "generate_response",
    }

    for domain, agents in domain_agents.items():
        # Resolve node names for this domain's agents
        db_node = get_agent_node_name(domain, "db_agent") if "db_agent" in agents else None
        official_node = (
            get_agent_node_name(domain, "official_docs_agent")
            if "official_docs_agent" in agents
            else get_agent_node_name(domain, "tech_spec_agent")
            if "tech_spec_agent" in agents
            else None
        )
        informal_node = (
            get_agent_node_name(domain, "informal_docs_agent")
            if "informal_docs_agent" in agents
            else get_agent_node_name(domain, "compliance_agent")
            if "compliance_agent" in agents
            else None
        )

        if not any([db_node, official_node, informal_node]):
            print(f"[GRAPH] ⚠️ No agents found for domain '{domain}', skipping wiring.")
            continue

        # --- Wire domain agents with CONDITIONAL edges (strictly sequential) ---
        # Each agent checks state["intent"] to decide its SINGLE successor.
        # This prevents LangGraph from following multiple edges (forking).

        # --- Routing logic for Discrepancy Agent (Auditor) ---
        def route_to_auditor(domain: str, fallback_node: str):
            def _router(state: GraphState) -> str:
                if state.get("intent") == "cross_reference":
                    # Find discrepancy node for this domain
                    agents = domain_agents.get(domain, {})
                    if "discrepancy_agent" in agents:
                        return "to_auditor"
                return "to_response"
            return _router

        # DB Agent → (Official or Auditor or Response)
        if db_node:
            route_map[f"{domain}__db_query"] = db_node
            route_map[f"{domain}__cross_reference"] = db_node
            
            if official_node:
                def _db_router(state: GraphState) -> str:
                    if state.get("intent") == "cross_reference":
                        return "to_official"
                    return "to_response"
                workflow.add_conditional_edges(db_node, _db_router, {"to_official": official_node, "to_response": "generate_response"})
            else:
                workflow.add_conditional_edges(db_node, route_to_auditor(domain, "generate_response"), 
                                                {"to_auditor": get_agent_node_name(domain, "discrepancy_agent"), "to_response": "generate_response"})

        # Official Agent → (Informal or Auditor or Response)
        if official_node:
            route_map[f"{domain}__spec_retrieval"] = official_node
            if f"{domain}__cross_reference" not in route_map:
                route_map[f"{domain}__cross_reference"] = official_node

            if informal_node:
                def _off_router(state: GraphState) -> str:
                    if state.get("intent") == "cross_reference":
                        return "to_informal"
                    return "to_response"
                workflow.add_conditional_edges(official_node, _off_router, {"to_informal": informal_node, "to_response": "generate_response"})
            else:
                workflow.add_conditional_edges(official_node, route_to_auditor(domain, "generate_response"), 
                                                {"to_auditor": get_agent_node_name(domain, "discrepancy_agent"), "to_response": "generate_response"})

        # Informal Agent → (Auditor or Response)
        if informal_node:
            if f"{domain}__cross_reference" not in route_map:
                route_map[f"{domain}__cross_reference"] = informal_node
            
            workflow.add_conditional_edges(informal_node, route_to_auditor(domain, "generate_response"), 
                                            {"to_auditor": get_agent_node_name(domain, "discrepancy_agent"), "to_response": "generate_response"})

    # --- Conditional edge from router to intent-based subchains ---
    workflow.add_conditional_edges(
        "route_query",
        router_agent.decide_route,
        route_map,
    )

    # --- Auditor → Responder ---
    for domain, agents in domain_agents.items():
        if "discrepancy_agent" in agents:
            d_node = get_agent_node_name(domain, "discrepancy_agent")
            # Auditor ALWAYS goes to Responder next
            workflow.add_edge(d_node, "generate_response")

    # --- Responder → END ---
    workflow.add_edge("generate_response", END)

    # Escalation → END
    workflow.add_edge("escalate", END)

    compiled_graph = workflow.compile()

    # Summary
    all_nodes = list(compiled_graph.nodes.keys())
    print(f"\n[GRAPH] VERA workflow compiled successfully!")
    print(f"[GRAPH] Nodes ({len(all_nodes)}): {all_nodes}")
    print(f"[GRAPH] Domains: {list(domain_agents.keys())}")
    print(f"[GRAPH] Route map: {route_map}")

    return compiled_graph


# ==============================================================================
# LOGGING UTILITIES
# ==============================================================================

def _log_to_files(state: dict, domain: str) -> None:
    """
    Log test results to domain-specific files in the output/ directory.
    - verification.log: question + generation
    - discrepancy.log: discrepancy_report
    - debug.log: thought_process + metadata_log
    """
    output_base = os.path.join(os.path.dirname(__file__), "output")
    domain_dir = os.path.join(output_base, domain)
    os.makedirs(domain_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 80

    # 1. Verification Log
    with open(os.path.join(domain_dir, "verification.log"), "a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n[{timestamp}] QUESTION: {state.get('question')}\n{separator}\n")
        f.write(f"{state.get('generation', 'No generation.')}\n")

    # 2. Discrepancy Log
    report = state.get("discrepancy_report")
    if report:
        with open(os.path.join(domain_dir, "discrepancy.log"), "a", encoding="utf-8") as f:
            f.write(f"\n{separator}\n[{timestamp}] AUDIT: {state.get('target_entity')}\n{separator}\n")
            f.write(f"{report}\n")

    # 3. Debug Log
    with open(os.path.join(domain_dir, "debug.log"), "a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n[{timestamp}] DEBUG: {state.get('question')}\n{separator}\n")
        f.write("--- THOUGHT PROCESS ---\n")
        for step in state.get("thought_process", []):
            f.write(f"  {step}\n")
        f.write("\n--- METADATA LOG ---\n")
        f.write(f"{state.get('metadata_log', 'No metadata log.')}\n")


# ==============================================================================
# TEST EXECUTION
# ==============================================================================

def run_test_query(
    graph,
    question: str,
    user_role: str,
    user_domain: str,
    test_number: int,
    test_description: str,
) -> None:
    """Execute a test query through the VERA graph and display the results."""
    print(f"\n{'#'*70}")
    print(f"# TEST {test_number}: {test_description}")
    print(f"# User Role: {user_role.upper()} | Domain: {user_domain.upper()}")
    print(f"# Question: {question}")
    print(f"{'#'*70}")

    initial_state = {
        "question": question,
        "generation": "",
        "user_role": user_role,
        "user_domain": user_domain,
        "documents": [],
        "route": "",
        "intent": "",
        "flagged": False,
        "metadata_log": "",
        "retrieved_docs": {},
        "db_result": "",
        "db_data": "",
        "discrepancy_report": "",
        "next_agent": "",
        "thought_process": [],
        "refinement_count": 0,
        "max_refinements": 3,
        "critique": "",
        "retrieval_confidence": "",
        # Structured Fact Passing fields
        "target_entity": "",
        "entity_type": "",
        "target_attribute": "",
        "time_context": "",
        "official_facts": [],
        "informal_facts": [],
        "db_facts": [],
        "discrepancy_verdict": {},
        "official_data": [],
        "informal_data": [],
        "latest_timestamp": "",
    }


    result = graph.invoke(initial_state)

    # --- Systematic Logging (Phase 8) ---
    _log_to_files(result, user_domain)

    print(f"\n{'─'*60}")
    print(f"📊 RESULT (Test {test_number}):")
    print(f"{'─'*60}")
    print(f"\n{result.get('generation', 'No response generated.')}")

    if result.get("discrepancy_report"):
        print(f"\n{'─'*60}")
        print(f"📋 Discrepancy Report:")
        print(f"{'─'*60}")
        print(result["discrepancy_report"])

    # --- Thought Process (DeepSeek-style) ---
    thought_process = result.get("thought_process", [])
    if thought_process:
        print(f"\n{'─'*60}")
        print(f"🧠 Agent Thinking ({len(thought_process)} steps):")
        print(f"{'─'*60}")
        for step in thought_process:
            print(f"  {step}")

    if result.get("metadata_log"):
        print(f"\n{'─'*60}")
        print(f"📋 Retrieval Metadata Log:")
        print(f"{'─'*60}")
        print(result["metadata_log"])

    docs = result.get("documents", [])
    if docs:
        sources = [d.metadata.get("source") for d in docs]
        access_levels = [d.metadata.get("access_level") for d in docs]
        domains = set(d.metadata.get("domain", "N/A") for d in docs)
        print(f"\n📄 Documents Retrieved: {len(docs)}")
        print(f"   Sources: {sources}")
        print(f"   Access Levels: {access_levels}")
        print(f"   Domains: {domains}")

    print(f"\n🔀 Route: {result.get('route')} | Domain: {result.get('next_agent')} | Flagged: {result.get('flagged')}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution function demonstrating the VERA multi-agent system.

    Runs 6 test scenarios:
      1. Senior engineer — full access technical query
      2. Junior intern — restricted access (public only)
      3. Junior intern — unauthorized access attempt (ESCALATION)
      4. Senior engineer — compliance query (SOPs + emails)
      5. Senior engineer — DB info + document version discrepancy
      6. Senior engineer — OUT-OF-DOMAIN (semiconductor user, public health query)
    """
    print("=" * 70)
    print("PROJECT VERA - Virtual Engineering Review Agent")
    print("Multi-Agent System Demo (Dynamic Loader Architecture)")
    print("=" * 70)

    graph = build_graph()

    # TEST 1: Senior — Full Access Technical Specs
    run_test_query(
        graph=graph,
        question="What is the maximum voltage limit for the RTX-9000?",
        user_role="senior",
        user_domain="semiconductor",
        test_number=1,
        test_description="Senior Engineer - Technical Query (Full Access)",
    )

    print("\n⏳ Waiting 15s between tests (rate limit protection)...")
    time.sleep(15)

    # TEST 2: Junior — Restricted Access
    run_test_query(
        graph=graph,
        question="What is the maximum voltage limit for the RTX-9000?",
        user_role="junior",
        user_domain="semiconductor",
        test_number=2,
        test_description="Junior Intern - Same Query (Public Only - RBAC Filter)",
    )

    print("\n⏳ Waiting 15s between tests (rate limit protection)...")
    time.sleep(15)

    # TEST 3: Junior — Unauthorized Access (ESCALATION)
    run_test_query(
        graph=graph,
        question="Were there any internal emails about skipping burn-in tests?",
        user_role="junior",
        user_domain="semiconductor",
        test_number=3,
        test_description="Junior Intern - Unauthorized Access Attempt (ESCALATION)",
    )

    print("\n⏳ Waiting 15s between tests (rate limit protection)...")
    time.sleep(15)

    # TEST 4: Senior — Compliance Query
    run_test_query(
        graph=graph,
        question=(
            "What are the quality audit procedures, and have there been "
            "any recent changes communicated via email?"
        ),
        user_role="senior",
        user_domain="semiconductor",
        test_number=4,
        test_description="Senior Engineer - Compliance Query (SOPs + Emails)",
    )

    print("\n⏳ Waiting 15s between tests (rate limit protection)...")
    time.sleep(15)

    # TEST 5: Senior — DB Info + Document Version Comparison
    run_test_query(
        graph=graph,
        question=(
            "Compare the RTX-9000 specification versions and check the "
            "production database for any lots that don't match the latest spec."
        ),
        user_role="senior",
        user_domain="semiconductor",
        test_number=5,
        test_description="Senior Engineer - DB Info + Document Version Discrepancy",
    )

    print("\n⏳ Waiting 15s between tests (rate limit protection)...")
    time.sleep(15)

    # TEST 6: Senior — OUT-OF-DOMAIN (semiconductor user, public health question)
    run_test_query(
        graph=graph,
        question="What are the CDC guidelines for preventing the spread of infectious diseases?",
        user_role="senior",
        user_domain="semiconductor",
        test_number=6,
        test_description="Senior Engineer - OUT-OF-DOMAIN Query (ESCALATION)",
    )

    print(f"\n{'='*70}")
    print("VERA Demo Complete - All test scenarios executed.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

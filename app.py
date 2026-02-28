"""
================================================================================
Project VERA — Central Orchestrator (Dynamic Loader Version)
Virtual Engineering Review Agent
================================================================================

This is the CENTRAL app.py that dynamically discovers and imports agent modules
from domain subfolders under agents_logic/ (e.g., semiconductor_agents/,
medical_agents/) and connects them using LangGraph's StateGraph.

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

import time
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

    # --- Step 6: Wire domain-specific agent chains (SEQUENTIAL) ---
    # DB (Reality) -> Official (Baseline) -> Informal (Exception) -> Response
    
    # Also build the route map: domain -> start_node
    route_map = {"escalate": "escalate"}

    for domain, agents in domain_agents.items():
        # Identify available nodes
        chain = []
        
        # Priority 1: DB Agent
        if "db_agent" in agents:
            chain.append(get_agent_node_name(domain, "db_agent"))
            
        # Priority 2: Official Docs (was tech_spec/compliance)
        if "official_docs_agent" in agents:
            chain.append(get_agent_node_name(domain, "official_docs_agent"))
        elif "tech_spec_agent" in agents: # Fallback support
            chain.append(get_agent_node_name(domain, "tech_spec_agent"))
            
        # Priority 3: Informal Docs
        if "informal_docs_agent" in agents:
            chain.append(get_agent_node_name(domain, "informal_docs_agent"))
        elif "compliance_agent" in agents: # Fallback support
            chain.append(get_agent_node_name(domain, "compliance_agent"))

        # Always end with response generation
        chain.append("generate_response")

        if not chain[:-1]: # If no domain agents found, skip
            print(f"[GRAPH] ⚠️ No agents found for domain '{domain}', skipping wiring.")
            continue

        # Wire the sequential chain
        print(f"[GRAPH] Wiring chain for {domain}: {' -> '.join(chain)}")
        for i in range(len(chain) - 1):
            workflow.add_edge(chain[i], chain[i+1])

        # Map the domain name to the START of the chain
        start_node = chain[0]
        route_map[domain] = start_node

    # --- Step 5: Conditional Routing (Updated) ---
    # The router returns "semiconductor" -> mapped to start_node
    workflow.add_conditional_edges(
        "route_query",
        router_agent.decide_route,
        route_map,
    )

    # Response -> Discrepancy (CONDITIONAL — must route to correct domain)
    # We use a conditional edge because generate_response is shared across
    # all domains, so we need to dispatch based on next_agent.
    
    def route_from_discrepancy(state: GraphState) -> str:
        """
        Route from Discrepancy Agent:
        - If 'critique' exists -> Loop back to 'generate_response' for refinement.
        - Else -> END.
        """
        if state.get("critique"):
            count = state.get("refinement_count", 0)
            print(f"[ROUTING] {state.get('next_agent')}_check_discrepancy -> Refinement Loop ({count})")
            return "generate_response"
        return END

    discrepancy_route_map = {}
    for domain, agents in domain_agents.items():
        if "discrepancy_agent" in agents:
            discrepancy_node = get_agent_node_name(domain, "discrepancy_agent")
            discrepancy_route_map[domain] = discrepancy_node
            
            # Conditional edge for feedback loop
            workflow.add_conditional_edges(
                discrepancy_node,
                route_from_discrepancy,
                {"generate_response": "generate_response", END: END}
            )

    def route_to_discrepancy(state: GraphState) -> str:
        """Route from generate_response to the correct domain's discrepancy agent."""
        domain = state.get("next_agent", "")
        if domain in discrepancy_route_map:
            print(f"[ROUTING] generate_response -> {discrepancy_route_map[domain]}")
            return domain
        # Fallback to first available domain
        if discrepancy_route_map:
            fallback_key = next(iter(discrepancy_route_map.keys()))
            print(f"[ROUTING] generate_response -> {discrepancy_route_map[fallback_key]} (fallback)")
            return fallback_key
        return END

    # Include END in the route map for cases where no domain match exists
    discrepancy_route_with_end = dict(discrepancy_route_map)
    discrepancy_route_with_end[END] = END

    workflow.add_conditional_edges(
        "generate_response",
        route_to_discrepancy,
        discrepancy_route_with_end,
    )

    # Escalation -> END
    workflow.add_edge("escalate", END)

    compiled_graph = workflow.compile()

    # Summary
    all_nodes = list(compiled_graph.nodes.keys())
    print(f"\n[GRAPH] VERA workflow compiled successfully!")
    print(f"[GRAPH] Nodes ({len(all_nodes)}): {all_nodes}")
    print(f"[GRAPH] Domains: {list(domain_agents.keys())}")

    return compiled_graph


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
      6. Senior engineer — OUT-OF-DOMAIN (semiconductor user, medical query)
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

    # TEST 6: Senior — OUT-OF-DOMAIN (semiconductor user, medical question)
    run_test_query(
        graph=graph,
        question="What are the FDA clinical trial requirements for our device?",
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

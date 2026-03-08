import os
from shared.graph_state import GraphState
from agents_logic.router_agent import run as router_run, decide_route
from agents_logic.medical_agents.official_docs_agent import run as medical_run
from shared.dynamic_loader import get_available_domains

def test_reproduction():
    print("--- Domain Bleed Reproduction Test ---")
    
    available = get_available_domains()
    print(f"Available domains: {available}")
    
    # Simulate a user in 'medical' asking about 'RTX-9000' (which is semiconductor)
    question = "What's the maximum voltage for RTX-9000?"
    state = {
        "question": question,
        "user_role": "senior",
        "user_domain": "medical",
        "documents": [],
        "metadata_log": "",
        "thought_process": [],
    }

    print(f"\n[1] Running Router Agent as 'medical'...")
    router_out = router_run(state)
    print(f"    - Detected Domain: {router_out.get('next_agent')}")
    print(f"    - Updated State Domain: {router_out.get('user_domain')}")
    print(f"    - Route Choice: {decide_route({**state, **router_out})}")

    # Update state
    state.update(router_out)

    if router_out.get('next_agent') == 'medical':
        print(f"\n[2] Running Medical Docs Agent (should return 0 docs)...")
        medical_out = medical_run(state)
        docs = medical_out.get("documents", [])
        print(f"    - Docs retrieved: {len(docs)}")
        for i, d in enumerate(docs):
            print(f"      [{i+1}] Title: {d.metadata.get('title')} | Source: {d.metadata.get('source')} | Domain: {d.metadata.get('domain')}")
            print(f"          Content Preview: {d.page_content[:150].replace(chr(10), ' ')}...")
        
        if any(d.metadata.get('domain') == 'semiconductor' for d in docs):
            print("❌ BLEED DETECTED: Semiconductor docs retrieved in Medical agent!")
        elif len(docs) > 0:
            print(f"⚠️  Unexpected docs retrieved: {len(docs)}")
        else:
            print("✅ SUCCESS: No docs retrieved (as expected).")
    else:
        print(f"❌ ROUTER ERROR: Switched domain to {router_out.get('next_agent')}")

if __name__ == "__main__":
    test_reproduction()

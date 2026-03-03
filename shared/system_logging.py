"""
================================================================================
Project VERA — Centralized System Logging
================================================================================
Provides a uniform logging interface for CLI and Web conversations, ensuring
that logs are always organized into domain-specific subfolders.
================================================================================
"""

import os
from datetime import datetime

def log_vera_step(state: dict, domain: str) -> None:
    """
    Log VERA execution results to domain-specific files in the output/ directory.
    - verification.log: question + generation
    - discrepancy.log: discrepancy_report
    - debug.log: thought_process + metadata_log
    
    Args:
        state: The final GraphState (dict) after an execution run.
        domain: The domain folder to log into (from user_domain or next_agent).
    """
    # Ensure we use a safe domain name
    safe_domain = domain.lower().replace(" ", "_") if domain else "general"
    
    # Path setup: /project_vera/output/{domain}/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_base = os.path.join(project_root, "output")
    domain_dir = os.path.join(output_base, safe_domain)
    
    os.makedirs(domain_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 80

    # 1. Verification Log (Final Answer)
    generation = state.get("generation")
    if generation:
        with open(os.path.join(domain_dir, "verification.log"), "a", encoding="utf-8") as f:
            f.write(f"\n{separator}\n[{timestamp}] QUESTION: {state.get('question')}\n{separator}\n")
            f.write(f"{generation}\n")

    # 2. Discrepancy Log (Audit Findings)
    report = state.get("discrepancy_report")
    if report:
        with open(os.path.join(domain_dir, "discrepancy.log"), "a", encoding="utf-8") as f:
            f.write(f"\n{separator}\n[{timestamp}] AUDIT: {state.get('target_entity', 'GENERAL')}\n{separator}\n")
            f.write(f"{report}\n")

    # 3. Debug Log (Internal Traces)
    with open(os.path.join(domain_dir, "debug.log"), "a", encoding="utf-8") as f:
        f.write(f"\n{separator}\n[{timestamp}] DEBUG: {state.get('question')}\n{separator}\n")
        
        # Thought Process
        f.write("--- THOUGHT PROCESS ---\n")
        thought_process = state.get("thought_process")
        if isinstance(thought_process, list):
            for step in thought_process:
                f.write(f"  {step}\n")
        else:
            f.write(f"  {thought_process}\n")
            
        # Metadata Log
        f.write("\n--- METADATA LOG ---\n")
        metadata_log = state.get("metadata_log", "No metadata log available.")
        f.write(f"{metadata_log}\n")
        
        # Route Info
        route = state.get("route", "unknown")
        intent = state.get("intent", "unknown")
        f.write(f"\n--- ROUTING ---\n")
        f.write(f"Route: {route} | Intent: {intent} | Next Agent: {state.get('next_agent')}\n")

    print(f"[SYSTEM LOG] 📝 Logged conversation to output/{safe_domain}/")

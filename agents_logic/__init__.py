# agents_logic/ — Individual agent modules for Project VERA
# Each agent MUST expose a run(state: GraphState) -> dict function.
#
# Structure:
#   agents_logic/
#   ├── router_agent.py            # Shared — intent + domain routing
#   ├── response_agent.py          # Shared — LLM response generation
#   ├── escalation_agent.py        # Shared — security escalation
#   ├── _template_agent.py         # Template for creating new agents
#   ├── semiconductor_agents/      # Domain-specific agents
#   │   ├── tech_spec_agent.py
#   │   ├── compliance_agent.py
#   │   └── discrepancy_agent.py
#   └── medical_agents/            # Domain-specific agents (placeholder)
#       ├── tech_spec_agent.py
#       ├── compliance_agent.py
#       └── discrepancy_agent.py

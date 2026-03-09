"""
Legal Informal Docs Agent.

Placeholder for legal email/memo retrieval; currently no-op unless the domain
adds informal legal sources.
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent


@vera_agent("Legal Informal Docs Agent")
def run(state: GraphState) -> dict:
    return {}


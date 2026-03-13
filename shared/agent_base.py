"""
================================================================================
Project VERA — Agent Base Decorator
================================================================================

Provides a `@vera_agent` decorator that enforces the agent contract:
  1. Validates the function accepts GraphState and returns dict
  2. Provides automatic entry/exit logging
  3. Reports which state fields were updated

Usage:
    from shared.agent_base import vera_agent

    @vera_agent("My Agent Name")
    def run(state: GraphState) -> dict:
        ...
        return {"generation": "result"}
================================================================================
"""

import functools
from shared.graph_state import GraphState


def vera_agent(name: str):
    """
    Decorator that wraps agent functions with logging + validation.

    Args:
        name: Human-readable agent name for log output (e.g. "Tech Spec Agent").

    Usage:
        @vera_agent("Router Agent")
        def run(state: GraphState) -> dict:
            # ... your logic ...
            return {"route": "technical", "flagged": False}
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state: GraphState) -> dict:
            question = state.get("question", "")
            user_role = state.get("user_role", "unknown")
            print(f"\n[{name}] Processing...")
            print(f"[{name}] Question: {question[:80]}{'...' if len(question) > 80 else ''}")
            print(f"[{name}] User role: {user_role}")

            result = func(state)

            # --- Validation ---
            if result is None:
                print(f"[{name}] ❌ ERROR: Agent returned None!")
            elif not isinstance(result, dict):
                print(f"[{name}] ❌ ERROR: Agent returned {type(result)} instead of dict")
            
            if not isinstance(result, dict):
                raise TypeError(
                    f"[{name}] Agent must return dict, got {type(result).__name__}"
                )

            # --- Thought Process (DeepSeek-style internal monologue) ---
            # If the agent set result["_thinking"], append it to thought_process
            thinking = result.pop("_thinking", None)
            if thinking:
                thought_process = list(state.get("thought_process") or [])
                thought_process.append(f"💭 {name}: {thinking}")
                result["thought_process"] = thought_process
                print(f"[{name}] 🧠 Thinking: {thinking[:100]}...")

            print(f"[{name}] Done. Updated fields: {list(result.keys())}")
            return result

        # Attach metadata for introspection
        wrapper.__agent_name__ = name
        wrapper.__wrapped_func__ = func
        return wrapper

    return decorator

# Attach a static label to the decorator factory itself for logging convenience
vera_agent.label = "VERA"
"""
================================================================================
Example usage in an agent file:
================================================================================

    from shared.agent_base import vera_agent
    from shared.graph_state import GraphState
    from shared.config import retrieve_with_rbac

    @vera_agent("Tech Spec Agent")
    def run(state: GraphState) -> dict:
        documents, metadata_log = retrieve_with_rbac(
            query=state["question"],
            user_role=state["user_role"],
            source_filter=["datasheet"],
            k=4,
        )
        return {"documents": documents, "metadata_log": metadata_log}

================================================================================
"""

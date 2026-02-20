"""
================================================================================
Discrepancy Detection Agent (The Referee) — Hierarchical & Entity-Scoped
================================================================================
OWNER: (Member 4)
DOMAIN: semiconductor (logic is domain-agnostic)
RESPONSIBILITY: 
  1. Detect conflicts using strict HIERARCHY OF AUTHORITY rules.
  2. Scope analysis to the SPECIFIC ENTITY only.
  3. Enforce Timeline Evaluation (Newer informal data > Older official data).
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
import shared.config as config
from shared.config import llm_invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Entity extraction — lightweight LLM call
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "Extract the PRIMARY entity (product name, patient ID, lot number, "
        "component, or subject) from the user's question below.\n\n"
        "Return ONLY the entity name/ID — nothing else. If the question is "
        "general and has no specific entity, return: GENERAL_QUERY\n\n"
        "Question: {question}"
    ))
])


def _extract_entity(question: str) -> str:
    """Use the LLM to extract the primary entity from the user's question."""
    chain = _ENTITY_PROMPT | config.llm | StrOutputParser()
    raw = llm_invoke_with_retry(chain, {"question": question})
    entity = raw.strip().strip('"').strip("'")
    return entity


def _format_docs(docs: list) -> str:
    """Helper to format a list of documents for the prompt."""
    if not docs:
        return "(None)"
    formatted = []
    for doc in docs:
        src = doc.metadata.get('source', 'unknown').upper()
        ver = doc.metadata.get('version', 'unknown')
        date = doc.metadata.get('date', 'unknown')
        formatted.append(f"[{src}] (Ver: {ver}, Date: {date}):\n{doc.page_content[:500]}")
    return "\n\n".join(formatted)


@vera_agent("Case Agent (Referee)")
def run(state: GraphState) -> dict:
    """
    CASE AGENT: Hierarchical Discrepancy & Timeline Analysis.
    """
    question = state["question"]
    generation = state["generation"]
    domain = state.get("user_domain", "unknown")
    
    # ── Step 0: Gather Hierarchical Data ───────────────────────────────
    db_data = state.get("db_data", "(No Database Results)")
    official_docs = state.get("official_data", [])
    informal_docs = state.get("informal_data", [])
    
    # Fallback if new agents haven't run (backward compatibility)
    if not official_docs and not informal_docs:
        all_docs = state.get("documents", [])
        official_docs = [d for d in all_docs if d.metadata.get('source') in ['datasheet', 'sop', 'spec']]
        informal_docs = [d for d in all_docs if d.metadata.get('source') in ['email', 'memo', 'dm']]

    # ── Step 1: Extract the primary entity ──────────────────────────────
    entity = _extract_entity(question)
    is_general = entity == "GENERAL_QUERY"
    print(f"[Case Agent ({domain})] Refereeing for Entity: {entity}")

    # ── Step 2: Hierarchical "Referee" Prompt ───────────────────────────
    formatted_official = _format_docs(official_docs)
    formatted_informal = _format_docs(informal_docs)
    
    entity_scope_instruction = (
        f"ENTITY SCOPE: The user is asking about \"{entity}\". "
        f"IGNORE information about other entities.\n"
    ) if not is_general else ""

    discrepancy_prompt = ChatPromptTemplate.from_messages([
        ("human", (
            "You are the 'REFEREE' agent. Your task is to audit data sources for discrepancies using a strict logic framework.\n\n"
            f"{entity_scope_instruction}\n"
            "***\n"
            "STEP 1: INTERNAL MAPPING (Chain of Thought)\n"
            "Build a mental matrix of all retrieved data points:\n"
            "[Entity] | [Attribute] | [Value] | [Source_Type] | [Timestamp/Version]\n\n"
            "STEP 2: CONFLICT RESOLUTION ALGORITHM\n"
            "A. Group values by Entity/Attribute.\n"
            "B. Are values identical? If YES -> Status: ALIGNED.\n"
            "C. If NO -> Apply Hierarchy:\n"
            "   1. DATABASE (Operational Reality) > All else.\n"
            "   2. OFFICIAL DOCS (Baseline) > Informal.\n"
            "   3. INFORMAL EMAILS (Exception) OVERRIDES Official Docs ONLY IF email date > Doc version date.\n\n"
            "***\n"
            "DATA SOURCES:\n"
            "=== DATABASE (Step 1) ===\n{db_data}\n\n"
            "=== OFFICIAL DOCS (Step 2) ===\n{official_data}\n\n"
            "=== INFORMAL EMAILS (Step 3) ===\n{informal_data}\n\n"
            "USER QUESTION: {question}\n\n"
            "***\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "You must output ONLY the following structured report. Do NOT include conversational filler like 'Based on...'.\n\n"
            "**AUDIT TARGET**: [Entity Name] (e.g. RTX-9000)\n"
            "**STATUS**: [ALIGNED / DISCREPANCY DETECTED / INSUFFICIENT DATA]\n\n"
            "**1. CURRENT AUTHORITATIVE VALUE**:\n"
            "- [Value] (Source: [Highest Priority Source], Date/Version: [X])\n\n"
            "**2. CONFLICTING DATA**:\n"
            "- Found [Value] in [Source Name] (Reason for override: Outdated version / Lower authority).\n\n"
            "**3. AUDIT CONCLUSION**:\n"
            "- [One direct, conclusive sentence explaining what the actual truth is according to the system's hierarchy.]"
        ))
    ])

    chain = discrepancy_prompt | config.llm | StrOutputParser()
    discrepancy_result = llm_invoke_with_retry(chain, {
        "db_data": db_data,
        "official_data": formatted_official,
        "informal_data": formatted_informal,
        "question": question,
    })

    # ── Step 3: Validation & Output ─────────────────────────────────────
    
    # Entity validation safety net
    if (not is_general 
            and entity.lower() not in discrepancy_result.lower() 
            and "NO_DISCREPANCY_FOUND" not in discrepancy_result):
        # Only override if the result seems to be about something completely different
        # For now, we trust the "Referee" prompt but log a warning
        print(f"[Case Agent] ⚠️  Warning: Report might not mention {entity} explicitly.")

    if "STATUS: ALIGNED" in discrepancy_result or "NO_DISCREPANCY_FOUND" in discrepancy_result:
        print(f"[Case Agent] ✅ No discrepancies found (Status: ALIGNED).")
        return {
            "generation": generation,
            "discrepancy_report": "", 
            "_thinking": f"Referee Check: {entity} -> ALIGNED."
        }
    else:
        print(f"[Case Agent] ⚠️  DISCREPANCY FOUND.")
        return {
            "generation": generation,
            "discrepancy_report": discrepancy_result,
            "_thinking": f"Referee Check: Discrepancy detected for {entity}."
        }

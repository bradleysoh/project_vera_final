"""
================================================================================
Response Generator Agent — "Report Compiler" (Structured Fact Passing)
================================================================================
RESPONSIBILITY: Format and synthesize structured facts + discrepancy verdict
                from domain agents into a final report.

ARCHITECTURAL CONSTRAINT:
    This agent is a REPORT COMPILER, NOT a reasoning engine.  It must:
    1. Base its ENTIRE output on the structured facts and verdict from state.
    2. NEVER infer, guess, or add information beyond the provided context.
    3. Output a structured "DATA NOT FOUND" report when no facts are available
       or retrieval confidence is LOW.
    4. Handle refinement loops when the discrepancy verdict flags conflicts.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.schemas import ExtractedFact, DiscrepancyVerdict, ConflictStatus
import shared.config as config
from shared.config import llm_invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Structured "Data Not Found" report template
# ---------------------------------------------------------------------------
_DATA_NOT_FOUND_TEMPLATE = (
    "📋 VERA REPORT — DATA NOT FOUND\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Query: {question}\n\n"
    "Status: INSUFFICIENT DATA\n\n"
    "The domain-specific agents returned {reason}. "
    "VERA cannot compile a reliable report from the available information.\n\n"
    "Recommended Actions:\n"
    "  • Verify the query targets a supported domain and entity\n"
    "  • Check that relevant documents have been ingested\n"
    "  • Try rephrasing the query with more specific identifiers\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

# ---------------------------------------------------------------------------
# Meta / capability query detection & canned response
# ---------------------------------------------------------------------------
_META_PATTERNS = [
    "what can you do", "what can u do", "what do you do",
    "who are you", "what are you", "what is vera",
    "help me", "how do you work", "what are your capabilities",
    "what can vera do", "introduce yourself", "tell me about yourself",
    "what is this", "how can you help", "what services",
]

_VERA_CAPABILITIES = (
    "🤖 **VERA — Verified Evidence & Retrieval Assistant**\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "I am an AI auditing assistant that cross-references multiple data sources "
    "to give you **verified, evidence-based answers**. Here's what I can do:\n\n"
    "📊 **Database Queries**\n"
    "  • Look up records from domain databases (e.g. wafer data, lot info, product specs)\n"
    "  • Example: *\"Tell me about WAF_003_A\"*\n\n"
    "📄 **Document Retrieval**\n"
    "  • Search official documents (datasheets, SOPs, specifications)\n"
    "  • Search informal sources (emails, memos, internal decisions)\n"
    "  • Example: *\"What is the max voltage for RTX-9000?\"*\n\n"
    "🔍 **Discrepancy Detection**\n"
    "  • Cross-reference data across DB, official docs, and informal sources\n"
    "  • Apply authority hierarchy: DB > Official Docs > Informal Sources\n"
    "  • Flag conflicts and identify authoritative values\n\n"
    "🔒 **Role-Based Access Control**\n"
    "  • Restrict confidential information based on your role level\n\n"
    "💡 **Tips for best results:**\n"
    "  • Use specific entity IDs (e.g. WAF_003_A, RTX-9000, LOT_001_A)\n"
    "  • Select the correct domain in the sidebar\n"
    "  • Ask specific questions about attributes you want to verify\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)


def _format_facts_as_table(facts: list[dict]) -> str:
    """
    Format structured facts into a compact text table for the LLM context.
    Much smaller token footprint than raw document chunks.
    """
    if not facts:
        return "(no facts)"

    lines = ["| Entity | Attribute | Value | Source | Date | Confidence |",
             "|--------|-----------|-------|--------|------|------------|"]
    for fd in facts:
        try:
            f = ExtractedFact(**fd)
            lines.append(
                f"| {f.entity} | {f.attribute} | {f.value} | "
                f"{f.source_type} | {f.date} | {f.confidence} |"
            )
        except Exception:
            continue
    return "\n".join(lines)


@vera_agent("Response Agent")
def run(state: GraphState) -> dict:
    """
    REPORT COMPILER: Formats structured facts and verdict into a report.
    Does NOT reason beyond the provided context.
    """
    question = state["question"]
    documents = state.get("documents", [])
    critique = state.get("critique", "")
    retrieval_confidence = state.get("retrieval_confidence", "MEDIUM")

    # --- Meta-query shortcut: capability/help questions ---
    q_lower = question.lower().strip()
    if any(p in q_lower for p in _META_PATTERNS):
        print("[Response Agent] ℹ️ Meta-query detected — returning VERA capabilities")
        return {
            "generation": _VERA_CAPABILITIES,
            "documents": documents,
            "critique": "",
            "_thinking": "Meta-query detected — returned static VERA capabilities response.",
        }

    # --- Gather structured facts ---
    official_facts = state.get("official_facts") or []
    informal_facts = state.get("informal_facts") or []
    db_facts = state.get("db_facts") or []
    all_facts = official_facts + informal_facts + db_facts

    # Also check raw DB data as fallback
    db_data = state.get("db_data", "") or state.get("db_result", "")
    has_db = bool(db_data) and "NO_MATCHING_DATA" not in db_data
    has_facts = bool(all_facts)
    has_documents = bool(documents)

    # --- Check for empty / low-confidence context ---
    if not has_facts and not has_documents and not has_db:
        print("[Response Agent] ⚠️ No context available — returning DATA NOT FOUND report")
        report = _DATA_NOT_FOUND_TEMPLATE.format(
            question=question,
            reason="no documents, no structured facts, and no database results",
        )
        return {
            "generation": report,
            "documents": documents,
            "critique": critique,
            "_thinking": "No context from domain agents. Returned structured DATA NOT FOUND report.",
        }

    if retrieval_confidence == "LOW" and not has_db and not has_facts:
        print("[Response Agent] ⚠️ Low confidence retrieval — returning DATA NOT FOUND report")
        report = _DATA_NOT_FOUND_TEMPLATE.format(
            question=question,
            reason="low-confidence retrieval results (no entity or attribute match found)",
        )
        return {
            "generation": report,
            "documents": documents,
            "critique": critique,
            "_thinking": "Retrieval confidence is LOW with no DB backup. Returned DATA NOT FOUND report.",
        }

    # --- Build context (compact for fast mode) ---
    context_parts = []
    from shared.config import RETRIEVAL_MODE

    # DB data always goes first (highest authority)
    if has_db and db_data:
        context_parts.append(f"DATABASE RESULT:\n{db_data[:500]}")

    if has_facts and RETRIEVAL_MODE == "deep":
        # Deep mode: use structured fact tables
        context_parts.append("--- STRUCTURED FACTS ---")
        if official_facts:
            context_parts.append("OFFICIAL FACTS:")
            context_parts.append(_format_facts_as_table(official_facts[:8]))
        if informal_facts:
            context_parts.append("INFORMAL FACTS:")
            context_parts.append(_format_facts_as_table(informal_facts[:5]))
    elif has_documents:
        # Fast mode / fallback: use raw documents (compact)
        doc_texts = []
        for d in documents[:5]:  # Limit to 5 docs max
            src = d.metadata.get("source", "unknown").upper()
            doc_texts.append(f"[{src}] {d.page_content[:200]}")
        context_parts.append("RETRIEVED DOCUMENTS:\n" + "\n---\n".join(doc_texts))

    # Include discrepancy verdict if available
    verdict_dict = state.get("discrepancy_verdict")
    if verdict_dict:
        try:
            verdict = DiscrepancyVerdict(**verdict_dict)
            context_parts.append("\n--- DISCREPANCY AUDIT ---")
            context_parts.append(verdict.to_report_string())
        except Exception:
            pass

    context_text = "\n".join(context_parts)

    # --- Report Compiler prompt ---
    if critique:
        print(f"[Response Agent] 🔄 Refinement iteration. Critique: {critique[:100]}...")
        system_instruction = (
            "You are VERA Report Compiler. Recompile the report addressing conflicts.\n"
            "Use ONLY the facts in CONTEXT. Database overrides documents.\n"
            f"DISCREPANCY:\n{critique[:300]}\n"
        )
    elif RETRIEVAL_MODE == "fast":
        print("[Response Agent] Compiling report (fast mode)...")
        system_instruction = (
            "You are VERA Report Compiler. Answer the question using ONLY the CONTEXT below.\n"
            "Rules: Use only provided data. Database overrides documents. "
            "If data not found, say so. Be concise and direct."
        )
    else:
        print("[Response Agent] Compiling report from structured facts...")
        system_instruction = (
            "You are VERA Report Compiler. Your ONLY job is to format and "
            "synthesize the pre-filtered facts below into a structured report.\n\n"
            "RULES:\n"
            "1. Use ONLY the facts in CONTEXT. DO NOT guess.\n"
            "2. DIRECT ANSWER FIRST.\n"
            "3. DB AUTHORITY: Database overrides documents.\n"
            "4. ENTITY ISOLATION: Only report on the queried entity.\n"
            "5. CITE SOURCES. If not found, say '[NOT FOUND IN AVAILABLE DATA]'.\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", (
            "CONTEXT:\n{context}\n\n"
            "USER QUESTION: {question}"
        ))
    ])

    chain = prompt | config.llm | StrOutputParser()
    response = llm_invoke_with_retry(chain, {
        "context": context_text,
        "question": question,
    })

    return {
        "generation": response,
        "documents": documents,
        "critique": critique,
        "_thinking": (
            f"Report Compiler: compiled response from {len(all_facts)} structured facts "
            f"+ {len(documents)} docs (confidence={retrieval_confidence}). "
            f"{'Refinement mode.' if critique else 'Initial compilation.'}"
        ),
    }

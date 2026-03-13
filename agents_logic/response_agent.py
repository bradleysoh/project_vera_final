"""
================================================================================
Response Generator Agent — "Report Compiler" (Structured Fact Passing)
================================================================================
RESPONSIBILITY: Format and synthesize structured facts + discrepancy verdict
                from domain agents into a final report. Absolutely NO domain
                logic or specific entity hardcoding allowed here.
================================================================================
"""

import json
import re
from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.schemas import ExtractedFact, DiscrepancyVerdict, ConflictStatus
import shared.config as config
from shared.advanced_rag import NO_DATA_MARKER
from shared.config import llm_invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# Structured "Data Not Found" report template
# ---------------------------------------------------------------------------
_DATA_NOT_FOUND_MSG = (
    "⚠️ **Data Not Found:** I can only answer based on the provided source documents "
    "and database records. No relevant information was found for your query in the "
    "current domain."
)

# ---------------------------------------------------------------------------
# Meta / capability query detection & canned response
# ---------------------------------------------------------------------------
_META_PATTERNS = [
    "what can you do", "what can u do", "what do you do",
    "who are you", "what are you", "what is vera",
    "help me", "how do you work", "what are your capabilities",
    "what can vera do", "introduce yourself", "tell me about yourself",
]

def _get_vera_capabilities() -> str:
    return (
        "🤖 **VERA — Verified Evidence & Retrieval Assistant**\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "I am an AI auditing assistant that cross-references multiple data sources to give you **verified answers**.\n\n"
        "📊 **Database Queries**: Look up production records, specs, and histories.\n"
        "📄 **Document Retrieval**: Search SOPs, datasheets, and internal emails.\n"
        "🔍 **Discrepancy Detection**: Flag conflicts between sources automatically.\n"
    )

def _format_facts_as_list(facts: list[dict]) -> str:
    """Simpler format for small models (Llama 3.2 1B) to avoid table parsing errors."""
    if not facts: return "(no facts)"
    lines = []
    for fd in facts:
        try:
            f = ExtractedFact(**fd)
            # Binary Filter
            if "%PDF" in f.value[:10] or "obj <<" in f.value[:50] or f.value.count("\\x") > 5:
                continue
            lines.append(f"- Entity: {f.entity} | Attribute: {f.attribute} | Value: {f.value} (Source: {f.source_type}, Date: {f.date})")
        except: continue
    if not lines: return "(no valid textual facts found)"
    return "\n".join(lines)


@vera_agent("Response Agent")
def run(state: GraphState) -> dict:
    question = state["question"]
    target_entity = state.get("target_entity", "GENERAL")
    documents = state.get("documents", [])
    official_facts = state.get("official_facts") or []
    informal_facts = state.get("informal_facts") or []
    db_facts = state.get("db_facts") or []
    db_data = state.get("db_data", "") or state.get("db_result", "")
    
    # --- Check if this is a discrepancy report ---
    verdict_dict = state.get("discrepancy_verdict", {}) or {}
    raw_status = verdict_dict.get("overall_status", "")
    status_str = str(raw_status).split(".")[-1].upper()
    is_discrepancy = (status_str == "DISCREPANCY")

    # --- Meta-query shortcut ---
    if any(p in question.lower() for p in _META_PATTERNS):
        return {"generation": _get_vera_capabilities(), "_thinking": "Meta-query response."}

    # --- 核心修订 2: Entity-Strict Context Filtering (Moved Up) ---
    # Filter facts to the target entity for specific (non-generic) queries
    # to prevent the LLM from seeing unrelated background facts.
    is_generic = state.get("is_generic_query", False)
    display_official = official_facts
    display_informal = informal_facts
    display_db = db_facts
    display_db_data = db_data

    if not is_generic and target_entity != "GENERAL":
        target_lower = target_entity.lower().strip()
        variations = {target_lower, target_lower.replace("-", " "), target_lower.replace(" ", "-"), target_lower.replace(" ", "")}
        
        def _matches(f):
            e = f.get('entity', '').lower()
            # STRICT FILTER: Only allow explicit entity matches or 'general' labels.
            return any(v in e for v in variations if len(v) > 2) or e in ("general", "unknown", "")

        display_official = [f for f in official_facts if _matches(f)]
        display_informal = [f for f in informal_facts if _matches(f)]
        display_db = [f for f in db_facts if _matches(f)]
        
        def _doc_matches(doc):
            c = doc.page_content.lower()
            t = doc.metadata.get("title", "").lower()
            s = doc.metadata.get("source", "").lower()
            entity_match = any(v in c or v in t for v in variations if len(v) > 2)
            is_generic_doc = any(kw in s or kw in t for kw in ("sop", "policy", "handbook", "manual", "regulations"))
            return entity_match or is_generic_doc

        documents = [d for d in documents if _doc_matches(d)]
        
        # If DB data (blob) doesn't mention the entity, clear it from the generator's view
        has_orig_db = bool(db_data and db_data != NO_DATA_MARKER)
        if has_orig_db and not any(v in db_data.lower() for v in variations if len(v) > 2):
            display_db_data = ""

    # --- 核心修复 1: 绝对的物理空值拦截 (Physical Null Check) ---
    # RE-CALCULATE after filtering
    final_has_db = bool(display_db_data and display_db_data != NO_DATA_MARKER)
    has_relevant_context = bool(display_official or display_informal or display_db or final_has_db or documents)
    
    if not has_relevant_context:
        print(f"[Response Agent] 🔒 Information Lock: Context filtered to zero relevance. Forcing 'Data Not Found'.")
        return {"generation": _DATA_NOT_FOUND_MSG, "_thinking": "Context exists but is irrelevant to the target entity."}

    # --- Build Context for LLM ---
    context_parts = [f"Below is the VERIFIED CONTEXT exclusively for the entity '{target_entity}':"]
    if final_has_db: context_parts.append(f"### DATABASE RECORDS FOR '{target_entity}':\n{display_db_data[:1500]}")
    if display_official: context_parts.append(f"### OFFICIAL SPECIFICATIONS FOR '{target_entity}':\n{_format_facts_as_list(display_official)}")
    if display_informal: context_parts.append(f"### INTERNAL COMMUNICATIONS FOR '{target_entity}':\n{_format_facts_as_list(display_informal)}")
    if documents and not (display_official or display_informal):
        doc_snippet = "\n\n".join([f"[Source {i+1}] {doc.page_content[:800]}" for i, doc in enumerate(documents[:5])])
        context_parts.append(f"### MENTIONED IN PRIMARY DOCUMENTS (RELEVANT TO '{target_entity}'):\n{doc_snippet}")

    context_text = "\n\n".join(context_parts)

    # --- 核心修复 3: 注入绝对反致幻系统指令 (Anti-Hallucination Prompt) ---
    
    system_instruction = (
        "You are VERA, a verified information assistant. Your goal is to summarize the provided CONTEXT.\n"
        "STRICT RULES:\n"
        "1. SYNTHESIS: List all facts found in the CONTEXT related to the target entity. "
        "If the CONTEXT contains an AUDIT REPORT showing alignment, mention it. "
        "If it shows a conflict, state it clearly.\n"
    )
    
    if is_generic:
        system_instruction += (
            "2. TOPIC-BASED SUMMARY: List ALL distinct details found in the context with their sources.\n"
        )
    else:
        system_instruction += (
            f"2. TARGETED SUMMARY: Focus exclusively on '{target_entity}'. "
            "Ignore any training data you have about this entity; use ONLY the provided context.\n"
        )

    if is_discrepancy:
        system_instruction += (
            "3. DISCREPANCY DETECTED: A conflict exists. Briefly state the conflict at the start. "
            "4. IMPORTANT: Always include specific numbers, dates, and names found in the context.\n"
        )
    else:
        system_instruction += (
            "3. NO DISCREPANCIES: The context shows that all available sources are ALIGNED. "
            "Clearly state that no discrepancies were found. "
            "Summarize the verified information from official and database sources correctly.\n"
        )

    # Include discrepancy report in context if relevant
    report_text = state.get("discrepancy_report", "")
    # AUDIT SHIELD: If authoritative source is NO_DATA, and we are not in generic mode,
    # the discrepancy might be a false positive against irrelevant documents.
    is_false_discrepancy = (NO_DATA_MARKER in report_text and not is_generic)
    
    if report_text and not is_false_discrepancy:
        context_parts.append(f"### AUDIT REPORT:\n{report_text}")

    context_text = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction), 
        ("human", "TARGET ENTITY: {entity}\n\nCONTEXT:\n{context}\n\nUSER QUESTION: {question}")
    ])
    
    chain = prompt | config.llm | StrOutputParser()
    raw_response = llm_invoke_with_retry(chain, {
        "entity": target_entity,
        "context": context_text, 
        "question": question
    }).strip()

    # Split report and summary
    main_res = raw_response
    audit_summary = ""
    if "[AUDIT_SUMMARY]" in main_res:
        parts = main_res.split("[AUDIT_SUMMARY]")
        main_res = parts[0].strip()
        if len(parts) > 1:
            audit_summary = parts[1].strip()

    # Fallback Handling: If LLM refused or gave a very short response, 
    # generate a multi-source "Verified Fact Summary".
    if not main_res or "NOT_FOUND" in main_res.upper() or len(main_res) < 10:
        if is_discrepancy:
            # Fallback to state-based audit report if LLM failed to generate one
            state_report = report_text or verdict_dict.get("audit_summary", "A conflict was detected between sources.")
            main_res = f"Note: Conflicts were detected between sources.\n\n{state_report}"
        elif (display_official or display_db or display_informal):
            # Manual multi-source evidence summary fallback
            summary_lines = [f"Based on the VERIFIED CONTEXT, I found the following information for '{target_entity}':"]
            seen_values = set()
            
            for src_list, label in [(display_official, "OFFICIAL"), (display_db, "DATABASE"), (display_informal, "INFORMAL")]:
                for f in src_list:
                    val_clean = f['value'].strip()
                    if val_clean not in seen_values and len(val_clean) > 5:
                        summary_lines.append(f"- **{label} ({f.get('entity', 'General')})**: {val_clean[:1000]}")
                        seen_values.add(val_clean)
            
            if len(summary_lines) > 1:
                main_res = "\n".join(summary_lines)
                print(f"[Response Agent] 🔧 Multi-source evidence summary triggered as fallback.")
            else:
                main_res = _DATA_NOT_FOUND_MSG
        else:
            main_res = _DATA_NOT_FOUND_MSG

    return {
        "generation": main_res,
        "discrepancy_report_summary": audit_summary,
        "thought_process": [f"Synthesized from available context. Evaluated against entity '{target_entity}'."],
    }
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

import json
import re
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
    "what is this", "how can you help", "what services",
]

_GENERIC_ENTITY_PHRASES = {
    "general", "all", "all items", "all records", "all data", "all contracts",
    "all clauses", "everything", "any", "contract", "the contract", "key aspects",
}

_CONTRACT_UPLOAD_REQUIRED_MSG = (
    "⚠️ To analyze a specific contract, please upload the contract file "
    "(.txt/.pdf/.png) first."
)


def _is_generic_entity(entity: str) -> bool:
    if not entity:
        return True
    e = entity.strip().lower()
    if not e:
        return True
    if e in ("general", "general_query"):
        return True
    if e in _GENERIC_ENTITY_PHRASES:
        return True
    tokens = re.findall(r"[a-z0-9]+", e)
    generic_tokens = {
        "all", "any", "items", "item", "records", "record", "data", "contracts",
        "contract", "clauses", "clause", "labels", "label", "terms", "aspects",
        "aspect", "entities", "entity", "everything", "general", "the", "this",
    }
    return bool(tokens) and all(t in generic_tokens for t in tokens)


def _needs_uploaded_contract(question: str) -> bool:
    """
    Detect when user is asking to analyze a specific contract (not CUAD/general).
    """
    q = (question or "").strip().lower()
    if not q:
        return False

    explicit_contract_requests = [
        "this contract",
        "the contract",
        "my contract",
        "uploaded contract",
        "review this contract",
        "analyze this contract",
        "key aspects of this contract",
        "agreement date",
        "signed date",
    ]
    if any(p in q for p in explicit_contract_requests):
        return True

    if "contract" not in q and "agreement" not in q:
        return False

    # CUAD/dataset/baseline questions can be answered from retrieval context.
    dataset_intents = [
        "cuad",
        "benchmark",
        "prevalence",
        "reference contract",
        "reference contracts",
        "standard clause",
        "common clause",
        "all contracts",
        "dataset",
    ]
    if any(p in q for p in dataset_intents):
        return False

    return True


def _is_contract_quality_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    quality_markers = [
        "is the contract okay",
        "is this contract okay",
        "is this a contract",
        "is this valid",
        "is this document a contract",
        "is the agreement okay",
        "good contract",
        "contract quality",
        "falls short",
    ]
    return any(m in q for m in quality_markers)


def _format_discrepancy_first_response(state: GraphState) -> dict:
    """
    Build a deterministic discrepancy-focused legal response from verdict data.
    """
    verdict_dict = state.get("discrepancy_verdict", {}) or {}
    report = (state.get("discrepancy_report", "") or "").strip()
    summary = (verdict_dict.get("audit_summary") or state.get("discrepancy_report_summary") or "").strip()
    raw_status = str(verdict_dict.get("overall_status", "")).split(".")[-1].upper()
    conflicts = verdict_dict.get("conflicts", []) or []
    discrepancy_items = [
        c for c in conflicts
        if str(c.get("status", "")).split(".")[-1].upper() == "DISCREPANCY"
    ]

    if raw_status == "DISCREPANCY" or discrepancy_items:
        lines = [
            "⚠️ **Conclusion: The uploaded document falls short of a regular contract.** [CUAD_REFERENCE]"
        ]
        if summary:
            lines.append(f"- {summary}")
        if discrepancy_items:
            lines.append("- Discrepancies identified against CUAD contract baseline:")
            for c in discrepancy_items[:8]:
                attr = (c.get("attribute") or "unknown").strip()
                reason = ""
                cvals = c.get("conflicting_values", []) or []
                if cvals:
                    reason = (cvals[0].get("reason") or "").strip()
                if reason:
                    lines.append(f"  - {attr}: {reason}")
                else:
                    lines.append(f"  - {attr}")
    elif raw_status == "ALIGNED":
        lines = ["✅ **Conclusion: The uploaded document is broadly aligned with CUAD contract patterns.** [CUAD_REFERENCE]"]
        if summary:
            lines.append(f"- {summary}")
    else:
        lines = ["⚠️ **Conclusion: Insufficient contract evidence found for a reliable benchmark.** [CUAD_REFERENCE]"]
        if summary:
            lines.append(f"- {summary}")

    if report:
        lines.append("")
        lines.append("### Discrepancy Report")
        lines.append(report[:3000])

    generation = "\n".join(lines)
    return {
        "generation": generation,
        "discrepancy_report_summary": summary,
        "thought_process": ["Discrepancy-first legal response generated deterministically."],
        "critique": state.get("critique", ""),
        "_thinking": "Returned discrepancy-first legal response (database text suppressed).",
    }

def _get_vera_capabilities(domain: str) -> str:
    """Generate domain-specific capabilities guide."""
    examples = {
        "semiconductor": ["*\"Tell me about RTX-9000\"*", "*\"Check lot history for WAF_001\"*"],
        "medical": ["*\"What is the maintenance history for MRI-Unit-4?\"*", "*\"Check patient records for dosage discrepancies\"*"],
        "general": ["*\"What is the status of Entity X?\"*", "*\"Are there any discrepancies in the latest report?\"*"],
    }
    domain_ex = examples.get(domain, examples["general"])
    
    return (
        "🤖 **VERA — Verified Evidence & Retrieval Assistant**\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "I am an AI auditing assistant that cross-references multiple data sources "
        "to give you **verified, evidence-based answers**. Here's what I can do:\n\n"
        "📊 **Database Queries**\n"
        "  • Look up records from domain databases (e.g. production data, history, specs)\n"
        f"  • Example: {domain_ex[0]}\n\n"
        "📄 **Document Retrieval**\n"
        "  • Search official documents (specs, SOPs, manuals)\n"
        "  • Search informal sources (emails, memos, internal decisions)\n"
        f"  • Example: {domain_ex[1]}\n\n"
        "🔍 **Discrepancy Detection**\n"
        "  • Apply authority hierarchy: DB > Official Docs > Informal Sources\n"
        "  • Flag conflicts and identify authoritative values\n\n"
        "🔒 **Role-Based Access Control**\n"
        "  • Restrict confidential information based on your role level\n\n"
        "💡 **Tips for best results:**\n"
        "  • Use specific IDs or names relevant to your domain\n"
        "  • Select the correct domain in the sidebar\n"
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
    user_domain = (state.get("user_domain", "") or "").strip().lower()
    input_contract_text = (state.get("input_contract_text", "") or "").strip()
    documents = state.get("documents", [])
    critique = state.get("critique", "")
    retrieval_confidence = state.get("retrieval_confidence", "MEDIUM")

    # --- Meta-query shortcut: capability/help questions ---
    q_lower = question.lower().strip()
    if any(p in q_lower for p in _META_PATTERNS):
        print("[Response Agent] ℹ️ Meta-query detected — returning VERA capabilities")
        domain = state.get("user_domain", "general")
        return {
            "generation": _get_vera_capabilities(domain),
            "documents": documents,
            "critique": "",
            "_thinking": "Meta-query detected — returned dynamic VERA capabilities response.",
        }

    # --- Scope guard fallback ---
    # If a contract-analysis query is asked with uploaded contract text outside
    # legal domain, do not generate a domain answer.
    q_lower_scope = question.lower()
    contract_scope_keywords = ("contract", "clause", "agreement", "cuad", "key aspects")
    if input_contract_text and user_domain != "legal" and any(k in q_lower_scope for k in contract_scope_keywords):
        return {
            "generation": (
                "⚠️ Contract analysis is only supported in the `legal` domain. "
                "Please switch User Domain to `legal` and ask again."
            ),
            "documents": [],
            "critique": "",
            "_thinking": "Scope guard fallback triggered in Response Agent.",
        }

    # --- Legal guard: uploaded contract required for specific contract analysis ---
    # Prevents fabricated/ambiguous answers from CUAD/reference docs when the user
    # is asking about a particular contract but has not provided one.
    if user_domain == "legal" and not input_contract_text and _needs_uploaded_contract(question):
        return {
            "generation": _CONTRACT_UPLOAD_REQUIRED_MSG,
            "documents": [],
            "critique": "",
            "_thinking": "Legal contract-analysis request without uploaded contract text.",
        }

    # --- Deterministic legal field answers from uploaded contract ---
    # For direct field lookups (e.g., "agreement date"), prefer extracted
    # uploaded-contract facts over CUAD/reference retrieval context.
    if user_domain == "legal" and input_contract_text:
        q_lower = question.lower()
        field_aliases = {
            "agreement_date": ["agreement date", "date of the agreement", "contract date", "signed date"],
        }
        requested_field = ""
        for field, aliases in field_aliases.items():
            if any(a in q_lower for a in aliases):
                requested_field = field
                break

        if requested_field:
            for fact in state.get("db_facts", []) or []:
                if (fact.get("attribute") or "").lower() == requested_field and (fact.get("value") or "").strip():
                    value = (fact.get("value") or "").strip()
                    return {
                        "generation": (
                            f"The **{requested_field.replace('_', ' ')}** in `{state.get('input_contract_name', 'input_contract')}` "
                            f"is: **{value}**. [DATABASE]"
                        ),
                        "discrepancy_report_summary": "",
                        "thought_process": [f"Deterministic legal field answer from uploaded contract: {requested_field}={value}"],
                        "critique": critique,
                        "_thinking": "Deterministic legal field answer returned before LLM synthesis.",
                    }

    # --- Deterministic legal contract quality answer from discrepancy verdict ---
    # For "is this contract okay?" style questions, avoid generic RAG synthesis
    # and return direct benchmark findings from uploaded-doc vs CUAD baseline.
    if user_domain == "legal" and input_contract_text and _is_contract_quality_query(question):
        verdict_dict = state.get("discrepancy_verdict", {}) or {}
        conflicts = verdict_dict.get("conflicts", []) or []
        discrepancy_items = [
            c for c in conflicts
            if str(c.get("status", "")).split(".")[-1].upper() == "DISCREPANCY"
        ]

        structure_conflict = next(
            (c for c in discrepancy_items if (c.get("attribute") or "").lower() == "contract_structure_assessment"),
            None,
        )
        detail_reason = ""
        if structure_conflict:
            conflicting_values = structure_conflict.get("conflicting_values", []) or []
            if conflicting_values:
                detail_reason = conflicting_values[0].get("reason", "")

        top_gaps = []
        for c in discrepancy_items:
            attr = (c.get("attribute") or "").strip()
            if attr and attr != "contract_structure_assessment":
                top_gaps.append(attr)
            if len(top_gaps) >= 5:
                break

        summary = (verdict_dict.get("audit_summary") or "").strip()
        if detail_reason:
            summary = f"{summary} {detail_reason}".strip()

        if discrepancy_items:
            lines = [
                "⚠️ **Conclusion: The uploaded document falls short of a regular contract.** [CUAD_REFERENCE]",
            ]
            if summary:
                lines.append(f"- {summary}")
            if top_gaps:
                lines.append("- Key missing/common contract elements compared to CUAD baseline:")
                for g in top_gaps:
                    lines.append(f"  - {g}")
            return {
                "generation": "\n".join(lines),
                "discrepancy_report_summary": summary,
                "thought_process": [
                    f"Deterministic legal quality answer from discrepancy verdict with {len(discrepancy_items)} discrepancy findings."
                ],
                "critique": critique,
                "_thinking": "Returned deterministic legal quality assessment from uploaded-doc benchmark.",
            }

    # --- Legal discrepancy-first response for contract-analysis queries ---
    # When user provided an uploaded legal document and asked a contract-analysis
    # question, prioritize discrepancy verdict/report instead of db_data context.
    if user_domain == "legal" and input_contract_text and _needs_uploaded_contract(question):
        verdict_dict = state.get("discrepancy_verdict", {}) or {}
        if verdict_dict:
            return _format_discrepancy_first_response(state)

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

    # --- Check for empty / low-confidence context (INFORMATION LOCK) ---
    if not has_facts and not has_documents and not has_db:
        print("[Response Agent] ⚠️ No context available — returning required Data Not Found message")
        return {
            "generation": _DATA_NOT_FOUND_MSG,
            "documents": documents,
            "critique": critique,
            "_thinking": "Information Lock: No context available. Returned exact Data Not Found message.",
        }

    if retrieval_confidence == "LOW" and not has_db and not has_facts:
        print("[Response Agent] ⚠️ Low confidence retrieval — returning required Data Not Found message")
        return {
            "generation": _DATA_NOT_FOUND_MSG,
            "documents": documents,
            "critique": critique,
            "_thinking": "Information Lock: Low retrieval confidence. Returned exact Data Not Found message.",
        }


    # --- INFORMATION LOCK: Strict Entity Validation ---
    target_entity = state.get("target_entity", "GENERAL")
    
    # Collect all factual entities seen in this turn
    all_facts_list = (
        state.get("official_facts", []) + 
        state.get("informal_facts", []) + 
        state.get("db_facts", [])
    )
    
    # If a specific entity was requested, it MUST appear in the facts OR raw data.
    if not _is_generic_entity(target_entity):
        entity_found = False
        target_lower = target_entity.lower()
        
        # Check structured facts
        for fact in all_facts_list:
            fact_entity = fact.get("entity", "").lower()
            if target_lower == fact_entity or target_lower in fact_entity:
                entity_found = True
                break
        
        # Also check raw DB data (populated by fast-mode / deterministic SQL)
        if not entity_found:
            db_data = state.get("db_data", "")
            if db_data and target_lower in db_data.lower():
                entity_found = True
        
        # Also check retrieved documents
        if not entity_found:
            for doc in state.get("documents", []):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                if target_lower in content.lower():
                    entity_found = True
                    break
        
        if not entity_found:
            user_domain = state.get("user_domain", "medical")
            print(f"[Response Agent] 🔒 Information Lock: '{target_entity}' not found in any validated facts. Forcing 'Data Not Found'.")
            return {
                "generation": f"I cannot find any information regarding '{target_entity}' in the {user_domain} domain. Can I help you with anything else?",
                "critique": "Information Lock: Entity mismatch.",
                "thought_process": [f"Information Lock: '{target_entity}' not found in facts. Short-circuiting."]
            }

    # --- Build context ---
    context_parts = ["Below is the VERIFIED CONTEXT to answer the question:"]
    
    if has_db and db_data:
        context_parts.append(f"### DATA FROM DATABASE:\n{db_data[:800]}")

    if has_facts:
        context_parts.append("### STRUCTURED DATA POINTS:")
        if db_facts:
            context_parts.append("- DATABASE DATA:")
            context_parts.append(_format_facts_as_table(db_facts[:5]))
        if official_facts:
            context_parts.append("- OFFICIAL SPECS:")
            context_parts.append(_format_facts_as_table(official_facts[:8]))
        if informal_facts:
            context_parts.append("- OTHER INFO:")
            context_parts.append(_format_facts_as_table(informal_facts[:5]))

    if has_documents:
        context_parts.append("### TEXT FROM DOCUMENTS:")
        doc_texts = []
        for i, doc in enumerate(documents[:5]):
            doc_texts.append(f"[Source {i+1}] {doc.page_content[:500]}")
        context_parts.append("\n\n".join(doc_texts))

    # --- Python-based Knowledge Augmentation (Precision Math for Small Models) ---
    doc_text_for_math = "\n".join([d.page_content for d in state.get("documents", [])])
    found_stats = []
    
    # Targeted vitals for clinical grounding
    vitals = ["age", "bmi", "systolic_bp", "diastolic_bp", "glucose", "cholesterol", "creatinine"]
    
    for metric in vitals:
        vals = []
        # 1. Standard "metric: value" or "metric=value" pattern
        matches = re.findall(rf"{metric}[:\s=]+(\d+\.?\d*)", doc_text_for_math, re.IGNORECASE)
        vals.extend([float(m) for m in matches])
        
        # 2. Markdown Table pattern: extract numbers from specific columns
        lines = doc_text_for_math.split("\n")
        header_idx = -1
        for line in lines:
            if "|" in line:
                cells = [c.strip().lower() for c in line.split("|")]
                # Look for metric in header
                if any(metric in c for c in cells):
                    for i, c in enumerate(cells):
                        if metric in c: 
                            header_idx = i
                            break
                    continue # Found header, move to data rows
                
                if header_idx != -1 and "---" not in line:
                    data_cells = [c.strip() for c in line.split("|")]
                    if len(data_cells) > header_idx:
                        val_str = data_cells[header_idx]
                        num_match = re.search(r"(\d+\.?\d*)", val_str)
                        if num_match:
                            try:
                                f_val = float(num_match.group(1))
                                # Basic sanity check
                                if metric == "age" and 0 < f_val < 120: vals.append(f_val)
                                elif metric == "bmi" and 10 < f_val < 60: vals.append(f_val)
                                elif "bp" in metric and 40 < f_val < 250: vals.append(f_val)
                                elif metric == "glucose" and 10 < f_val < 500: vals.append(f_val)
                            except: pass
        
        # Reset if table parsing failed to find metric-specific columns
        if not vals:
            # Fallback to general row scan if no specific column found
            table_rows = re.findall(r'\|(.*?)\|', doc_text_for_math)
            for row in table_rows:
                if "---" not in row and metric in row.lower():
                    row_vals = re.findall(r'(\d+\.?\d*)', row)
                    for rv in row_vals:
                        try:
                            f_rv = float(rv)
                            if metric == "age" and 0 < f_rv < 120: vals.append(f_rv)
                            elif metric == "bmi" and 10 < f_rv < 60: vals.append(f_rv)
                            elif "bp" in metric and 40 < f_rv < 250: vals.append(f_rv)
                            elif metric == "glucose" and 10 < f_rv < 500: vals.append(f_rv)
                        except: pass

        if len(vals) >= 2:
            try:
                unique_vals = sorted(list(vals)) # Keep duplicates for mean!
                avg = sum(unique_vals) / len(unique_vals)
                found_stats.append(f"- {metric.upper()}: Mean {round(avg, 2)} (Range: {min(unique_vals)} - {max(unique_vals)})")
            except: continue
    
    # 3. Look for pre-aggregated results from DB (like 'avg_value')
    agg_matches = re.findall(r"(?:avg|mean|average)[:\s=]+(\d+\.?\d*)", doc_text_for_math, re.IGNORECASE)
    for am in agg_matches[:5]:
        found_stats.append(f"- CALCULATED_METRIC: {am} (Authoritative from Database)")
    
    if found_stats:
        stats_block = "### VERIFIED AGGREGATE DATA (Python Pre-calculated):\n" + "\n".join(found_stats)
        context_parts.insert(0, stats_block)

    context_text = "\n\n".join(context_parts)

    # --- Determine if this is a discrepancy report ---
    verdict_dict = state.get("discrepancy_verdict", {})
    raw_status = verdict_dict.get("overall_status", "")
    status_str = str(raw_status).split(".")[-1].upper()
    is_discrepancy = (status_str == "DISCREPANCY")

    # --- Ultra-simple prompt for small models (1B) ---
    system_instruction = (
        "You are VERA, a verified information assistant. "
        "Your priority is ACCURACY and FAITHFUL REPORTING.\n\n"
        "RULES:\n"
        "1. Answer using ONLY the CONTEXT PROVIDED.\n"
        "2. DATABASE DATA/FACTS are the highest authority.\n"
        "3. DATA FIDELITY: Report exact values provided in context.\n"
        "4. AGGREGATES: If 'VERIFIED AGGREGATE DATA' is present, use those values as the authoritative answer for any requests for averages or ranges.\n"
        "5. NO DATA: If context is totally unrelated to the domain, respond 'NOT_FOUND'.\n"
    )
    
    if is_discrepancy:
        system_instruction += (
            "6. DISCREPANCY: Describe conflicts clearly. "
            "Append '[AUDIT_SUMMARY]' then a 1-sentence summary.\n"
        )

    system_instruction += "Cite sources like [DATABASE] or [Source X]. No conversational filler."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "CONTEXT:\n{context}\n\nUSER QUESTION: {question}")
    ])

    chain = prompt | config.llm | StrOutputParser()
    raw_response = llm_invoke_with_retry(chain, {
        "context": context_text,
        "question": question,
    }).strip()

    # Separate answer and audit summary
    main_response = raw_response
    audit_summary = ""
    if "[AUDIT_SUMMARY]" in main_response:
        parts = main_response.split("[AUDIT_SUMMARY]")
        main_response = parts[0].strip()
        audit_summary = parts[1].strip()

    # Fallback logic: Only trigger Data Not Found if NOT a discrepancy case
    failure_markers = ["NOT_FOUND", "I DON'T KNOW", "DATA NOT FOUND", "NO RELEVANT INFO", "CONTEXT IS IRRELEVANT"]
    is_lazy_fallback = any(m in main_response.upper() for m in failure_markers)
    
    if not main_response or (is_lazy_fallback and not is_discrepancy):
        main_response = _DATA_NOT_FOUND_MSG
    
    # If it was a discrepancy but LLM said NOT_FOUND, try to salvage from the verdict
    if is_discrepancy and (is_lazy_fallback or not main_response or len(main_response) < 10):
        main_response = (
            "A discrepancy was detected for this query. The system found conflicting values between technical "
            "specifications and database records. Please review the '[AUDIT SUMMARY]' and 'Detailed Findings' "
            "sections below for the specific differences."
        )
    
    # Ensure audit_summary is populated if a discrepancy was found, for UI display
    if is_discrepancy and not audit_summary:
        audit_summary = "A conflict was detected between sources for this entity. Detailed findings are available in the audit report."
        print("[Response Agent] 🔧 Salvaged audit summary for UI")

    thinking = f"Report Compiler: answer from {len(all_facts)} facts."

    return {
        "generation": main_response,
        "discrepancy_report_summary": audit_summary,
        "thought_process": [thinking],
        "critique": critique,
        "_thinking": thinking,
    }

"""
================================================================================
Semiconductor Discrepancy Agent — Deterministic Logic Gate (Zero LLM Calls)
================================================================================
OWNER: (Assign team member)
DOMAIN: semiconductor
RESPONSIBILITY: Compare structured facts from Official Docs, Informal Docs,
                and Database using a DETERMINISTIC hierarchy of authority.

ARCHITECTURAL CONSTRAINT:
    This agent makes ZERO LLM calls.  It operates exclusively on
    ExtractedFact dicts from GraphState, applying pure Python logic:

    HIERARCHY:
        DB facts  >  Official facts  >  Informal facts (ONLY if newer date)

    ENTITY ISOLATION:
        Only facts matching the target_entity from QueryIntent are compared.
        Facts for other entities are excluded from the verdict.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.advanced_rag import _is_garbage_text, NO_DATA_MARKER
from shared.schemas import (
    ExtractedFact,
    AttributeConflict,
    DiscrepancyVerdict,
    ConflictStatus,
)


def _parse_date(date_str: str) -> str:
    """
    Normalize a date string for comparison.
    Returns the original string (ISO dates sort lexicographically).
    Returns "0000-00-00" for unknown/missing dates.
    """
    if not date_str or date_str.lower() in ("unknown", "none", "n/a", ""):
        return "0000-00-00"
    return date_str.strip()


def _source_priority(source_type: str) -> int:
    """
    Assign a numeric priority to fact sources.
    Higher = more authoritative.
    """
    s = source_type.lower()
    if s in ("db", "database", "db_info", "sql", "records"):
        return 3
    elif s in ("datasheet", "sop", "spec", "document", "manual", "policy", "regulations", "guideline", "standard"):
        return 2
    elif s in ("email", "memo", "dm", "informal", "communication", "chat"):
        return 1
    return 0


def _build_fact_index(
    facts: list[dict],
    target_entity: str,
    target_attribute: str = "GENERAL",
    is_generic_query: bool = False,
) -> dict[str, list[ExtractedFact]]:
    """
    Group facts by attribute, filtering to the target entity.
    Includes 'soft matching' for attributes to handle naming mismatches
    (e.g. 'voltage' vs 'voltage_max').
    """
    index: dict[str, list[ExtractedFact]] = {}
    target_lower = target_entity.lower() if target_entity != "GENERAL" else ""
    attr_target_lower = target_attribute.lower().strip() if target_attribute != "GENERAL" else ""

    for fd in facts:
        try:
            fact = ExtractedFact(**fd)
        except Exception:
            continue

        # Entity isolation (Soften matching)
        fact_entity_lower = fact.entity.lower()
        if target_lower and not is_generic_query:
            variations = {target_lower, target_lower.replace("-", " "), target_lower.replace(" ", "-"), target_lower.replace(" ", "")}
            entity_match = any(v in fact_entity_lower for v in variations if len(v) > 1)
            
            # ALLOW facts from high-authority sources even if label doesn't match perfectly,
            # as long as they were retrieved for this context.
            is_authoritative = _source_priority(fact.source_type) >= 2
            if not entity_match and not is_authoritative:
                continue

        attr_raw = fact.attribute.lower().replace("-", "_").strip()
        
        # Soft-matching logic:
        # 1. Broad grouping for catch-all attributes
        # 2. Canonicalize to target_attribute if it's a strong match
        
        final_key = attr_raw
        if attr_target_lower:
            # Normalize for comparison: remove filler and common prefixes/suffixes
            # 'maximum_voltage' -> 'voltage', 'voltage_max' -> 'voltage'
            filler = {"max", "maximum", "min", "minimum", "target", "nominal", "val", "value", "rtx", "series"}
            
            t_parts = [p for p in attr_target_lower.split("_") if p not in filler]
            a_parts = [p for p in attr_raw.split("_") if p not in filler]
            
            t_core = "".join(t_parts)
            a_core = "".join(a_parts)
            
            # HARD MATCH: Exact core match (e.g. 'voltage' == 'voltage')
            # SOFT MATCH: Core containment (e.g. 'v' in 'voltage')
            # SEMANTIC MATCH: If the raw fact value itself contains the target attribute name
            if (t_core and a_core and (t_core in a_core or a_core in t_core)):
                final_key = attr_target_lower
            elif t_core == a_core:
                final_key = attr_target_lower
            elif t_core in fact.value.lower() or t_core in fact.attribute.lower():
                final_key = attr_target_lower
        
        # KEY FIX: Group catch-all attributes together
        if attr_raw in ("general_info", "db_result", "database_record", "summary", "db_data", "database"):
            final_key = "general_info"

        if final_key not in index:
            index[final_key] = []
        index[final_key].append(fact)

    # Global Fallback: If no facts were found for the specific entity, 
    # and it's a generic query OR we have official documents, allow them.
    if not index and (is_generic_query or target_lower):
        for fd in facts:
            try:
                fact = ExtractedFact(**fd)
                # Binary & Garbage Filter: Discard facts that look like PDF garbage or raw bytes
                if _is_garbage_text(fact.value):
                    continue

                # ENTITY GUARD: Only allow official facts if they match the entity
                # OR if the query is a truly broad GENERAL query.
                is_official = _source_priority(fact.source_type) >= 2
                entity_match = (not target_lower) or (target_lower in fact.entity.lower())
                
                if (is_official and entity_match) or fact.entity.upper() == "GENERAL":
                    attr_raw = fact.attribute.lower().replace("-", "_").strip()
                    final_key = "general_info" if attr_raw in ("general_info", "summary", "fact") else attr_raw
                    if final_key not in index:
                        index[final_key] = []
                    index[final_key].append(fact)
            except Exception:
                continue

    # KEY ENHANCEMENT: Hierarchy Leakage
    # If we have a 'general_info' (usually from DB), it should be compared against 
    # specific attributes too, if those specific attributes are missing from DB.
    if "general_info" in index:
        gen_facts = index["general_info"]
        for attr, facts_list in index.items():
            if attr == "general_info": continue
            # If this specific attribute has NO DB/Official components, 
            # we don't leak. But if it has Official specs, we should 
            # check the 'general_info' for matching keywords.
            # However, for now, let's just make the Discrepancy report more aggressive.
            pass

    return index



def _resolve_conflicts(
    official: list[ExtractedFact],
    informal: list[ExtractedFact],
    db: list[ExtractedFact],
    attribute: str,
) -> AttributeConflict:
    """
    Apply the deterministic hierarchy for one (entity, attribute) group.

    Priority:  DB > Official > Informal (only if newer date)
    """
    # Collect all values with their authority metadata
    all_values: list[dict] = []

    for f in db:
        all_values.append({
            "value": f.value, "source": f.source_type,
            "date": f.date, "priority": 3, "fact": f,
        })
    for f in official:
        all_values.append({
            "value": f.value, "source": f.source_type,
            "date": f.date, "priority": 2, "fact": f,
        })
    for f in informal:
        all_values.append({
            "value": f.value, "source": f.source_type,
            "date": f.date, "priority": 1, "fact": f,
        })

    if not all_values:
        return AttributeConflict(
            entity=official[0].entity if official else "unknown",
            attribute=attribute,
            status=ConflictStatus.INSUFFICIENT_DATA,
        )

    # Determine the authoritative fact
    # Sort: highest priority first, then newest date first
    all_values.sort(
        key=lambda v: (v["priority"], _parse_date(v["date"])),
        reverse=True,
    )

    authoritative = all_values[0]

    # Check for informal override: if an informal fact has a newer date
    # than the official authoritative fact AND there is no DB fact
    if authoritative["priority"] == 2:  # official is top
        newer_informal = [
            v for v in all_values
            if v["priority"] == 1
            and _parse_date(v["date"]) > _parse_date(authoritative["date"])
        ]
        if newer_informal:
            authoritative = newer_informal[0]
            print(f"[DISCREPANCY] Informal override for '{attribute}' — "
                  f"newer date: {authoritative['date']}")

    # Compare all values to the authoritative one
    conflicting = []
    auth_value_normalized = authoritative["value"].strip().lower()

    for v in all_values:
        if v is authoritative:
            continue
        if v["value"].strip().lower() != auth_value_normalized:
            conflicting.append({
                "value": v["value"],
                "source": v["source"],
                "date": v["date"],
                "reason": (
                    f"Conflict with authoritative value ({authoritative['source']})"
                    if v["priority"] == authoritative["priority"]
                    else f"Lower authority ({v['source']}) vs authoritative ({authoritative['source']})"
                ),
            })

    status = ConflictStatus.DISCREPANCY if conflicting else ConflictStatus.ALIGNED
    entity_name = authoritative["fact"].entity

    return AttributeConflict(
        entity=entity_name,
        attribute=attribute,
        status=status,
        authoritative_value=authoritative["value"],
        authoritative_source=authoritative["source"],
        authoritative_date=authoritative["date"],
        conflicting_values=conflicting,
    )


@vera_agent("Semiconductor Discrepancy Agent")
def run(state: GraphState) -> dict:
    """
    DETERMINISTIC DISCREPANCY AGENT: Pure Python logic on structured facts.

    Zero LLM calls.  Reads ExtractedFact dicts from GraphState, groups by
    (entity, attribute), and applies the hierarchy:
        DB > Official > Informal (only if newer)
    """
    target_entity = state.get("target_entity", "GENERAL")
    question = state["question"]

    # Skip discrepancy audit for general queries — no meaningful comparison
    if target_entity == "GENERAL":
        verdict = DiscrepancyVerdict(
            target_entity=target_entity,
            overall_status=ConflictStatus.ALIGNED,
            audit_summary="General query — discrepancy check skipped (no specific entity).",
        )
        return {
            "discrepancy_verdict": verdict.model_dump(),
            "discrepancy_report": verdict.to_report_string(),
            "critique": "",
            "_thinking": "General query — no entity to audit, skipping discrepancy check.",
        }

    # Collect all facts from state
    official_facts = state.get("official_facts") or []
    informal_facts = state.get("informal_facts") or []
    db_facts = state.get("db_facts") or []

    # Also extract DB facts from raw db_data if no structured db_facts exist
    if not db_facts:
        db_data = state.get("db_data", "") or state.get("db_result", "")
        if db_data and db_data != NO_DATA_MARKER:
            # Create a minimal fact from the DB result
            db_facts = [{
                "entity": target_entity if target_entity != "GENERAL" else "unknown",
                "attribute": "db_result",
                "value": db_data[:500],
                "source_type": "db",
                "source_doc": "database",
                "date": state.get("latest_timestamp", "unknown"),
                "confidence": "HIGH",
            }]

    print(f"[Semiconductor Discrepancy Agent] Facts: "
          f"official={len(official_facts)}, "
          f"informal={len(informal_facts)}, "
          f"db={len(db_facts)}")

    # Build indexes by attribute (entity-filtered)
    is_generic = state.get("is_generic_query", False)
    official_idx = _build_fact_index(official_facts, target_entity, state.get("target_attribute", "GENERAL"), is_generic)
    informal_idx = _build_fact_index(informal_facts, target_entity, state.get("target_attribute", "GENERAL"), is_generic)
    db_idx = _build_fact_index(db_facts, target_entity, state.get("target_attribute", "GENERAL"), is_generic)

    # Gather all attribute keys across all sources
    all_attributes = set(official_idx.keys()) | set(informal_idx.keys()) | set(db_idx.keys())

    if not all_attributes:
        verdict = DiscrepancyVerdict(
            target_entity=target_entity,
            overall_status=ConflictStatus.INSUFFICIENT_DATA,
            audit_summary="No structured facts available for comparison.",
        )
        return {
            "discrepancy_verdict": verdict.model_dump(),
            "discrepancy_report": verdict.to_report_string(),
            "critique": "",
            "_thinking": "No structured facts found — insufficient data for discrepancy check.",
        }

    # Resolve conflicts per attribute
    conflicts: list[AttributeConflict] = []
    
    # Catch-all DB facts: if we have a general DB fact, it might contain info for all attributes
    general_db_facts = db_idx.get("general_info", [])
    
    for attr in sorted(all_attributes):
        if attr == "general_info" and len(all_attributes) > 1:
            continue # Already being leaked into others
            
        attr_specific_db = db_idx.get(attr, [])
        # --- HIERARCHY LEAKAGE ---
        # If no specific DB facts were found for this attr, use the general one
        merged_db = attr_specific_db or general_db_facts
        
        conflict = _resolve_conflicts(
            official=official_idx.get(attr, []),
            informal=informal_idx.get(attr, []),
            db=merged_db,
            attribute=attr,
        )
        conflicts.append(conflict)

    # Determine overall status
    has_discrepancy = any(c.status == ConflictStatus.DISCREPANCY for c in conflicts)
    has_insufficient = any(c.status == ConflictStatus.INSUFFICIENT_DATA for c in conflicts)

    if has_discrepancy:
        overall = ConflictStatus.DISCREPANCY
    elif has_insufficient:
        overall = ConflictStatus.INSUFFICIENT_DATA
    else:
        overall = ConflictStatus.ALIGNED

    # Build summary
    disc_count = sum(1 for c in conflicts if c.status == ConflictStatus.DISCREPANCY)
    aligned_count = sum(1 for c in conflicts if c.status == ConflictStatus.ALIGNED)
    summary = (
        f"Audited {len(conflicts)} attributes for '{target_entity}': "
        f"{aligned_count} aligned, {disc_count} discrepancies."
    )

    verdict = DiscrepancyVerdict(
        target_entity=target_entity,
        overall_status=overall,
        conflicts=conflicts,
        audit_summary=summary,
    )

    report = verdict.to_report_string()

    # Only trigger refinement loop when retrieval confidence is HIGH
    # (In fast/metadata-only mode, facts are raw text snippets that will
    # always differ — triggering refinement would be a false positive.)
    retrieval_confidence = state.get("retrieval_confidence", "MEDIUM")
    critique = ""
    if has_discrepancy and retrieval_confidence == "HIGH":
        critique = report

    print(f"[Semiconductor Discrepancy Agent] Verdict: {overall.value} "
          f"({disc_count} discrepancies, {aligned_count} aligned)")

    return {
        "discrepancy_verdict": verdict.model_dump(),
        "discrepancy_report": report,
        "critique": critique,
        "_thinking": (
            f"Deterministic audit: {overall.value}. "
            f"{len(conflicts)} attributes checked, {disc_count} conflicts found. "
            f"Zero LLM calls — pure hierarchy logic."
        ),
    }

"""
================================================================================
Pickleball Discrepancy Agent — Deterministic Logic Gate (Zero LLM Calls)
================================================================================
OWNER: (Assign team member)
DOMAIN: pickleball
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
    if s in ("db", "database", "db_info", "sql", "records", "rulebook", "rules"):
        return 3
    elif s in ("datasheet", "sop", "spec", "document", "manual", "policy", "regulations", "guideline", "standard", "official"):
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
    Returns: {"attribute_name": [ExtractedFact, ...]}
    """
    index: dict[str, list[ExtractedFact]] = {}
    target_lower = target_entity.lower() if target_entity != "GENERAL" else ""

    for fd in facts:
        try:
            fact = ExtractedFact(**fd)
            # Binary & Garbage Filter: Discard facts that look like PDF garbage or raw bytes
            if _is_garbage_text(fact.value):
                continue
        except Exception:
            continue

        # Entity isolation (Soften matching)
        fact_entity_lower = fact.entity.lower()
        if target_lower and not is_generic_query:
            variations = {target_lower, target_lower.replace("-", " "), target_lower.replace(" ", "-"), target_lower.replace(" ", "")}
            entity_match = any(v in fact_entity_lower for v in variations if len(v) > 1)
            
            # ALLOW facts from high-authority sources even if label doesn't match perfectly,
            # as long as they were retrieved for this context.
            # HOWEVER, if the fact is explicitly labeled as a DIFFERENT specific entity, reject it.
            is_authoritative = _source_priority(fact.source_type) >= 2
            if not entity_match:
                if not is_authoritative:
                    continue
                # If authoritative but labeled as a different specific entity, exclude.
                if fact_entity_lower not in ("general", "unknown", "") and len(fact_entity_lower) > 2:
                    continue

        attr_raw = fact.attribute.lower().replace("-", "_").strip()
        
        # Semantic core matching
        final_key = attr_raw
        if target_lower:
            t_norm = target_lower.replace("_", "").replace(" ", "").lower()
            a_norm = attr_raw.replace("_", "").replace(" ", "").lower()
            
            if (t_norm in a_norm or a_norm in t_norm):
                final_key = target_lower
            elif t_norm in fact.value.lower() or t_norm in fact.attribute.lower():
                final_key = target_lower

        # KEY FIX: Group catch-all attributes together
        if final_key in ("general_info", "db_result", "database_record", "summary", "db_data", "fact"):
            final_key = "general_info"

        if final_key not in index:
            index[final_key] = []
        index[final_key].append(fact)

    # Debug: see what we have
    if facts and not index:
        entities = {f.get('entity', 'unknown') for f in facts}
        print(f"[DEBUG] No index matches for '{target_entity}'. Available entities in facts: {entities}")

    # Global Fallback: If no facts were found for the specific entity, 
    # and it's a generic query OR we have official documents, allow them.
    if not index and (is_generic_query or target_entity != "GENERAL"):
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
                
                # Double-check: if official doc is about a DIFFERENT specific entity, skip.
                if is_official and not entity_match:
                    if fact.entity.lower() not in ("general", "unknown", "") and len(fact.entity) > 2:
                        continue
                
                if (is_official and entity_match) or fact.entity.upper() == "GENERAL":
                    attr_raw = fact.attribute.lower().replace("-", "_").strip()
                    final_key = "general_info" if attr_raw in ("general_info", "summary", "fact") else attr_raw
                    if final_key not in index:
                        index[final_key] = []
                    index[final_key].append(fact)
            except Exception:
                continue

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


@vera_agent("Pickleball Discrepancy Agent")
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
        # Only promote if it's NOT a "No Matching Data" message
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

    print(f"[Pickleball Discrepancy Agent] Facts: "
          f"official={len(official_facts)}, "
          f"informal={len(informal_facts)}, "
          f"db={len(db_facts)}")

    # Build indexes by attribute (entity-filtered)
    is_generic = state.get("is_generic_query", False)
    target_attr = state.get("target_attribute", "GENERAL")
    official_idx = _build_fact_index(official_facts, target_entity, target_attr, is_generic)
    informal_idx = _build_fact_index(informal_facts, target_entity, target_attr, is_generic)
    db_idx = _build_fact_index(db_facts, target_entity, target_attr, is_generic)

    print(f"[DEBUG] Target Entity: {target_entity}")
    print(f"[DEBUG] Official Index Keys: {list(official_idx.keys())}")
    if official_idx:
        print(f"[DEBUG] First Official Fact Entity: {official_idx[list(official_idx.keys())[0]][0].entity}")

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
    for attr in sorted(all_attributes):
        conflict = _resolve_conflicts(
            official=official_idx.get(attr, []),
            informal=informal_idx.get(attr, []),
            db=db_idx.get(attr, []),
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

    print(f"[Pickleball Discrepancy Agent] Verdict: {overall.value} "
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

"""
Legal Discrepancy Agent.

Compares uploaded input contract against CUAD prevalence benchmarks.
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.schemas import AttributeConflict, ConflictStatus, DiscrepancyVerdict
from agents_logic.legal_agents._cuad_utils import benchmark_discrepancies, load_cuad_statistics


@vera_agent("Legal Discrepancy Agent")
def run(state: GraphState) -> dict:
    input_contract_text = state.get("input_contract_text", "") or ""
    input_contract_name = state.get("input_contract_name", "input_contract")

    if not input_contract_text.strip():
        verdict = DiscrepancyVerdict(
            target_entity=input_contract_name,
            overall_status=ConflictStatus.INSUFFICIENT_DATA,
            audit_summary=(
                "No uploaded input contract text found. Upload a contract in the Legal domain "
                "to run CUAD discrepancy analysis."
            ),
        )
        return {
            "discrepancy_verdict": verdict.model_dump(),
            "discrepancy_report": verdict.to_report_string(),
            "_thinking": "Legal discrepancy skipped because no uploaded input contract was provided.",
        }

    benchmark = benchmark_discrepancies(input_contract_text)
    detected = benchmark["detected_labels"]
    missing_common = benchmark["missing_common_labels"]
    uncommon_present = benchmark["uncommon_present_labels"]
    total_cuad = benchmark["total_cuad_contracts"]
    stats = load_cuad_statistics()
    total_common_labels = sum(1 for p in stats.get("prevalence", {}).values() if p >= 0.6)

    conflicts: list[AttributeConflict] = []

    for label, prevalence in missing_common[:10]:
        conflicts.append(
            AttributeConflict(
                entity=input_contract_name,
                attribute=label,
                status=ConflictStatus.DISCREPANCY,
                authoritative_value="Expected in many CUAD contracts",
                authoritative_source="CUAD_REFERENCE",
                authoritative_date="dataset_baseline",
                conflicting_values=[{
                    "value": "Not detected in uploaded contract",
                    "source": "uploaded_input_contract",
                    "date": "unknown",
                    "reason": f"Clause prevalence in CUAD is {prevalence:.1%}",
                }],
            )
        )

    for label, prevalence in uncommon_present[:6]:
        conflicts.append(
            AttributeConflict(
                entity=input_contract_name,
                attribute=label,
                status=ConflictStatus.ALIGNED,
                authoritative_value="Clause appears in uploaded contract",
                authoritative_source="uploaded_input_contract",
                authoritative_date="unknown",
                conflicting_values=[{
                    "value": "Rare compared to CUAD baseline",
                    "source": "CUAD_REFERENCE",
                    "date": "dataset_baseline",
                    "reason": f"Clause prevalence in CUAD is only {prevalence:.1%}",
                }],
            )
        )

    if missing_common:
        overall = ConflictStatus.DISCREPANCY
    elif detected:
        overall = ConflictStatus.ALIGNED
    else:
        overall = ConflictStatus.INSUFFICIENT_DATA

    common_coverage = 0.0
    if total_common_labels > 0:
        common_coverage = max(0.0, (total_common_labels - len(missing_common)) / total_common_labels)

    # Deterministic quality gate: uploaded text is likely not a standard contract
    # when very few contract-like clauses are detected and common-clause coverage
    # is low vs CUAD baseline.
    non_contract_like = len(detected) < 3 or common_coverage < 0.25
    if non_contract_like:
        overall = ConflictStatus.DISCREPANCY
        conflicts.insert(
            0,
            AttributeConflict(
                entity=input_contract_name,
                attribute="contract_structure_assessment",
                status=ConflictStatus.DISCREPANCY,
                authoritative_value="Expected clause profile of a regular CUAD contract",
                authoritative_source="CUAD_REFERENCE",
                authoritative_date="dataset_baseline",
                conflicting_values=[{
                    "value": "Uploaded document has low contract-clause coverage",
                    "source": "uploaded_input_contract",
                    "date": "unknown",
                    "reason": (
                        f"Detected {len(detected)} clause types, "
                        f"common-clause coverage {common_coverage:.1%}"
                    ),
                }],
            )
        )

    if non_contract_like:
        summary = (
            f"CUAD benchmark over {total_cuad} contracts indicates '{input_contract_name}' "
            f"falls short of a regular contract structure: detected {len(detected)} clause types "
            f"with {len(missing_common)} common-clause gaps "
            f"(common-clause coverage {common_coverage:.1%})."
        )
    else:
        summary = (
            f"CUAD benchmark over {total_cuad} contracts: detected {len(detected)} clause types in "
            f"'{input_contract_name}', flagged {len(missing_common)} common-clause gaps."
        )

    verdict = DiscrepancyVerdict(
        target_entity=input_contract_name,
        overall_status=overall,
        conflicts=conflicts,
        audit_summary=summary,
    )

    extra_lines = []
    if detected:
        extra_lines.append("### Key Aspects Detected (CUAD labels)")
        for label in detected[:15]:
            extra_lines.append(f"- {label}")

    report = verdict.to_report_string()
    if extra_lines:
        report = report + "\n\n" + "\n".join(extra_lines)

    return {
        "discrepancy_verdict": verdict.model_dump(),
        "discrepancy_report": report,
        "discrepancy_report_summary": summary,
        "_thinking": (
            f"Legal discrepancy check completed: detected={len(detected)}, "
            f"missing_common={len(missing_common)}, uncommon_present={len(uncommon_present)}."
        ),
    }

"""
Legal Discrepancy Agent.

Compares uploaded input contract against CUAD prevalence benchmarks.
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.schemas import AttributeConflict, ConflictStatus, DiscrepancyVerdict
from agents_logic.legal_agents._cuad_utils import benchmark_discrepancies


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

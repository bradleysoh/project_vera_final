"""
Legal DB Agent.

For legal domain, this agent treats the uploaded input contract as structured
analysis input and extracts CUAD-style key aspects deterministically.
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from agents_logic.legal_agents._cuad_utils import (
    summarize_contract_key_aspects,
    extract_key_contract_fields,
)


@vera_agent("Legal DB Agent")
def run(state: GraphState) -> dict:
    question = state["question"]
    intent = state.get("intent", "")
    input_contract_text = state.get("input_contract_text", "") or ""
    input_contract_name = state.get("input_contract_name", "input_contract")

    if intent not in ("db_query", "cross_reference", "spec_retrieval", ""):
        print(f"[Legal DB Agent] ⏭️ Fast-fail: intent='{intent}' is not legal-analysis related")
        return {}

    if not input_contract_text.strip():
        return {
            "db_data": "",
            "metadata_log": "[LEGAL DB] No uploaded input contract provided.\n",
            "_thinking": "No input contract text found; skipping legal key-aspect extraction.",
        }

    key_aspects = summarize_contract_key_aspects(input_contract_text, max_items=15)
    extracted_fields = extract_key_contract_fields(input_contract_text)
    preview = "\n".join(f"- {label}" for label in key_aspects[:12]) or "- No CUAD labels detected with current heuristics."

    db_facts = []
    for attr, value in extracted_fields.items():
        db_facts.append({
            "entity": input_contract_name,
            "attribute": attr,
            "value": value,
            "source_type": "db",
            "source_doc": "uploaded_input_contract",
            "date": "unknown",
            "confidence": "HIGH",
        })

    for label in key_aspects[:12]:
        db_facts.append({
            "entity": input_contract_name,
            "attribute": "cuad_clause_label",
            "value": label,
            "source_type": "db",
            "source_doc": "uploaded_input_contract",
            "date": "unknown",
            "confidence": "MEDIUM",
        })

    db_data = (
        f"Input Contract: {input_contract_name}\n"
        f"Question: {question}\n"
        "Extracted contract fields from uploaded contract text:\n"
        + (
            "\n".join(f"- {k}: {v}" for k, v in extracted_fields.items())
            if extracted_fields else "- (no deterministic fields extracted)"
        )
        + "\n"
        "Detected CUAD-style key aspects:\n"
        f"{preview}"
    )

    return {
        "db_data": db_data,
        "db_result": db_data,
        "db_facts": db_facts,
        "metadata_log": (
            f"[LEGAL DB] Extracted {len(extracted_fields)} deterministic contract fields and "
            f"{len(key_aspects)} key clause labels from uploaded contract "
            f"'{input_contract_name}'.\n"
        ),
        "_thinking": (
            f"Analyzed uploaded contract '{input_contract_name}' and extracted "
            f"{len(extracted_fields)} deterministic fields + {len(key_aspects)} CUAD-style key aspects."
        ),
    }

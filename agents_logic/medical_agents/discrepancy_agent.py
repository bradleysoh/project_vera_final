"""
================================================================================
Discrepancy Detection Agent (Case Agent) — Medical Domain
================================================================================
OWNER: (Assign team member)
DOMAIN: medical (logic is domain-agnostic)
RESPONSIBILITY: Detect conflicts in medical documents, scoped to the
                SPECIFIC ENTITY the user asked about.
                Uses an Entity-Attribute-Value (EAV) comparison model.
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


@vera_agent("Case Agent")
def run(state: GraphState) -> dict:
    """
    MEDICAL CASE AGENT: Entity-scoped discrepancy analysis.

    Three-layer anti-hallucination approach:
      1. Extract the primary entity from the user's question (LLM).
      2. Run an EAV-model comparison prompt scoped to that entity.
      3. Post-LLM validation: discard results that don't mention the entity.

    Returns:
        dict: {"generation": str, "discrepancy_report": str}
    """
    documents = state["documents"]
    generation = state["generation"]
    question = state["question"]
    domain = state.get("user_domain", "medical")
    retrieved_docs = state.get("retrieved_docs") or {}

    print(f"[Case Agent ({domain})] Checking for discrepancies across "
          f"{len(documents)} documents...")

    # ── Step 1: Extract the primary entity ──────────────────────────────
    entity = _extract_entity(question)
    is_general = entity == "GENERAL_QUERY"
    print(f"[Case Agent ({domain})] Target entity: "
          f"{'(general query)' if is_general else entity}")

    # ── Gather documents ────────────────────────────────────────────────
    tech_docs = retrieved_docs.get("tech", [])
    compliance_docs = retrieved_docs.get("compliance", [])

    if tech_docs and compliance_docs:
        all_docs = tech_docs + compliance_docs
        print(f"[Case Agent ({domain})] Cross-agent mode: {len(tech_docs)} tech + "
              f"{len(compliance_docs)} compliance docs")
    else:
        all_docs = documents

    # ── Format documents for analysis ───────────────────────────────────
    doc_summaries = []
    for doc in all_docs:
        source = doc.metadata.get('source', 'unknown').upper()
        doc_id = doc.metadata.get('document_id', 'N/A')
        doc_summaries.append(
            f"[{source}] ({doc_id}): {doc.page_content[:500]}"
        )

    docs_text = "\n\n".join(doc_summaries)

    # Include DB SQL results if available
    db_result = state.get("db_result", "")
    db_section = ""
    if db_result and "NO_RELEVANT_TABLE" not in db_result:
        db_section = f"\n\nDATABASE QUERY RESULTS:\n{db_result}"

    # ── Step 2: Entity-scoped EAV discrepancy prompt ────────────────────
    entity_scope_instruction = (
        f"ENTITY SCOPE: The user is asking about \"{entity}\".\n"
        f"You MUST ONLY report discrepancies that involve \"{entity}\".\n"
        f"If a discrepancy involves a DIFFERENT entity (e.g., a different "
        f"product, patient, or lot), do NOT include it even if it appears "
        f"in the documents.\n\n"
    ) if not is_general else ""

    discrepancy_prompt = ChatPromptTemplate.from_messages([
        ("human", (
            "You are a discrepancy detection agent. Compare the documents "
            "and database results below using an Entity-Attribute-Value (EAV) "
            "analysis.\n\n"
            f"{entity_scope_instruction}"
            "For each potential discrepancy, verify:\n"
            "  1. Are BOTH sources referring to the SAME entity?\n"
            "  2. Are they describing the SAME attribute (e.g., voltage, "
            "dosage, temperature)?\n"
            "  3. Do the values CONTRADICT each other?\n\n"
            "Only report a discrepancy if ALL THREE conditions are met.\n\n"
            "DOCUMENTS:\n{docs_text}\n"
            "{db_section}\n\n"
            "QUESTION: {question}\n\n"
            "Check specifically for:\n"
            "1. Document vs Document conflicts for the queried entity\n"
            "2. Document vs Database conflicts for the queried entity\n"
            "3. Version discrepancies across sources for the queried entity\n\n"
            "If contradictions exist for the queried entity, list each one "
            "with source references.\n"
            "If none, say exactly: NO_DISCREPANCY_FOUND"
        ))
    ])

    chain = discrepancy_prompt | config.llm | StrOutputParser()
    discrepancy_result = llm_invoke_with_retry(chain, {
        "docs_text": docs_text,
        "db_section": db_section,
        "question": question,
    })

    # ── Step 3: Post-LLM validation ─────────────────────────────────────
    if (not is_general
            and entity.lower() not in discrepancy_result.lower()
            and "NO_DISCREPANCY_FOUND" not in discrepancy_result):
        print(f"[Case Agent ({domain})] ⚠️  Post-validation: discrepancy "
              f"result does not mention '{entity}' — overriding to NO_DISCREPANCY_FOUND")
        discrepancy_result = "NO_DISCREPANCY_FOUND"

    # ── Evaluate result ─────────────────────────────────────────────────
    if "NO_DISCREPANCY_FOUND" in discrepancy_result or len(discrepancy_result) < 10:
        print(f"[Case Agent ({domain})] ✅ No material discrepancies "
              f"detected for {entity}.")
        return {
            "generation": generation,
            "discrepancy_report": state.get("discrepancy_report", ""),
            "critique": "",
            "_thinking": (
                f"Compared {len(all_docs)} docs "
                f"(entity: '{entity}'). "
                f"No contradictions found for query: '{question[:50]}'."
            ),
        }
    else:
        print(f"[Case Agent ({domain})] ⚠️  DISCREPANCY DETECTED for {entity}!")

        current_refinement_count = state.get("refinement_count", 0)
        MAX_REFINEMENTS = state.get("max_refinements", 3)

        if current_refinement_count < MAX_REFINEMENTS:
            print(f"[Case Agent ({domain})] 🔄 Triggering refinement loop "
                  f"({current_refinement_count + 1}/{MAX_REFINEMENTS})")
            return {
                "discrepancy_report": discrepancy_result,
                "critique": discrepancy_result,
                "refinement_count": current_refinement_count + 1,
                "_thinking": (
                    f"⚠️ DISCREPANCY DETECTED for '{entity}' across "
                    f"{len(all_docs)} docs. Triggering refinement loop "
                    f"({current_refinement_count + 1}/{MAX_REFINEMENTS})."
                ),
            }
        else:
            print(f"[Case Agent ({domain})] 🛑 Max refinements reached. "
                  f"Finalizing report.")

            return {
                "generation": generation,
                "discrepancy_report": discrepancy_result,
                "critique": "",
                "_thinking": (
                    f"⚠️ DISCREPANCY DETECTED for '{entity}'. "
                    f"Max refinements ({MAX_REFINEMENTS}) reached. "
                    f"Finalizing report."
                ),
            }

"""
================================================================================
Response Generator Agent (Response Agent)
================================================================================
RESPONSIBILITY: Synthesize final answers from retrieved documents and DB records.
                Handles refinement loops if discrepancies are flagged.
                Enforces DB Authority and Entity Isolation rules.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
import shared.config as config
from shared.config import llm_invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@vera_agent("Response Agent")
def run(state: GraphState) -> dict:
    """
    Generates the final response using retrieved documents and DB results.
    Refines the answer if a critique is present.
    """
    question = state["question"]
    documents = state["documents"]
    critique = state.get("critique", "")
    db_result = state.get("db_result", "")
    
    # Combined context for the LLM
    context_text = ""
    if documents:
        doc_texts = []
        for d in documents:
            src = d.metadata.get("source", "unknown").upper()
            doc_texts.append(f"[{src}] {d.page_content}")
        context_text += "--- RETRIEVED DOCUMENTS ---\n" + "\n\n".join(doc_texts)
    
    if db_result and "NO_MATCHING_DATA" not in db_result:
        context_text += "\n\n--- DATABASE RESULTS (AUTHORITATIVE) ---\n" + db_result

    # System Instruction - Enforcing DB Authority & Entity Isolation
    if critique:
        print(f"[Response Agent] 🔄 Refinement iteration. Critique: {critique[:100]}...")
        system_instruction = (
            "You are VERA (Virtual Engineering Review Agent). You previously provided "
            "an answer that has been FLAGGED FOR DISCREPANCIES by the Case Agent.\n\n"
            "Your goal is to REWRITE the answer to address these conflicts.\n\n"
            "CRITIQUE (Discrepancy Report):\n"
            f"{critique}\n\n"
            "INSTRUCTIONS:\n"
            "1. Review the original documents and the critique carefully.\n"
            "2. Acknowledge the conflict explicitly (e.g., 'Note: There is a "
            "conflict between...').\n"
            "3. DB AUTHORITY RULE: If the Database (SQL) results provide a specific "
            "value but the text documents are silent or ambiguous, TREAT THE "
            "DATABASE AS THE PRIMARY SOURCE OF TRUTH.\n"
            "4. Provide a corrected, nuanced answer.\n\n"
            "ENTITY ISOLATION RULE:\n"
            "Your revised answer must ONLY address the specific entity, product, "
            "or subject the user originally asked about. Do NOT include "
            "information or discrepancies for other entities even if they "
            "appear in the documents.\n"
        )
    else:
        print(f"[Response Agent] Generating initial response...")
        system_instruction = (
            "You are VERA (Virtual Engineering Review Agent), a professional "
            "AI assistant that helps users find and verify information from "
            "technical documents, SOPs, database records, and internal "
            "communications.\n\n"
            "INSTRUCTIONS:\n"
            "1. DIRECT ANSWER FIRST: Begin your response with a clear, concise "
            "answer to the user's specific question.\n"
            "2. DB AUTHORITY RULE: If the Database (SQL) results provide a "
            "specific value/fact but the text documents (PDFs) are silent "
            "or ambiguous, TREAT THE DATABASE AS THE PRIMARY SOURCE OF TRUTH.\n"
            "3. CROSS-CHECK PROTOCOL: Explicitly state if information comes "
            "from the Database vs Documents. If they conflict, highlight it.\n"
            "4. ENTITY ISOLATION: Answer ONLY about the specific entity, "
            "product, patient, lot, or subject the user asked about. Even if "
            "the documents mention other entities, do NOT include their data "
            "in your answer.\n"
            "5. Always cite the source type (DATASHEET, EMAIL, SOP, DB_INFO, "
            "DOCUMENT) for each piece of information.\n"
            "6. Be precise and technical in your response.\n"
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
        "question": question
    })

    return {
        "generation": response,
        "documents": documents,
        "critique": critique,  # Pass through
        "_thinking": "Generated response with DB Authority and Entity Isolation rules."
    }

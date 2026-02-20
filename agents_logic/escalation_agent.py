"""
================================================================================
Escalation Agent — Unauthorized Access & Detailed Context Summaries
================================================================================
OWNER: (Member 5)
RESPONSIBILITY: Handle security violations with detailed escalation summaries.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
import shared.config as config
from shared.config import llm_invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@vera_agent("Escalation Agent")
def run(state: GraphState) -> dict:
    """
    ESCALATION NODE: Handles unauthorized access and out-of-domain queries.

    Triggered when:
      1. A junior user attempts to access internal/confidential documents
      2. A user asks a question outside their assigned domain
      3. The system has low confidence in its routing or response

    Uses LLM to generate a detailed summary explaining WHY escalation
    is necessary, without revealing the actual restricted content.

    Returns:
        dict: {"generation": str, "documents": []}
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "")
    router_log = state.get("metadata_log", "")

    print(f"[Escalation Agent] ⚠️  Escalation triggered!")

    # Determine escalation reason from router metadata
    is_out_of_domain = "OUT-OF-DOMAIN" in router_log

    if is_out_of_domain:
        reason_context = (
            f"This is an OUT-OF-DOMAIN escalation.\n"
            f"User's assigned domain: {user_domain}\n"
            f"The query appears to target a different domain.\n"
        )
    else:
        reason_context = (
            f"This is a SECURITY escalation.\n"
            f"The user with role '{user_role}' attempted to access "
            f"restricted information.\n"
        )

    # Generate a detailed escalation summary using LLM
    escalation_prompt = ChatPromptTemplate.from_messages([
        ("human", (
            "You are a security escalation system for a multi-domain company.\n\n"
            "Context:\n{reason_context}\n"
            "User Role: {user_role}\n"
            "User Domain: {user_domain}\n"
            "Query: '{question}'\n\n"
            "Generate a DETAILED ESCALATION SUMMARY that includes:\n"
            "1. REASON: Why this query requires escalation\n"
            "2. RISK ASSESSMENT: What kind of restricted information "
            "the query is seeking (without revealing actual content)\n"
            "3. RECOMMENDED ACTION: What a supervisor should do\n\n"
            "ESCALATION SUMMARY:"
        ))
    ])

    chain = escalation_prompt | config.llm | StrOutputParser()
    try:
        detailed_summary = llm_invoke_with_retry(chain, {
            "question": question,
            "user_role": user_role,
            "user_domain": user_domain,
            "reason_context": reason_context,
        })
    except Exception:
        detailed_summary = (
            "Unable to generate detailed summary. "
            "Manual review of the query context is required."
        )

    escalation_type = "🌐 OUT-OF-DOMAIN" if is_out_of_domain else "🔒 SECURITY"

    escalation_message = (
        f"🚨 ESCALATION NOTICE — {escalation_type} 🚨\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "This query has been escalated to a Human Supervisor.\n\n"
        f"User Role: {user_role}\n"
        f"User Domain: {user_domain}\n"
        f"Query: {question}\n\n"
        f"{'─'*40}\n"
        f"📋 DETAILED ESCALATION SUMMARY\n"
        f"{'─'*40}\n"
        f"{detailed_summary}\n\n"
        f"{'─'*40}\n"
        "Action Required:\n"
        "  • A supervisor must review this request\n"
    )

    if is_out_of_domain:
        escalation_message += (
            "  • If cross-domain access is needed, request domain authorization\n"
        )
    else:
        escalation_message += (
            "  • If access is legitimate, please upgrade the user's role\n"
        )

    escalation_message += (
        "  • This event has been logged for audit purposes\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # --- Send email alert to supervisor ---
    try:
        from shared.email_utils import send_escalation_email
        email_subject = f"{escalation_type} — {user_role} user in {user_domain}"
        send_escalation_email(email_subject, escalation_message)
    except Exception as e:
        print(f"[Escalation Agent] Email alert failed: {e}")

    reason = "out-of-domain" if is_out_of_domain else "security"
    thinking = (
        f"Escalation triggered: {reason}. "
        f"User role='{user_role}', domain='{user_domain}'. "
        f"Query attempted: '{question[:60]}'."
    )

    return {"generation": escalation_message, "documents": [], "_thinking": thinking}

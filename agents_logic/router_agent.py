"""
================================================================================
Router Agent — Query Classification, Security Check & Domain Routing
================================================================================
OWNER: (Member 1)
RESPONSIBILITY: Classify query intent, detect domain, perform RBAC security,
                and detect out-of-domain queries.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
import shared.config as config
from shared.config import llm_invoke_with_retry, DOMAIN_KEYWORDS
from shared.dynamic_loader import get_available_domains
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@vera_agent("Router Agent")
def run(state: GraphState) -> dict:
    """
    ROUTER NODE: Classifies intent, detects domain, performs security checks.
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "semiconductor")

    # --- Step 1: Classify intent using keyword matching (NOT LLM) ---
    # The 1B model misclassifies "voltage limit" as "compliance".
    # Keyword matching is instant, deterministic, and never wrong.
    q_lower = question.lower()

    TECHNICAL_KEYWORDS = [
        "voltage", "thermal", "datasheet", "spec", "limit", "current",
        "power", "temperature", "watt", "rtx", "chip", "die", "wafer",
        "burn-in", "silicon", "yield", "performance", "frequency", "clock",
        "maximum", "minimum", "rating", "tolerance",
        "production", "lot", "batch", "database", "db",
    ]
    COMPLIANCE_KEYWORDS = [
        "sop", "audit", "email", "procedure", "quality", "regulation",
        "compliance", "checklist", "process change", "internal decision",
        "waiver", "approval", "sign-off", "certification",
        "record", "tracking", "inventory", "report",
    ]

    tech_score = sum(1 for kw in TECHNICAL_KEYWORDS if kw in q_lower)
    comp_score = sum(1 for kw in COMPLIANCE_KEYWORDS if kw in q_lower)

    if comp_score > tech_score:
        route = "compliance"
    else:
        route = "technical"  # default to technical

    print(f"[Router Agent] Classified intent: {route} (tech={tech_score}, comp={comp_score})")

    # --- Step 2: Determine query domain ---
    # If user_domain is already set (from UI/session), trust it directly.
    # Only use LLM-based detection when no domain is pre-assigned.
    available_domains = get_available_domains()
    if not available_domains:
        available_domains = ["semiconductor"]

    flagged = False
    metadata_log = ""

    if user_domain and user_domain in available_domains:
        # User's domain is pre-assigned from the UI — use it directly
        detected_domain = user_domain
        print(f"[Router Agent] Using pre-assigned domain: {detected_domain}")
    else:
        # No domain set — use LLM to detect dynamically
        domain_keywords_str = ""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if domain in available_domains:
                domain_keywords_str += f"- '{domain}': {keywords}\n"
        for domain in available_domains:
            if domain not in DOMAIN_KEYWORDS:
                domain_keywords_str += f"- '{domain}': General queries for {domain}\n"

        domain_list_str = ", ".join(f"'{d}'" for d in available_domains)
        DOMAIN_ALIASES = {"engineering": "semiconductor"}

        domain_prompt = ChatPromptTemplate.from_messages([
            ("human", (
                "You are a domain classifier. Determine which domain this "
                "question belongs to.\n\n"
                f"Available domains: {domain_list_str}\n\n"
                "Domain Keywords:\n"
                f"{domain_keywords_str}\n"
                "Question: {question}\n\n"
                f"Respond with ONLY one of: {domain_list_str}. "
                "No other text."
            ))
        ])

        domain_chain = domain_prompt | config.llm | StrOutputParser()
        detected_domain = llm_invoke_with_retry(
            domain_chain, {"question": question}
        ).strip().lower()

        # Resolve aliases and validate
        detected_domain = DOMAIN_ALIASES.get(detected_domain, detected_domain)
        if detected_domain not in available_domains:
            matched = False
            for d in available_domains:
                if d in detected_domain or detected_domain in d:
                    detected_domain = d
                    matched = True
                    break
            if not matched:
                detected_domain = available_domains[0]

        print(f"[Router Agent] LLM detected domain: {detected_domain}")

    # --- Step 3: Security check for junior users ---
    if not flagged and user_role == "junior":
        security_prompt = ChatPromptTemplate.from_messages([
            ("human", (
                "You are a security classifier for a semiconductor company.\n\n"
                "A JUNIOR INTERN is asking the following question:\n"
                "'{question}'\n\n"
                "Determine if this question is trying to access INTERNAL or "
                "CONFIDENTIAL information. Look for keywords like:\n"
                "- 'internal', 'confidential', 'secret', 'private'\n"
                "- 'email', 'informal decision', 'undocumented change'\n"
                "- 'waiver', 'skip', 'bypass'\n"
                "- References to internal communications\n\n"
                "Respond with ONLY 'yes' or 'no'."
            ))
        ])

        security_chain = security_prompt | config.llm | StrOutputParser()
        security_result = llm_invoke_with_retry(
            security_chain, {"question": question}
        ).strip().lower()

        if "yes" in security_result:
            flagged = True
            metadata_log += (
                "[ROUTER] ⚠️ SECURITY FLAG: Junior user attempting "
                "restricted data access\n"
            )
            print("[Router Agent] ⚠️ SECURITY FLAG: Junior user attempting restricted data")

    # Use detected domain for routing (next_agent drives which domain's agents run)
    next_agent = detected_domain

    # --- Build thinking trace ---
    thinking = (
        f"User role='{user_role}', domain='{user_domain}'. "
        f"Intent: '{route}' (tech={tech_score}, comp={comp_score}). "
        f"Detected domain: '{next_agent}'. Flagged: {flagged}."
    )

    return {
        "route": route,
        "flagged": flagged,
        "next_agent": next_agent,
        "metadata_log": metadata_log,
        "_thinking": thinking,
    }


def decide_route(state: GraphState) -> str:
    """
    Conditional edge function: Routes to the start of the domain's agent chain.
    """
    if state.get("flagged", False):
        print("[ROUTING] -> Escalation (security flag or out-of-domain)")
        return "escalate"

    # Simplified routing: The router just identifies the domain.
    # app.py handles mapping the domain name to the correct start node (e.g. db_agent).
    domain = state.get("next_agent", "semiconductor")
    print(f"[ROUTING] -> {domain} (Start of Chain)")
    return domain

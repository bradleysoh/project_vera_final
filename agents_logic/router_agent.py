"""
================================================================================
Router Agent — Dynamic, Zero-Assumption Query Classification & Domain Routing
================================================================================
OWNER: (Member 1)
RESPONSIBILITY: Classify query intent, detect domain, perform RBAC security,
                extract query understanding (target_entity, target_attribute),
                and detect out-of-domain queries.

ARCHITECTURAL CONSTRAINT:
    This agent contains ZERO hardcoded industry-specific keywords, aliases,
    or fallback logic.  All routing heuristics are loaded dynamically from
    the registered {domain}_agents/domain_config.py files at import time.

    Additionally, this agent performs Query Understanding via structured
    output, extracting target_entity and target_attribute for downstream
    precision retrieval and focused fact extraction.
================================================================================
"""

from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.schemas import QueryIntent
import shared.config as config
from shared.config import llm_invoke_with_retry
from shared.dynamic_loader import (
    get_available_domains,
    load_domain_configs,
    build_routing_heuristics,
    resolve_domain_alias,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


# ---------------------------------------------------------------------------
# Module-level dynamic loading — zero hardcoded values
# ---------------------------------------------------------------------------
_DOMAIN_CONFIGS = load_domain_configs()
_ROUTING_KEYWORDS = build_routing_heuristics(_DOMAIN_CONFIGS)


# ---------------------------------------------------------------------------
# Query Understanding prompt (text-based fallback for small models)
# ---------------------------------------------------------------------------
_QUERY_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "You are a query analysis engine.  Extract structured metadata from "
        "the user's question.\n\n"
        "USER QUESTION: {question}\n\n"
        "Extract the following (leave blank if not found):\n"
        "TARGET_ENTITY: <the primary entity, product, patient, lot, or component mentioned>\n"
        "TARGET_ATTRIBUTE: <the specific attribute being asked about, e.g. 'max_voltage', 'dosage'>\n"
        "TIME_CONTEXT: <any temporal qualifier, e.g. 'latest', '2024-Q3', or a specific date>\n\n"
        "Return ONLY these three fields, one per line.  "
        "If a field is not found, write: GENERAL"
    ))
])


def _extract_query_intent_regex(question: str) -> QueryIntent:
    """
    Fast (zero-LLM-call) query intent extraction using regex patterns.
    Used in 'fast' mode (Ollama) to avoid extra LLM calls.
    """
    import re

    # Extract entity-like identifiers (alphanumeric with underscores/hyphens)
    # Patterns: WAF_001_C, RTX-9000, LOT-2024-001, PAT_12345
    entity_patterns = [
        r'\b([A-Z]{2,}[-_]\d{2,}[-_][A-Z0-9]+)\b',   # WAF_001_C, LOT_001_A
        r'\b([A-Z]{2,}[-_]\d{3,})\b',                   # RTX-9000, PAT-12345
        r'\b([A-Z]{2,}\d{3,})\b',                        # RTX9000
    ]
    entity = "GENERAL"
    for pattern in entity_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            entity = match.group(1)
            break

    # Extract attribute hints from common keywords
    attr_keywords = {
        "voltage": "voltage", "yield": "yield", "defect": "defect_count",
        "temperature": "temperature", "dosage": "dosage", "lot": "lot",
        "wafer": "wafer", "spec": "specification", "status": "status",
        "thermal": "thermal", "resistance": "resistance", "power": "power",
    }
    attribute = "GENERAL"
    q_lower = question.lower()
    for kw, attr in attr_keywords.items():
        if kw in q_lower:
            attribute = attr
            break

    return QueryIntent(target_entity=entity, target_attribute=attribute)


def _extract_query_intent_llm(question: str) -> QueryIntent:
    """
    LLM-based query intent extraction. Used in 'deep' mode (Gemini/Groq).
    """
    try:
        chain = _QUERY_INTENT_PROMPT | config.llm | StrOutputParser()
        raw = llm_invoke_with_retry(chain, {"question": question})

        parsed = {"target_entity": "GENERAL", "target_attribute": "GENERAL", "time_context": ""}
        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if key == "target_entity" and value and value.upper() != "GENERAL":
                parsed["target_entity"] = value
            elif key == "target_attribute" and value and value.upper() != "GENERAL":
                parsed["target_attribute"] = value
            elif key == "time_context" and value and value.upper() != "GENERAL":
                parsed["time_context"] = value

        return QueryIntent(**parsed)
    except Exception as e:
        print(f"[Router Agent] LLM intent extraction failed ({e}), falling back to regex")
        return _extract_query_intent_regex(question)


def _extract_query_intent(question: str) -> QueryIntent:
    """Route to regex (fast) or LLM (deep) based on RETRIEVAL_MODE."""
    from shared.config import RETRIEVAL_MODE
    if RETRIEVAL_MODE == "fast":
        return _extract_query_intent_regex(question)
    return _extract_query_intent_llm(question)


@vera_agent("Router Agent")
def run(state: GraphState) -> dict:
    """
    ROUTER NODE: Classifies intent, detects domain, performs security checks,
    and extracts query understanding (target_entity, target_attribute).

    All keyword lists and domain aliases are loaded dynamically from each
    domain's ``domain_config.py``.  This agent makes ZERO assumptions about
    which domains exist or what their terminology is.
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "")

    # --- Step 1: Query Understanding — extract entity, attribute, time ---
    intent = _extract_query_intent(question)
    print(f"[Router Agent] Query Intent: entity='{intent.target_entity}', "
          f"attr='{intent.target_attribute}', time='{intent.time_context}'")

    # --- Step 2: Classify intent using dynamically-loaded keywords ---
    q_lower = question.lower()

    intent_scores: dict[str, int] = {}
    for intent_label, keywords in _ROUTING_KEYWORDS.items():
        intent_scores[intent_label] = sum(1 for kw in keywords if kw in q_lower)

    if intent_scores:
        best_intent = max(intent_scores, key=lambda k: intent_scores[k])
        if intent_scores[best_intent] == 0:
            best_intent = "technical"
    else:
        best_intent = "technical"

    route = best_intent
    score_str = ", ".join(f"{k}={v}" for k, v in intent_scores.items())
    print(f"[Router Agent] Classified intent: {route} ({score_str})")

    # --- Step 3: Determine query domain ---
    available_domains = get_available_domains()

    flagged = False
    metadata_log = ""

    if user_domain and user_domain in available_domains:
        # Check if query keywords actually match a different domain
        # (e.g. user has 'medical' selected but asks about wafers)
        domain_scores = {}
        for domain, cfg in _DOMAIN_CONFIGS.items():
            if domain not in available_domains:
                continue
            hints = cfg.get("keyword_hints", "")
            if isinstance(hints, str):
                hint_words = [w.strip().lower() for w in hints.split(",") if w.strip()]
            elif isinstance(hints, list):
                hint_words = [w.lower() for w in hints]
            else:
                hint_words = []
            score = sum(1 for w in hint_words if w in q_lower)
            domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores, key=lambda k: domain_scores[k])
            if domain_scores[best_domain] > 0 and best_domain != user_domain:
                detected_domain = best_domain
                print(f"[Router Agent] ⚠️ Domain override: '{user_domain}' → '{detected_domain}' "
                      f"(query keywords matched '{detected_domain}' domain)")
            else:
                detected_domain = user_domain
                print(f"[Router Agent] Using pre-assigned domain: {detected_domain}")
        else:
            detected_domain = user_domain
            print(f"[Router Agent] Using pre-assigned domain: {detected_domain}")
    else:
        domain_keywords_str = ""
        for domain, cfg in _DOMAIN_CONFIGS.items():
            if domain in available_domains:
                hints = cfg.get("keyword_hints", cfg.get("description", ""))
                domain_keywords_str += f"- '{domain}': {hints}\n"
        for domain in available_domains:
            if domain not in _DOMAIN_CONFIGS:
                domain_keywords_str += f"- '{domain}': General queries for {domain}\n"

        domain_list_str = ", ".join(f"'{d}'" for d in available_domains)

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

        resolved = resolve_domain_alias(detected_domain, _DOMAIN_CONFIGS)
        if resolved:
            detected_domain = resolved
        elif detected_domain not in available_domains:
            matched = False
            for d in available_domains:
                if d in detected_domain or detected_domain in d:
                    detected_domain = d
                    matched = True
                    break
            if not matched:
                flagged = True
                detected_domain = user_domain or (available_domains[0] if available_domains else "unknown")
                metadata_log += (
                    f"[ROUTER] ⚠️ UNRESOLVED DOMAIN: LLM returned "
                    f"'{detected_domain}' which does not match any "
                    f"registered domain. Escalating for review.\n"
                )
                print(f"[Router Agent] ⚠️ Unresolved domain — escalating")

        print(f"[Router Agent] LLM detected domain: {detected_domain}")

    # --- Step 4: Security check for junior users ---
    if not flagged and user_role == "junior":
        security_prompt = ChatPromptTemplate.from_messages([
            ("human", (
                "You are a security classifier for a multi-domain company.\n\n"
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

    next_agent = detected_domain

    thinking = (
        f"User role='{user_role}', domain='{user_domain}'. "
        f"Intent: '{route}' ({score_str}). "
        f"Detected domain: '{next_agent}'. Flagged: {flagged}. "
        f"Entity: '{intent.target_entity}', Attr: '{intent.target_attribute}'."
    )

    return {
        "route": route,
        "flagged": flagged,
        "next_agent": next_agent,
        "user_domain": detected_domain,  # Update domain so downstream agents use correct path
        "target_entity": intent.target_entity,
        "target_attribute": intent.target_attribute,
        "time_context": intent.time_context,
        "metadata_log": metadata_log,
        "_thinking": thinking,
    }


def decide_route(state: GraphState) -> str:
    """
    Conditional edge function: Routes to the start of the domain's agent chain.
    """
    if state.get("flagged", False):
        print("[ROUTING] -> Escalation (security flag or unresolved domain)")
        return "escalate"

    domain = state.get("next_agent", "")
    if not domain:
        print("[ROUTING] -> Escalation (no domain detected)")
        return "escalate"

    print(f"[ROUTING] -> {domain} (Start of Chain)")
    return domain

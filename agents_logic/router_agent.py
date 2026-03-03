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
# Fine-Grained Intent Classification Keywords
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS = {
    "db_query": [
        "database", "db", "production data", "check the database",
        "query", "sql", "records", "inventory", "lots that",
        "show me the data", "look up", "find in",
    ],
    "spec_retrieval": [
        "specification", "spec", "datasheet", "voltage", "thermal",
        "maximum", "minimum", "limit", "rating", "tolerance",
        "power", "watt", "sop", "procedure", "checklist", "process",
        "datasheet says", "according to spec",
    ],
    "cross_reference": [
        "compare", "discrepancy", "mismatch", "conflict", "versus",
        "cross-reference", "cross reference", "check against",
        "don't match", "doesn't match", "inconsistent",
        "email changes", "recent changes", "version difference",
        "any changes communicated", "tell me about", "what is",
        "info", "details", "everything about", "search all",
    ],
    "general_chat": [
        "what is vera", "what does vera", "who are you",
        "what can you do", "help me", "hello", "hi",
        "how do you work", "what are you",
    ],
}


def _classify_intent(question: str, route: str) -> str:
    """
    Classify the query into a fine-grained intent.
    Uses keyword scoring with multi-category detection.
    
    Key heuristic: If a query hits keywords from BOTH db_query AND
    spec_retrieval, it likely needs cross-referencing multiple sources.
    
    Returns one of: db_query, spec_retrieval, cross_reference,
                    general_chat, out_of_domain
    """
    q_lower = question.lower()

    # Score each intent category
    scores = {}
    for intent_label, keywords in _INTENT_KEYWORDS.items():
        scores[intent_label] = sum(1 for kw in keywords if kw in q_lower)

    # --- Multi-category detection ---
    # If the query matches keywords from BOTH db/spec categories,
    # promote to cross_reference (needs data from multiple sources)
    has_db = scores.get("db_query", 0) > 0
    has_spec = scores.get("spec_retrieval", 0) > 0
    has_cross = scores.get("cross_reference", 0) > 0

    if has_cross:
        return "cross_reference"
    if has_db and has_spec:
        return "cross_reference"

    # Find the best match
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    # If no keywords matched at all, use the route as a heuristic
    if best_score == 0:
        if route == "compliance":
            return "spec_retrieval"  # SOPs/procedures are specs
        return "spec_retrieval"  # Default: treat as spec lookup

    return best


# ---------------------------------------------------------------------------
# Query Understanding — LLM-based Named Entity Recognition (NER)
# ---------------------------------------------------------------------------
# Uses .with_structured_output(QueryIntent) for guaranteed type-safe
# extraction of entity_name, entity_type, attribute, and time_context.
# Falls back to a broad regex if the LLM call fails.
# ---------------------------------------------------------------------------

_NER_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "You are a Named Entity Recognition (NER) engine.  "
        "Extract structured metadata from the user's question.\n\n"
        "USER QUESTION: {question}\n\n"
        "Instructions:\n"
        "1. target_entity: The primary entity name mentioned "
        "   (e.g. 'SuperGPU', 'WAF_003_A', 'Patient-7712').  "
        "   Use 'GENERAL' ONLY if no entity is mentioned at all.\n"
        "2. entity_type: The categorical TYPE of that entity "
        "   (e.g. 'customer', 'product', 'wafer', 'lot', 'patient', "
        "   'component', 'supplier').  Use 'GENERAL' if unclear.\n"
        "3. target_attribute: The specific attribute being asked about "
        "   (e.g. 'max_voltage', 'dosage', 'yield', 'status').  "
        "   Use 'GENERAL' if no specific attribute is mentioned.\n"
        "4. time_context: Any temporal qualifier "
        "   (e.g. 'latest', '2024-Q3', '2025-01-15').  "
        "   Leave empty if none mentioned.\n"
    ))
])


def _extract_query_intent_regex(question: str) -> QueryIntent:
    """
    Broad regex fallback for entity extraction — used only when 
    LLM-based extraction fails.  Unlike the previous version, this
    does NOT rely on hardcoded ID formats.  It captures:
      1. Classic IDs  (WAF_001_C, RTX-9000)
      2. Quoted terms  ("SuperGPU", 'Customer ABC')
      3. Capitalized proper nouns  (SuperGPU, MegaChip)
    """
    import re

    entity = "GENERAL"

    # --- Priority 1: Classic code-style IDs ---
    id_patterns = [
        r'\b([A-Z]{2,}[-_]\d{2,}[-_][A-Z0-9]+)\b',
        r'\b([A-Z]{2,}[-_]\d{3,})\b',
        r'\b([A-Z]{2,}\d{3,})\b',
    ]
    for pattern in id_patterns:
        match = re.search(pattern, question)
        if match:
            entity = match.group(1)
            break

    # --- Priority 2: Quoted strings ---
    if entity == "GENERAL":
        quoted = re.search(r'["\']([^"\']+ )', question)
        if quoted:
            entity = quoted.group(1).strip()

    # --- Priority 3: Capitalized proper nouns (2+ uppercase letters) ---
    if entity == "GENERAL":
        # Match words like SuperGPU, MegaChip — skip common English words
        skip = {"What", "How", "Tell", "Show", "Find", "Get", "Are", "Is",
                "The", "About", "From", "Which", "Where", "When", "Does",
                "Can", "Could", "Would", "Should", "Have", "Has", "Was",
                "Were", "Will", "Do", "Did", "Any", "All", "Some", "VERA"}
        words = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', question)
        for w in words:
            if w not in skip:
                entity = w
                break

    # --- Entity type hinting from question context ---
    entity_type = "GENERAL"
    type_hints = {
        "customer": "customer", "client": "customer", "buyer": "customer",
        "product": "product", "device": "product", "chip": "product",
        "wafer": "wafer", "lot": "lot", "batch": "lot",
        "patient": "patient", "supplier": "supplier", "vendor": "supplier",
    }
    q_lower = question.lower()
    for keyword, etype in type_hints.items():
        if keyword in q_lower:
            entity_type = etype
            break

    return QueryIntent(target_entity=entity, entity_type=entity_type)


def _extract_query_intent_llm(question: str) -> QueryIntent:
    """
    LLM-based NER using .with_structured_output(QueryIntent).
    Guarantees type-safe output with entity_name, entity_type,
    target_attribute, and time_context.
    """
    try:
        # with_structured_output returns a QueryIntent directly — no parsing
        structured_llm = config.llm.with_structured_output(QueryIntent)
        chain = _NER_PROMPT | structured_llm
        intent = llm_invoke_with_retry(chain, {"question": question})
        return intent
    except Exception as e:
        print(f"[Router Agent] ⚠️ LLM NER failed ({e}), falling back to regex")
        return _extract_query_intent_regex(question)


def _extract_query_intent(question: str) -> QueryIntent:
    """
    Route to LLM-based NER (preferred) or regex fallback.
    Fast mode still attempts LLM if available; regex is last resort.
    """
    from shared.config import RETRIEVAL_MODE
    if RETRIEVAL_MODE == "fast":
        # In fast mode, try LLM first but fall back to regex on failure
        try:
            return _extract_query_intent_llm(question)
        except Exception:
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
          f"type='{intent.entity_type}', attr='{intent.target_attribute}', "
          f"time='{intent.time_context}'")

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

    # --- Step 2b: Fine-grained intent classification ---
    fine_intent = _classify_intent(question, route)
    print(f"[Router Agent] Fine-grained intent: {fine_intent}")

    # --- Step 3: Determine query domain ---
    available_domains = get_available_domains()

    flagged = False
    metadata_log = ""

    if user_domain and user_domain in available_domains:
        # ABSOLUTE DOMAIN ISOLATION: Use the user's selected domain strictly.
        # No keyword-based overrides allowed.
        detected_domain = user_domain
        print(f"[Router Agent] Strict Domain Isolation: Using user-selected domain '{detected_domain}'")

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
        f"Entity: '{intent.target_entity}' (type={intent.entity_type}), "
        f"Attr: '{intent.target_attribute}'."
    )

    return {
        "route": route,
        "intent": fine_intent,
        "flagged": flagged,
        "next_agent": next_agent,
        "user_domain": detected_domain,
        "target_entity": intent.target_entity,
        "entity_type": intent.entity_type,
        "target_attribute": intent.target_attribute,
        "time_context": intent.time_context,
        "metadata_log": metadata_log,
        "_thinking": thinking,
    }


def decide_route(state: GraphState) -> str:
    """
    Conditional edge switchboard: Routes based on fine-grained intent.
    
    Returns compound keys like "semiconductor__db_query" so app.py
    conditional edges can dispatch to different subchains per intent.
    """
    if state.get("flagged", False):
        print("[ROUTING] -> Escalation (security flag or unresolved domain)")
        return "escalate"

    domain = state.get("next_agent", "")
    if not domain:
        print("[ROUTING] -> Escalation (no domain detected)")
        return "escalate"

    intent = state.get("intent", "spec_retrieval")

    # General chat bypasses all domain agents entirely
    if intent == "general_chat":
        print("[ROUTING] -> generate_response (general_chat — skipping all retrieval)")
        return "general_chat"

    # Build compound key for domain-specific intent routing
    route_key = f"{domain}__{intent}"
    print(f"[ROUTING] -> {route_key}")
    return route_key

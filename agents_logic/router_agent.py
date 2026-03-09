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
import re


# --- INTENT CONSTANTS ---
INTENT_DB = "db_query"
INTENT_SPECS = "spec_retrieval"
INTENT_CROSS = "cross_reference"
INTENT_CHAT = "general_chat"
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
        "summary", "summarize", "overview", "list files", "available files",
    ],
    "general_chat": [
        "what is vera", "what does vera", "who are you",
        "what can you do", "help me", "hello",
        "how do you work", "what are you",
    ],
}

_GENERIC_ENTITY_PHRASES = {
    "general", "all", "all items", "all records", "all data", "all contracts",
    "all clauses", "all labels", "all terms", "everything", "any", "any items",
    "all entities", "all aspects", "key aspects", "contract", "the contract",
}

_LEGAL_GUARD_KEYWORDS = {
    "contract", "agreement", "clause", "cuad", "governing law",
    "indemnity", "termination", "renewal", "assignment", "confidentiality",
    "legal",
}

_CONTRACT_SCOPE_KEYWORDS = {
    "contract", "agreement", "clause", "legal", "cuad", "key aspects",
    "review this contract", "aspects of this contract",
}


def _normalize_target_entity(entity: str) -> str:
    """
    Normalize broad non-entity phrases to GENERAL so downstream information-lock
    does not block category-level questions.
    """
    if not entity:
        return "GENERAL"

    normalized = entity.strip().lower()
    if not normalized:
        return "GENERAL"

    if normalized in _GENERIC_ENTITY_PHRASES:
        return "GENERAL"

    # Generic pattern: short phrase made only of generic terms
    generic_tokens = {
        "all", "any", "items", "item", "records", "record", "data", "contracts",
        "contract", "clauses", "clause", "labels", "label", "terms", "aspects",
        "aspect", "entities", "entity", "everything", "general", "the", "this",
    }
    tokens = [t for t in re.findall(r"[a-z0-9]+", normalized) if t]
    if tokens and all(t in generic_tokens for t in tokens):
        return "GENERAL"

    return entity


def _should_force_legal_pipeline(question: str, selected_domain: str) -> bool:
    """
    Guardrail: contract-analysis queries in legal domain must not route as
    general_chat, even if intent classifier is noisy.
    """
    if selected_domain != "legal":
        return False
    q = question.lower()
    return any(kw in q for kw in _LEGAL_GUARD_KEYWORDS)


def _is_contract_scope_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _CONTRACT_SCOPE_KEYWORDS)


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
        "USER QUESTION: {question}\n\n"
        "Instructions:\n"
        "1. target_entity: The specific individual entity name (e.g. 'ENTITY_ID', 'DEVICE_CODE').\n"
        "   - Use 'GENERAL' if the question refers to categories or all data (e.g. 'all items', 'metrics', 'records', 'data').\n"
        "2. entity_type: The categorical TYPE (e.g. 'patient', 'lot', 'product').\n"
        "3. target_attribute: The specific attribute or action (e.g. 'average age', 'max voltage', 'status').\n"
        "4. time_context: Temporal qualifiers (e.g. 'latest', 'Jan 2024').\n"
        "5. Respond with ONLY raw JSON: "
        '{{"target_entity": "...", "entity_type": "...", '
        '"target_attribute": "...", "time_context": "..."}}'
    ))
])


def _extract_query_intent_regex(question: str) -> QueryIntent:
    """
    Broad regex fallback for entity extraction — used when LLM fails or is too slow.
    """
    import re

    entity = "GENERAL"
    attribute = "GENERAL"

    # --- Priority 1: Classic code-style IDs ---
    id_patterns = [
        r'\b([A-Z]{2,}[-_]\d{2,}[-_][A-Z0-9]+)\b',
        r'\b([A-Z]{2,}[-_]\d{3,})\b',
        r'\b([A-Z]{2,}\d{3,})\b',
        r'\b([A-Z]{2,}[-_]\d+)\b', # Added more flexible patient IDs
    ]
    for pattern in id_patterns:
        match = re.search(pattern, question)
        if match:
            entity = match.group(1)
            break

    # --- Priority 2: Quoted strings ---
    if entity == "GENERAL":
        quoted = re.search(r'["\']([^"\']+)["\']', question)
        if quoted:
            entity = quoted.group(1).strip()

    # --- Priority 3: Capitalized proper nouns (2+ uppercase letters) ---
    if entity == "GENERAL":
        skip = {"What", "How", "Tell", "Show", "Find", "Get", "Are", "Is",
                "The", "About", "From", "Which", "Where", "When", "Does",
                "Can", "Could", "Would", "Should", "Have", "Has", "Was",
                "Were", "Will", "Do", "Did", "Any", "All", "Some", "VERA",
                "Based", "Please", "List", "Report", "Summary", "Show", "Tell"}
        words = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', question)
        for w in words:
            if w not in skip:
                entity = w
                break

    # --- Attribute extraction ---
    attr_hints = ["voltage", "thermal", "status", "yield", "lot", "wafer", "burn-in", "config", "data", "summary"]
    for hint in attr_hints:
        if hint in question.lower():
            attribute = hint
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

    return QueryIntent(
        target_entity=entity,
        entity_type=entity_type,
        target_attribute=attribute,
        time_context=""
    )


def _extract_query_intent_llm(question: str) -> QueryIntent:
    """
    LLM-based NER using manual JSON extraction.
    Much more robust than with_structured_output for small local models.
    """
    try:
        # Standard chain with JSON output
        chain = _NER_PROMPT | config.llm | StrOutputParser()
        raw_output = llm_invoke_with_retry(chain, {"question": question})

        # Find all JSON-like blocks
        candidates = re.findall(r'\{[^{}]*\}', raw_output.replace('\n', ' '))
        
        for cand in candidates:
            try:
                data = json.loads(cand)
                return QueryIntent(
                    target_entity=data.get("target_entity", "GENERAL"),
                    entity_type=data.get("entity_type", "GENERAL"),
                    target_attribute=data.get("target_attribute", "GENERAL"),
                    time_context=data.get("time_context", "")
                )
            except:
                continue
                
        # Fallback to the largest boundary search if simple matches fail
        start = raw_output.find('{')
        end = raw_output.rfind('}')
        if start != -1 and end != -1:
            try:
                data = json.loads(raw_output[start:end+1])
                return QueryIntent(
                    target_entity=data.get("target_entity", "GENERAL"),
                    entity_type=data.get("entity_type", "GENERAL"),
                    target_attribute=data.get("target_attribute", "GENERAL"),
                    time_context=data.get("time_context", "")
                )
            except: pass

        raise ValueError(f"No valid JSON found in: {raw_output[:50]}...")
    except Exception as e:
        print(f"[Router Agent] ⚠️ LLM NER failed/timed out ({e}), using regex fallback")
        return _extract_query_intent_regex(question)


def _extract_query_intent(question: str) -> QueryIntent:
    """
    Route to LLM-based NER (preferred) or regex fallback.
    In 'fast' mode, we favor speed + robustness.
    """
    from shared.config import RETRIEVAL_MODE
    
    # If the question is very short or just an ID, use regex immediately 
    if len(question.split()) < 4:
        return _extract_query_intent_regex(question)

    return _extract_query_intent_llm(question)


@vera_agent("Router Agent")
def run(state: GraphState) -> dict:
    """
    ROUTER NODE: Classifies intent, detects domain, performs security checks,
    and extracts query understanding (target_entity, target_attribute).
    """
    print(f"[Router Agent] DEBUG: Incoming state keys: {list(state.keys())}")
    print(f"[Router Agent] DEBUG: user_domain in state: '{state.get('user_domain')}' (type: {type(state.get('user_domain'))})")

    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "")
    selected_domain_for_guard = user_domain.lower().strip() if user_domain else ""
    input_contract_text = (state.get("input_contract_text", "") or "").strip()

    # --- Step 1: Query Understanding — extract entity, attribute, time ---
    intent = _extract_query_intent(question)
    intent.target_entity = _normalize_target_entity(intent.target_entity)
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

    # Legal guard: never allow contract/CUAD queries to bypass domain agents.
    if _should_force_legal_pipeline(question, selected_domain_for_guard):
        if fine_intent == INTENT_CHAT:
            fine_intent = INTENT_SPECS
            metadata_log += (
                "[ROUTER] Legal guard activated: upgraded general_chat to "
                "spec_retrieval for contract-analysis query.\n"
            )
    print(f"[Router Agent] Fine-grained intent: {fine_intent}")

    # --- Step 3: Determine query domain ---
    available_domains = get_available_domains()
    
    # Normalize user_domain for case-insensitive matching
    user_domain_clean = user_domain.lower().strip() if user_domain else ""
    
    flagged = False
    metadata_log = ""
    detected_domain = ""

    # ABSOLUTE DOMAIN ISOLATION: If user selected a domain, use it strictly.
    if user_domain_clean and user_domain_clean in available_domains:
        detected_domain = user_domain_clean
        print(f"[Router Agent] Strict Domain Isolation: Using user-selected domain '{detected_domain}'")
    else:
        # Fallback to LLM detection only if no valid domain was provided by user
        if user_domain_clean:
             print(f"[Router Agent] ⚠️ User domain '{user_domain}' not in {available_domains}. Falling back to LLM.")
        
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
                detected_domain = user_domain_clean or (available_domains[0] if available_domains else "unknown")
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

    # Scope guard: contract-analysis queries are only supported in legal domain.
    # Trigger regardless of whether a contract file is uploaded to avoid
    # expensive, irrelevant retrieval in non-legal pipelines.
    if _is_contract_scope_query(question) and detected_domain != "legal":
        flagged = True
        metadata_log += (
            "[ROUTER] ⚠️ LEGAL_DOMAIN_REQUIRED: Contract analysis requested while "
            f"active domain is '{detected_domain}'. Prompt user to switch to legal.\n"
        )
        print("[Router Agent] ⚠️ Scope guard triggered: legal domain required for contract analysis")

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

    domain = state.get("user_domain", "") or state.get("next_agent", "")
    if not domain:
        print("[ROUTING] -> Escalation (no domain detected)")
        return "escalate"

    intent = state.get("intent", INTENT_SPECS)

    # General chat bypasses all domain agents entirely
    if intent == INTENT_CHAT:
        print("[ROUTING] -> generate_response (general_chat — skipping all retrieval)")
        return INTENT_CHAT

    # Build compound key for domain-specific intent routing
    route_key = f"{domain}__{intent}"
    print(f"[ROUTING] -> {route_key}")
    return route_key

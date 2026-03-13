"""
================================================================================
Pickleball Domain Configuration
================================================================================
Declares routing heuristics, keywords, aliases, and metadata schema for the
pickleball domain. Loaded dynamically by the shared router — no generic
agent should ever hardcode these values.
================================================================================
"""

DOMAIN_CONFIG = {
    # Human-readable label
    "name": "pickleball",
    "description": (
        "Pickleball equipment specifications, rulebooks, standards compliance, "
        "and performance data analysis."
    ),

    # ------------------------------------------------------------------
    # Routing keywords — used by the router to score intent categories.
    # Each key is an intent label; its value is a list of representative
    # keywords the router will match against the user's query.
    # ------------------------------------------------------------------
    "keywords": {
        "technical": [
            "paddle", "equipment", "spec", "specification", "standard",
            "manual", "datasheet", "weight", "length", "material",
            "thickness", "diameter", "performance", "rating", "tolerance",
            "upaa", "usap", "equipment type", "model", "measurement",
            "dimension", "database", "db", "performance data",
        ],
        "compliance": [
            "rule", "rulebook", "regulation", "standard", "approval",
            "certification", "compliance", "procedure", "policy",
            "rule change", "audit", "email", "memo", "decision",
            "record", "tracking", "report", "amendment",
        ],
        "db_query": [
            "database", "db", "performance data", "check the database",
            "sql", "records", "statistics", "results", "player data",
        ],
        "spec_retrieval": [
            "specification", "spec", "datasheet", "standard", "manual",
            "equipment standard", "paddle spec", "rating", "requirement",
            "measurement", "dimension", "material",
        ],
        "cross_reference": [
            "compare", "discrepancy", "mismatch", "conflict",
            "cross-reference", "check against", "inconsistent", "verify",
            "added", "newly", "difference", "comparison", "new", "versus",
        ],
    },

    # Domain aliases — alternative names the LLM might produce when
    # classifying domain.  The router resolves these to "pickleball".
    "aliases": ["paddle", "equipment standards", "racket sports", "sport"],

    # ------------------------------------------------------------------
    # Metadata schema — tells the Advanced RAG query-understanding step
    # which structured filters can be extracted from a user query.
    # ------------------------------------------------------------------
    "metadata_schema": {
        "entity_types": ["equipment_id", "paddle_id", "player_id", "tournament_id"],
        "doc_versions": True,
        "access_levels": ["public", "internal_only", "confidential"],
    },

    # Short keyword hints used by the LLM domain classifier prompt
    "keyword_hints": (
        "paddle, equipment, spec, standard, rulebook, UPAA, USAP, performance"
    ),
}

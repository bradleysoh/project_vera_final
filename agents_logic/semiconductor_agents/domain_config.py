"""
================================================================================
Semiconductor Domain Configuration
================================================================================
Declares routing heuristics, keywords, aliases, and metadata schema for the
semiconductor domain. Loaded dynamically by the shared router — no generic
agent should ever hardcode these values.
================================================================================
"""

DOMAIN_CONFIG = {
    # Human-readable label
    "name": "semiconductor",
    "description": (
        "Semiconductor manufacturing, process specifications, device "
        "characterization, and production quality control."
    ),

    # ------------------------------------------------------------------
    # Routing keywords — used by the router to score intent categories.
    # Each key is an intent label; its value is a list of representative
    # keywords the router will match against the user's query.
    # ------------------------------------------------------------------
    "keywords": {
        "technical": [
            "voltage", "thermal", "datasheet", "spec", "limit", "current",
            "power", "temperature", "watt", "rtx", "chip", "die", "wafer",
            "burn-in", "silicon", "yield", "performance", "frequency", "clock",
            "maximum", "minimum", "rating", "tolerance",
            "production", "lot", "batch", "database", "db",
        ],
        "compliance": [
            "sop", "audit", "email", "procedure", "quality", "regulation",
            "compliance", "checklist", "process change", "internal decision",
            "waiver", "approval", "sign-off", "certification",
            "record", "tracking", "inventory", "report",
        ],
    },

    # Domain aliases — alternative names the LLM might produce when
    # classifying domain.  The router resolves these to "semiconductor".
    "aliases": ["engineering", "chip", "fab", "chips", "electronics"],

    # ------------------------------------------------------------------
    # Metadata schema — tells the Advanced RAG query-understanding step
    # which structured filters can be extracted from a user query.
    # ------------------------------------------------------------------
    "metadata_schema": {
        "entity_types": ["product_id", "lot_number", "equipment_id"],
        "doc_versions": True,
        "access_levels": ["public", "internal_only", "confidential"],
    },

    # Short keyword hints used by the LLM domain classifier prompt
    "keyword_hints": (
        "chips, voltage, thermal, RTX, silicon, wafer, die, yield, burn-in"
    ),
}

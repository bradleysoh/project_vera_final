"""
================================================================================
Energy Domain Configuration
================================================================================
"""

DOMAIN_CONFIG = {
    "name": "energy",
    "description": "Power generation audits, grid compliance, and energy plant operations.",

    "keywords": {
        "db_query": [
            "plant", "generator", "turbine", "output", "capacity",
            "emission", "audit", "power", "data", "records", "history",
        ],
        "spec_retrieval": [
            "specification", "limit", "threshold", "regulation", "compliance",
            "sop", "procedure", "safety", "calibration",
        ],
        "cross_reference": [
            "compare", "discrepancy", "mismatch", "conflict",
            "cross-reference", "check against", "inconsistent",
        ],
    },

    "aliases": ["power", "grid", "electricity", "utility"],

    "keyword_hints": "plant, generator, turbine, emission, power output, compliance, audit, grid",

    "metadata_schema": {
        "entity_types": ["plant_id", "generator_id", "unit_id", "region"],
        "doc_versions": False,
        "access_levels": ["public", "internal_only", "confidential"],
    },

    "generic_entities": [
        "plant", "plants", "generator", "generators", "turbine", "turbines",
        "data", "records", "record", "conflicts", "discrepancy", "mismatch",
        "general", "potential", "rule", "rulebook", "manual", "guide", "policy",
        "actions", "update", "procedure", "procedures", "audit", "audits"
    ],
}

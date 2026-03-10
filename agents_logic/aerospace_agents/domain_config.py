"""
================================================================================
Aerospace Domain Configuration
================================================================================
"""

DOMAIN_CONFIG = {
    "name": "aerospace",
    "description": "Aerospace predictive maintenance, engine telemetry, and component lifecycle.",

    "keywords": {
        "db_query": [
            "engine", "sensor", "telemetry", "unit", "cycle", "rul",
            "remaining useful life", "maintenance", "failure", "data",
            "records", "history", "lookup",
        ],
        "spec_retrieval": [
            "specification", "limit", "threshold", "tolerance", "manual",
            "procedure", "sop", "calibration", "safety",
        ],
        "cross_reference": [
            "compare", "discrepancy", "mismatch", "conflict",
            "cross-reference", "check against", "inconsistent",
        ],
    },

    "aliases": ["aviation", "aircraft", "engine", "flight"],

    "keyword_hints": "engine, sensor, telemetry, RUL, maintenance, failure, cycle, turbine, predictive",

    "metadata_schema": {
        "entity_types": ["engine_id", "unit_id", "sensor_id", "component_id"],
        "doc_versions": False,
        "access_levels": ["public", "internal_only", "confidential"],
    },
}

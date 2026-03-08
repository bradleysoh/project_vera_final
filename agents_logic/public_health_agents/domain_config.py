# agents_logic/public_health_agents/domain_config.py

DOMAIN_CONFIG = {
    "name": "public_health",
    "alias": ["medical", "clinical", "tb", "tuberculosis", "healthcare"],
    "description": "TB prevention, clinical guidelines, and patient screening audits.",
    "keyword_hints": [
        "tuberculosis", "tb", "who", "patient", "screening", 
        "clinical", "treatment", "dosage", "rifampicin", "isoniazid",
        "healthcare", "clinic", "medical"
    ],
    "authority_hierarchy": ["db", "official", "informal"]
}
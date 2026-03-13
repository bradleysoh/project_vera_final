"""
Legal domain configuration.
"""

DOMAIN_CONFIG = {
    "name": "legal",
    "description": (
        "Legal contract analysis, clause extraction, and discrepancy checks "
        "against CUAD-labeled contract data."
    ),
    "keywords": {
        "technical": [
            "contract", "agreement", "clause", "term", "obligation",
            "governing law", "termination", "liability", "indemnity",
            "renewal", "assignment", "confidentiality", "exclusivity",
            "legal", "cuad",
        ],
        "compliance": [
            "compliance", "dispute", "breach", "audit", "review",
            "risk", "notice", "jurisdiction", "penalty", "default",
        ],
        "db_query": [
            "input contract", "uploaded contract", "contract text",
            "extract clauses", "key aspects", "what clauses",
        ],
        "spec_retrieval": [
            "cuad", "labeled data", "standard clauses", "reference contracts",
            "benchmark", "common clauses",
        ],
        "cross_reference": [
            "compare", "discrepancy", "mismatch", "conflict",
            "missing clause", "against cuad", "difference",
        ],
    },
    "aliases": ["law", "contracts", "contract", "legal-review"],
    "keyword_hints": (
        "contract, clause, indemnity, termination, governing law, renewal, "
        "assignment, confidentiality, discrepancy, CUAD"
    ),
    "metadata_schema": {
        "entity_types": ["contract_title", "clause_label"],
        "doc_versions": True,
        "access_levels": ["public", "internal_only", "confidential"],
    },
    "generic_entities": [
        "contract", "contracts", "clause", "clauses", "agreement", "agreements",
        "data", "records", "record", "conflicts", "discrepancy", "mismatch",
        "general", "potential", "rule", "rulebook", "manual", "guide", "policy",
        "actions", "update", "cuad", "labeled"
    ],
}


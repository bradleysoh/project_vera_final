"""
================================================================================
Finance Domain Configuration
================================================================================
"""

DOMAIN_CONFIG = {
    "name": "finance",
    "description": "Financial auditing, compliance, trial balances, and regulatory reporting.",

    "keywords": {
        "db_query": [
            "audit", "balance", "transaction", "account", "ledger",
            "revenue", "expense", "trial", "data", "records", "history",
        ],
        "spec_retrieval": [
            "regulation", "compliance", "standard", "gaap", "ifrs",
            "policy", "procedure", "sop", "guideline",
        ],
        "cross_reference": [
            "compare", "discrepancy", "mismatch", "conflict",
            "cross-reference", "check against", "inconsistent",
        ],
    },

    "aliases": ["financial", "accounting", "banking", "audit"],

    "keyword_hints": "audit, balance, transaction, revenue, expense, trial, compliance, GAAP, IFRS",

    "metadata_schema": {
        "entity_types": ["account_id", "transaction_id", "company_id", "fiscal_period"],
        "doc_versions": False,
        "access_levels": ["public", "internal_only", "confidential"],
    },
}

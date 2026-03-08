"""
================================================================================
Medical Domain Configuration
================================================================================
Example configuration for a healthcare/medical device audit domain.
================================================================================
"""

DOMAIN_CONFIG = {
    "name": "medical",
    "description": "Healthcare, medical device maintenance, and patient record audits.",
    
    "keywords": {
        "db_query": [
            "patient", "record", "doctor", "hospital", "clinic", "dosage",
            "medication", "prescription", "appointment", "history", "billing"
        ],
        "spec_retrieval": [
            "maintenance", "calibration", "device", "manual", "sop", "mri",
            "ct scan", "x-ray", "calibration date", "safety check", "procedure"
        ],
        "cross_reference": [
            "compare", "discrepancy", "mismatch", "conflict",
            "cross-reference", "check against", "inconsistent",
        ],
    },

    "aliases": ["healthcare", "hospital", "clinical", "med"],

    "keyword_hints": "patient, record, dosage, MRI, CT, calibration, safety, doctor, maintenance, sop, manual, procedure, device, hospital, clinical",

    "normalization_rules": [
        "Dosage → mg or mL",
        "Temperature → Celsius (°C)",
        "Safety Rating → 1-10 Scale"
    ],

    "example_queries": [
        "What is the maintenance history for MRI-Unit-4?",
        "Check patient records for dosage discrepancies",
        "What is the calibration SOP for CT-Scanner-A?"
    ]
}

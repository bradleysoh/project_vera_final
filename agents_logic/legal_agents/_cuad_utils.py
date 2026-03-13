"""
Shared CUAD utilities for legal agents.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache


_STOP_WORDS = {
    "of", "to", "for", "and", "or", "the", "a", "an", "in", "on",
    "by", "with", "from", "any", "is", "are", "be",
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _cuad_path() -> str:
    return os.path.join(_project_root(), "source_documents", "legal", "CUAD_v1.json")


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_EXPR = (
    rf"((?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+\d{{4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+day\s+of\s+(?:{_MONTHS}),?\s+\d{{4}}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
)


def _clean_value(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,:;")


def extract_agreement_date(contract_text: str) -> str:
    """
    Extract agreement/signing date from uploaded contract text.
    """
    if not contract_text:
        return ""

    text = contract_text

    contextual_patterns = [
        rf"(?:agreement\s+date|date\s+of\s+this\s+agreement|dated\s+as\s+of)\s*[:\-]?\s*{_DATE_EXPR}",
        rf"(?:made\s+and\s+entered\s+into\s+as\s+of|effective\s+as\s+of|made\s+on)\s*[:\-]?\s*{_DATE_EXPR}",
        rf"this\s+agreement.*?\b(?:dated|date|on)\b.*?{_DATE_EXPR}",
    ]

    for pat in contextual_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            # Last capturing group is the date value
            return _clean_value(m.groups()[-1])

    # Fallback: first date in heading area
    header = text[:2000]
    m = re.search(_DATE_EXPR, header, flags=re.IGNORECASE)
    if m:
        return _clean_value(m.group(1))

    return ""


def extract_key_contract_fields(contract_text: str) -> dict[str, str]:
    """
    Deterministically extract core fields from uploaded contract text.
    """
    fields: dict[str, str] = {}
    agreement_date = extract_agreement_date(contract_text)
    if agreement_date:
        fields["agreement_date"] = agreement_date
    return fields


def _label_tokens(label: str) -> list[str]:
    tokens = [t for t in _normalize(label).split() if t and t not in _STOP_WORDS]
    return tokens


@lru_cache(maxsize=1)
def load_cuad_statistics() -> dict:
    """
    Load CUAD once and compute per-label prevalence statistics.
    """
    path = _cuad_path()
    if not os.path.exists(path):
        return {"total_contracts": 0, "prevalence": {}, "labels": []}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        payload = json.load(f)

    contracts = payload.get("data", [])
    total = len(contracts)
    if total == 0:
        return {"total_contracts": 0, "prevalence": {}, "labels": []}

    counts = Counter()
    labels_seen = set()

    for contract in contracts:
        present_in_contract = set()
        for paragraph in contract.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id", "")
                label = qa_id.split("__", 1)[1] if "__" in qa_id else "Unknown"
                labels_seen.add(label)
                if not qa.get("is_impossible", False):
                    answers = qa.get("answers") or []
                    if any((a.get("text", "") or "").strip() for a in answers):
                        present_in_contract.add(label)
        for label in present_in_contract:
            counts[label] += 1

    prevalence = {label: counts[label] / total for label in labels_seen}
    return {
        "total_contracts": total,
        "prevalence": prevalence,
        "labels": sorted(labels_seen),
    }


def detect_clauses_in_contract(contract_text: str) -> dict[str, float]:
    """
    Heuristic clause detection: returns {label: score} for labels detected in text.
    """
    stats = load_cuad_statistics()
    text = _normalize(contract_text)
    if not text:
        return {}

    scores: dict[str, float] = {}
    for label in stats["labels"]:
        tokens = _label_tokens(label)
        if not tokens:
            continue

        hit_count = sum(1 for tok in tokens if f" {tok} " in f" {text} ")
        if hit_count == 0:
            continue

        required = 2 if len(tokens) >= 3 else 1
        if hit_count < required:
            continue

        score = hit_count / max(len(tokens), 1)
        scores[label] = round(score, 3)

    return scores


def summarize_contract_key_aspects(contract_text: str, max_items: int = 12) -> list[str]:
    """
    Return top detected CUAD labels from input contract text.
    """
    detected = detect_clauses_in_contract(contract_text)
    ranked = sorted(detected.items(), key=lambda x: x[1], reverse=True)
    return [label for label, _ in ranked[:max_items]]


def benchmark_discrepancies(contract_text: str) -> dict:
    """
    Compare detected clauses in input contract against CUAD prevalence.

    Returns:
      - detected_labels
      - missing_common_labels (common in CUAD but not detected)
      - uncommon_present_labels (detected but rare in CUAD)
    """
    stats = load_cuad_statistics()
    prevalence = stats["prevalence"]
    detected_scores = detect_clauses_in_contract(contract_text)
    detected_labels = set(detected_scores.keys())

    missing_common = []
    uncommon_present = []

    for label, p in prevalence.items():
        if p >= 0.6 and label not in detected_labels:
            missing_common.append((label, p))

    for label in detected_labels:
        p = prevalence.get(label, 0.0)
        if p <= 0.1:
            uncommon_present.append((label, p))

    missing_common.sort(key=lambda x: x[1], reverse=True)
    uncommon_present.sort(key=lambda x: x[1])

    return {
        "detected_labels": sorted(
            detected_labels,
            key=lambda lbl: detected_scores.get(lbl, 0.0),
            reverse=True,
        ),
        "detected_scores": detected_scores,
        "missing_common_labels": missing_common,
        "uncommon_present_labels": uncommon_present,
        "total_cuad_contracts": stats["total_contracts"],
    }

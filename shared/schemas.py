"""
================================================================================
Project VERA — Pydantic Schemas (Data Contracts)
================================================================================

Defines the structured data contracts between agents.  These schemas enforce
the "Structured Fact Passing" architecture: no raw `Document` objects transit
downstream of retrieval agents.

Contracts:
    QueryIntent       — Router → all downstream agents
    ExtractedFact     — Retrieval agents → Discrepancy Agent / Response Agent
    FactCollection    — Batch container for structured extraction via LLM
    DiscrepancyVerdict — Discrepancy Agent → Response Agent

Serialization:
    GraphState is a TypedDict and cannot hold Pydantic objects directly.
    Use `.model_dump()` to serialize into state and `Model(**d)` to restore.
================================================================================
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query Understanding — Router output
# ---------------------------------------------------------------------------

class QueryIntent(BaseModel):
    """
    Structured decomposition of the user's query.
    Extracted by the Router Agent and passed through GraphState to all
    downstream agents for precision retrieval and focused extraction.
    """
    target_entity: str = Field(
        default="GENERAL",
        description="The primary entity mentioned (product ID, patient ID, "
                    "lot number).  'GENERAL' if no specific entity.",
    )
    target_attribute: str = Field(
        default="GENERAL",
        description="The specific attribute being asked about (e.g., "
                    "'max_voltage', 'dosage', 'burn_in_duration').  "
                    "'GENERAL' if not attribute-specific.",
    )
    time_context: str = Field(
        default="",
        description="Temporal qualifier — 'latest', '2024-Q3', a specific "
                    "date, or empty string if none mentioned.",
    )


# ---------------------------------------------------------------------------
# Extracted Fact — Retrieval agent output
# ---------------------------------------------------------------------------

class ExtractedFact(BaseModel):
    """
    A single, isolated fact distilled from a raw document chunk.

    Token footprint: ~30 tokens per fact (vs ~500 for a raw chunk).
    This is the fundamental unit of the "Structured Fact Passing" contract.
    """
    entity: str = Field(
        description="The entity this fact pertains to (e.g. 'RTX-9000').",
    )
    attribute: str = Field(
        description="The attribute being described (e.g. 'max_voltage').",
    )
    value: str = Field(
        description="The extracted value (e.g. '5.0V', '150°C', 'approved').",
    )
    source_type: str = Field(
        description="Document source type: 'datasheet', 'sop', 'email', "
                    "'memo', 'db', 'spec', etc.",
    )
    source_doc: str = Field(
        default="unknown",
        description="Source filename or document identifier.",
    )
    date: str = Field(
        default="unknown",
        description="Timestamp or version date (ISO format preferred).",
    )
    confidence: str = Field(
        default="MEDIUM",
        description="Extraction confidence: HIGH / MEDIUM / LOW.",
    )


class FactCollection(BaseModel):
    """
    Batch container for LLM-based structured extraction.
    Used with `llm.with_structured_output(FactCollection)`.
    """
    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="List of extracted facts from retrieved documents.",
    )


# ---------------------------------------------------------------------------
# Discrepancy Verdict — Discrepancy agent output
# ---------------------------------------------------------------------------

class ConflictStatus(str, Enum):
    """Status of a single entity-attribute comparison."""
    ALIGNED = "ALIGNED"
    DISCREPANCY = "DISCREPANCY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AttributeConflict(BaseModel):
    """A single conflict for one entity-attribute pair."""
    entity: str
    attribute: str
    status: ConflictStatus
    authoritative_value: str = Field(
        default="",
        description="The value from the highest-priority source.",
    )
    authoritative_source: str = Field(
        default="",
        description="Source type of the authoritative value (db/official/informal).",
    )
    authoritative_date: str = Field(
        default="",
        description="Date/version of the authoritative source.",
    )
    conflicting_values: List[dict] = Field(
        default_factory=list,
        description="List of {'value': ..., 'source': ..., 'date': ..., 'reason': ...} "
                    "for each conflicting entry.",
    )


class DiscrepancyVerdict(BaseModel):
    """
    Structured output from the deterministic discrepancy agent.
    No LLM involvement — pure hierarchical logic.
    """
    target_entity: str = Field(
        default="GENERAL",
        description="The entity that was audited.",
    )
    overall_status: ConflictStatus = Field(
        default=ConflictStatus.ALIGNED,
        description="Overall status across all attribute comparisons.",
    )
    conflicts: List[AttributeConflict] = Field(
        default_factory=list,
        description="Per-attribute conflict details.",
    )
    audit_summary: str = Field(
        default="",
        description="One-sentence summary of the audit conclusion.",
    )

    def has_discrepancy(self) -> bool:
        return self.overall_status == ConflictStatus.DISCREPANCY

    def to_report_string(self) -> str:
        """Serialize to a human-readable audit report string."""
        lines = [
            f"**AUDIT TARGET**: {self.target_entity}",
            f"**STATUS**: {self.overall_status.value}",
            "",
        ]
        for i, c in enumerate(self.conflicts, 1):
            lines.append(f"**{i}. {c.attribute}** ({c.status.value})")
            if c.authoritative_value:
                lines.append(
                    f"   Authoritative: {c.authoritative_value} "
                    f"(Source: {c.authoritative_source}, Date: {c.authoritative_date})"
                )
            for cv in c.conflicting_values:
                lines.append(
                    f"   Conflict: {cv.get('value', '?')} "
                    f"(Source: {cv.get('source', '?')}, "
                    f"Reason: {cv.get('reason', 'lower authority')})"
                )
            lines.append("")

        if self.audit_summary:
            lines.append(f"**AUDIT CONCLUSION**: {self.audit_summary}")

        return "\n".join(lines)

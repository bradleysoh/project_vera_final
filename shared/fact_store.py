"""
================================================================================
Project VERA — Fact Store (Shift-Left Governance)
================================================================================
Handles relational storage of ExtractedFact objects in SQLite.
Links facts to their parent Document IDs for O(1) retrieval during RAG.
================================================================================
"""

import os
import sqlite3
import json
from typing import List, Optional
from shared.schemas import ExtractedFact

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FACT_DB_PATH = os.path.join(_PROJECT_ROOT, "fact_store.db")

class FactStore:
    def __init__(self, db_path: str = FACT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the facts table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                entity TEXT,
                attribute TEXT,
                value TEXT,
                source_type TEXT,
                source_doc TEXT,
                date TEXT,
                confidence TEXT,
                metadata_json TEXT
            )
        """)
        # Index on document_id for fast retrieval during RAG
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON extracted_facts(document_id)")
        # Index on entity for direct lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity ON extracted_facts(entity)")
        conn.commit()
        conn.close()

    def save_facts(self, document_id: str, facts: List[ExtractedFact]):
        """Persist a list of facts for a specific document."""
        self._init_db() # Ensure tables exist
        if not facts:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear old facts for this document to avoid duplicates on re-ingestion
        cursor.execute("DELETE FROM extracted_facts WHERE document_id = ?", (document_id,))
        
        for fact in facts:
            cursor.execute("""
                INSERT INTO extracted_facts 
                (document_id, entity, attribute, value, source_type, source_doc, date, confidence, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                fact.entity,
                fact.attribute,
                fact.value,
                fact.source_type,
                fact.source_doc,
                fact.date,
                fact.confidence,
                json.dumps(fact.model_dump())
            ))
        
        conn.commit()
        conn.close()

    def get_facts_by_doc_id(self, document_id: str) -> List[ExtractedFact]:
        """Retrieve all pre-extracted facts for a document."""
        self._init_db() # Ensure tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_json FROM extracted_facts WHERE document_id = ?", (document_id,))
        rows = cursor.fetchall()
        conn.close()
        
        facts = []
        for row in rows:
            try:
                facts.append(ExtractedFact(**json.loads(row[0])))
            except Exception:
                continue
        return facts

    def get_facts_by_entity(self, entity: str) -> List[ExtractedFact]:
        """Retrieve all facts across all documents for a specific entity."""
        self._init_db() # Ensure tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Case-insensitive partial match
        cursor.execute("SELECT metadata_json FROM extracted_facts WHERE entity LIKE ?", (f"%{entity}%",))
        rows = cursor.fetchall()
        conn.close()
        
        facts = []
        for row in rows:
            try:
                facts.append(ExtractedFact(**json.loads(row[0])))
            except Exception:
                continue
        return facts

# Singleton instance
store = FactStore()

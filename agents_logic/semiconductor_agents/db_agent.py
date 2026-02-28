"""
================================================================================
Database Agent — Domain-Generic NL-to-SQL Retrieval (Precision Mode)
================================================================================
DOMAIN: semiconductor (also reusable for any domain with .db files)
RESPONSIBILITY: Auto-discover SQLite databases, convert NL to PRECISION SQL,
                and provide authoritative structured results.

                Features:
                  - "Don't Give Up" Protocol: Always triggered for entities.
                  - Precision NL2SQL: Uses entity extraction for exact WHERE clauses.
                  - Schema-Aware: Injects schema context to prevent hallucinations.
================================================================================
"""

import re
import shared.config as config
from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.config import llm_invoke_with_retry
from shared.db_utils import (
    discover_databases,
    get_all_schemas,
    execute_read_only,
    format_results,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Schema-query detection keywords
# ---------------------------------------------------------------------------
_SCHEMA_KEYWORDS = [
    "what tables", "what data", "what information", "what can we search",
    "list tables", "show schema", "database structure", "what fields",
    "available data", "show tables", "describe database", "db schema",
]


def _is_schema_query(question: str) -> bool:
    """Check if the user is asking about the database structure itself."""
    q = question.lower()
    return any(kw in q for kw in _SCHEMA_KEYWORDS)


# ---------------------------------------------------------------------------
# Helper: Entity Extraction (Duplicated from Discrepancy Agent for independence)
# ---------------------------------------------------------------------------
_ENTITY_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "Extract the PRIMARY entity (product name, patient ID, lot number, "
        "component, or subject) from the user's question below.\n\n"
        "Return ONLY the entity name/ID — nothing else. If the question is "
        "general and has no specific entity, return: GENERAL_QUERY\n\n"
        "Question: {question}"
    ))
])


def _extract_entity(question: str) -> str:
    """Use the LLM to extract the primary entity from the user's question."""
    chain = _ENTITY_PROMPT | config.llm | StrOutputParser()
    raw = llm_invoke_with_retry(chain, {"question": question})
    entity = raw.strip().strip('"').strip("'")
    return entity


@vera_agent("Semiconductor DB Agent")
def run(state: GraphState) -> dict:
    """
    DB AGENT: Precision NL-to-SQL retrieval.

    Flow:
        1. Auto-discover .db files in source_documents/{user_domain}/
        2. If schema query → return table listing
        3. Extract Entity → if found, restrict SQL to that entity.
        4. LLM generates SQL (Schema-Aware + Entity-Specific).
        5. Execute SQL against all discovered DBs.
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "semiconductor")

    thinking_steps = []
    metadata_log = state.get("metadata_log", "")

    # --- Step 1: Discover databases for this domain ---
    db_paths = discover_databases(user_domain)

    if not db_paths:
        msg = f"No databases found for domain '{user_domain}'."
        thinking_steps.append(f"Inspecting source_documents/{user_domain}/... {msg}")
        print(f"[DB Agent] ⚠️ {msg}")
        return {
            "documents": state.get("documents", []),
            "metadata_log": metadata_log + f"[DB] {msg}\n",
            "db_result": "",
            "_thinking": " | ".join(thinking_steps),
        }

    db_names = [p.split("/")[-1] for p in db_paths]
    thinking_steps.append(
        f"Inspecting source_documents/{user_domain}/... "
        f"found databases: {db_names}"
    )
    print(f"[DB Agent] Found {len(db_paths)} database(s) for '{user_domain}': {db_names}")

    # --- Step 2: Get schemas for all databases ---
    all_schemas = get_all_schemas(user_domain)
    thinking_steps.append(
        f"Inspecting database schema... {all_schemas.split(chr(10))[0]}"
    )
    metadata_log += f"[DB] Schema for '{user_domain}':\n{all_schemas}\n"

    # --- Step 2b: Schema query shortcut ---
    if _is_schema_query(question):
        thinking_steps.append("User asked about database structure — returning schema listing.")
        return {
            "documents": state.get("documents", []),
            "metadata_log": metadata_log,
            "retrieved_docs": state.get("retrieved_docs") or {},
            "db_result": f"Available database structure:\n{all_schemas}",
            "_thinking": " | ".join(thinking_steps),
        }

    # --- Step 3: Entity — use router's target_entity (no extra LLM call) ---
    entity = state.get("target_entity", "").strip()
    if not entity or entity == "GENERAL":
        # Fallback: regex extraction from question
        import re as _re
        for pattern in [
            r'\b([A-Z]{2,}[-_]\d{2,}[-_][A-Z0-9]+)\b',
            r'\b([A-Z]{2,}[-_]\d{3,})\b',
            r'\b([A-Z]{2,}\d{3,})\b',
        ]:
            match = _re.search(pattern, question, _re.IGNORECASE)
            if match:
                entity = match.group(1)
                break
        else:
            entity = "GENERAL_QUERY"
    is_general = entity == "GENERAL_QUERY"

    thinking_steps.append(f"Analyzing query context... Target Entity: {entity}")
    print(f"[DB Agent] 🎯 Target Entity: {entity}")

    # Construct specific instructions based on entity presence
    entity_instruction = ""
    if not is_general:
        entity_instruction = (
            f"IMPORTANT: The user is asking specifically about '{entity}'.\n"
            f"Your SQL MUST include a WHERE clause to filter for '{entity}'.\n"
            f"Example: WHERE product_id LIKE '%{entity}%' OR id = '{entity}'\n"
            f"DO NOT return data for other entities (like RTX-9000 if asking for RTX-8000).\n"
        )

    thinking_steps.append(f"Generating SQL for: '{question[:60]}'...")

    # --- Fast mode: deterministic SQL (no LLM) ---
    from shared.config import RETRIEVAL_MODE

    if RETRIEVAL_MODE == "fast" and not is_general:
        # Build SQL deterministically by scanning all text columns for entity
        sql_queries = []
        for db_path in db_paths:
            db_name = db_path.split("/")[-1]
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    text_cols = [col[1] for col in columns if col[2].upper() in ("TEXT", "VARCHAR", "CHAR", "")]
                    if text_cols:
                        where_clauses = [f'"{col}" LIKE \'%{entity}%\'' for col in text_cols]
                        sql = f'SELECT * FROM "{table}" WHERE {" OR ".join(where_clauses)}'
                        sql_queries.append((db_path, db_name, sql))
                conn.close()
            except Exception as e:
                print(f"[DB Agent] Schema scan error for {db_name}: {e}")

        # Execute deterministic queries
        all_results = []
        total_rows = 0
        for db_path, db_name, sql in sql_queries:
            try:
                columns, rows = execute_read_only(db_path, sql)
                total_rows += len(rows)
                if rows:
                    result_text = format_results(columns, rows)
                    all_results.append(f"[{db_name}] {result_text}")
                    print(f"[DB Agent] ✅ {db_name}: {len(rows)} rows (deterministic)")
            except Exception as e:
                print(f"[DB Agent] ❌ {db_name}: {e}")

        if all_results:
            combined = "\n\n".join(all_results)
            thinking_steps.append(f"Deterministic SQL: {total_rows} rows from {len(all_results)} tables.")
            retrieved_docs = state.get("retrieved_docs") or {}
            retrieved_docs["db_sql"] = combined
            return {
                "documents": state.get("documents", []),
                "metadata_log": metadata_log + f"[DB] Deterministic: {total_rows} rows\n",
                "retrieved_docs": retrieved_docs,
                "db_data": combined,
                "_thinking": " | ".join(thinking_steps),
            }
        # Fall through to LLM if deterministic found nothing

    # --- Deep mode / fallback: LLM-based SQL ---
    sql_prompt = ChatPromptTemplate.from_messages([
        ("human", (
            "You are a SQL expert. Convert the user's question into a PRECISION SQL query "
            "for a SQLite database.\n\n"
            "DATABASE SCHEMA:\n{schema}\n\n"
            "USER QUESTION: {question}\n\n"
            "RULES:\n"
            "1. {entity_instruction}\n"
            "2. Return ONLY the SQL statement, no explanation.\n"
            "3. Use only SELECT statements.\n"
            "4. Use LIKE for flexible matching if ID is not exact, but prefer exact matches.\n"
            "5. If the question cannot be answered from these tables, "
            "return: SELECT 'NO_RELEVANT_TABLE' AS result;\n"
        ))
    ])

    chain = sql_prompt | config.llm | StrOutputParser()
    raw_sql = llm_invoke_with_retry(chain, {
        "schema": all_schemas,
        "question": question,
        "entity_instruction": entity_instruction
    })

    # Extract the SQL statement
    sql_match = re.search(r"(SELECT\s.+?)(?:;|$)", raw_sql, re.IGNORECASE | re.DOTALL)
    sql = sql_match.group(1).strip() if sql_match else raw_sql.strip()

    thinking_steps.append(f"Executing SQL: {sql[:100]}...")
    metadata_log += f"[DB] SQL: {sql}\n"
    print(f"[DB Agent] Generated SQL: {sql}")

    # --- Step 4: Execute against all databases ---
    all_results = []
    total_rows = 0

    for db_path in db_paths:
        db_name = db_path.split("/")[-1]
        try:
            columns, rows = execute_read_only(db_path, sql)
            total_rows += len(rows)
            if rows:
                result_text = format_results(columns, rows)
                all_results.append(f"[{db_name}]\n{result_text}")
                metadata_log += f"[DB] {db_name}: {len(rows)} rows returned\n"
                print(f"[DB Agent] ✅ {db_name}: {len(rows)} rows")
            else:
                metadata_log += f"[DB] {db_name}: No matching rows\n"
        except Exception as e:
            error_msg = str(e)
            # Try next DB if table doesn't exist in this one
            if "no such table" in error_msg.lower():
                continue
            all_results.append(f"[{db_name}] Error: {error_msg}")
            metadata_log += f"[DB] {db_name} ❌ Error: {error_msg}\n"
            print(f"[DB Agent] ❌ {db_name}: {e}")

    combined_result = "\n\n".join(all_results) if all_results else "No matching data found."
    thinking_steps.append(f"Retrieved {total_rows} rows from database.")

    # --- Step 5: Store in state ---
    retrieved_docs = state.get("retrieved_docs") or {}
    retrieved_docs["db_sql"] = combined_result

    return {
        "documents": state.get("documents", []),
        "metadata_log": metadata_log,
        "retrieved_docs": retrieved_docs,
        "db_data": combined_result,  # Updated key
        "_thinking": " | ".join(thinking_steps),
    }

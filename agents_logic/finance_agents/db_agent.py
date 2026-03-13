"""
================================================================================
Database Agent — Domain-Generic NL-to-SQL Retrieval (Precision Mode)
================================================================================
DOMAIN: finance (also reusable for any domain with .db files)
RESPONSIBILITY: Auto-discover SQLite databases, convert NL to PRECISION SQL,
                and provide authoritative structured results.

                Features:
                  - "Don't Give Up" Protocol: Always triggered for entities.
                  - Precision NL2SQL: Uses entity extraction for exact WHERE clauses.
                  - Schema-Aware: Injects schema context to prevent hallucinations.
================================================================================
"""

import os
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
from langchain_core.documents import Document
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
# Few-Shot NL2SQL Prompt — Domain-Agnostic with 3 Complex Examples
# ---------------------------------------------------------------------------
# Examples demonstrate entity-specific WHERE, aggregation, and cross-table
# joins so the LLM generates precise SQL the first time.
# ---------------------------------------------------------------------------

_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "You are a SQL expert. Convert the user's question into a PRECISION "
        "SQL query for a SQLite database.\n\n"
        "DATABASE SCHEMA (ONLY use tables and columns from this schema):\n"
        "{schema}\n\n"
        "--- ABSTRACT EXAMPLES (do NOT use these table/column names literally) ---\n\n"
        "Example 1 — Entity lookup:\n"
        "Q: Tell me about entity ENTITY_X\n"
        "REASONING: ENTITY_X appears to be a [type]. Table '[table_from_schema]' "
        "has column '[col_from_schema]' that stores [type] identifiers.\n"
        "SQL: SELECT * FROM [table_from_schema] WHERE [col_from_schema] "
        "LIKE '%ENTITY_X%';\n\n"
        "Example 2 — Aggregation:\n"
        "Q: What is the average VALUE_Y across all records?\n"
        "REASONING: VALUE_Y maps to column '[numeric_col]' in table '[table_from_schema]'.\n"
        "SQL: SELECT AVG([numeric_col]) AS avg_value FROM [table_from_schema];\n\n"
        "Example 3 — Cross-table join:\n"
        "Q: Show details for entity ENTITY_Z from related data\n"
        "REASONING: ENTITY_Z appears in '[table_a]'.'[col_a]'. Related "
        "data is in '[table_b]' joined on '[join_col]'.\n"
        "SQL: SELECT a.*, b.* FROM [table_a] a "
        "JOIN [table_b] b ON a.[join_col] = b.[join_col] "
        "WHERE a.[col_a] LIKE '%ENTITY_Z%';\n\n"
        "--- END ABSTRACT EXAMPLES ---\n\n"
        "ENTITY TYPE HINT: {entity_type_hint}\n\n"
        "USER QUESTION: {question}\n\n"
        "STRICT RULES:\n"
        "1. {entity_instruction}\n"
        "2. SCHEMA BINDING: Before writing SQL, verify that every table and "
        "column you reference EXISTS in the DATABASE SCHEMA above. "
        "If a table or column does not exist in the schema, do NOT use it.\n"
        "3. Return ONLY the SQL statement, no explanation.\n"
        "4. Use only SELECT statements.\n"
        "5. Use LIKE for flexible matching if the identifier is not exact.\n"
        "6. If no table in the schema logically matches the entity type or "
        "question, return: SELECT 'NO_RELEVANT_TABLE' AS result;\n"
        "7. NEVER invent table or column names not present in the schema.\n"
    ))
])

_SQL_FIX_PROMPT = ChatPromptTemplate.from_messages([
    ("human", (
        "The following SQL query failed with an error. Fix the query.\n\n"
        "DATABASE SCHEMA:\n{schema}\n\n"
        "ORIGINAL SQL:\n{sql}\n\n"
        "ERROR MESSAGE:\n{error}\n\n"
        "STRICT RULE: Do NOT prefix table names with database file names.\n"
        "Return ONLY the corrected SQL statement, no explanation.\n"
        "Use only columns and tables that exist in the schema above."
    ))
])

_MAX_SQL_RETRIES = 3


def _sanitize_sql(sql: str) -> str:
    """
    Remove database filename prefixes from SQL (e.g. 'products.db.table_name' -> 'table_name').
    LLMs like Gemini often add these despite instructions.
    """
    # Matches patterns like words followed by .db. or .sqlite.
    sql = re.sub(r'\b[\w-]+\.(?:db|sqlite|sqlite3)\.', '', sql, flags=re.IGNORECASE)
    # Also catch double prefixes like 'products.db."table"'
    sql = re.sub(r'\b[\w-]+\.(?:db|sqlite|sqlite3)\."', '"', sql, flags=re.IGNORECASE)
    return sql.strip()


def _self_correct_sql(
    sql: str,
    db_path: str,
    all_schemas: str,
    thinking_steps: list,
    metadata_log: str,
) -> tuple[list[str], list[tuple], str, str]:
    """
    Execute SQL with self-correction loop.

    If the query fails with a *fixable* error (wrong column, ambiguous name,
    syntax error), the error is fed back to the LLM to regenerate the query.
    Retries up to ``_MAX_SQL_RETRIES`` times.

    "no such table" is NOT retried — it means the table simply doesn't exist
    in this particular .db file, so the caller should try the next database.

    Returns:
        (columns, rows, updated_metadata_log, last_error)
    """
    current_sql = _sanitize_sql(sql)
    last_error = ""

    for attempt in range(_MAX_SQL_RETRIES):
        try:
            columns, rows = execute_read_only(db_path, current_sql)
            return columns, rows, metadata_log, ""
        except Exception as e:
            last_error = str(e)

            # "no such table" → table doesn't exist in this DB, skip to next
            if "no such table" in last_error.lower():
                return [], [], metadata_log, last_error

            # Only retry on fixable schema errors (wrong column, syntax, etc.)
            if any(kw in last_error.lower() for kw in [
                "no such column", "ambiguous column",
                "syntax error", "near \"",
            ]):
                thinking_steps.append(
                    f"SQL error (attempt {attempt + 1}): {last_error[:80]}... retrying"
                )
                print(
                    f"[Finance DB Agent] ⚠️ SQL error (attempt {attempt + 1}/{_MAX_SQL_RETRIES}): "
                    f"{last_error[:100]}"
                )
                metadata_log += f"[DB] SQL error (attempt {attempt + 1}): {last_error}\n"

                # Ask LLM to fix the query
                try:
                    fix_chain = _SQL_FIX_PROMPT | config.llm | StrOutputParser()
                    raw_fix = llm_invoke_with_retry(fix_chain, {
                        "schema": all_schemas,
                        "sql": current_sql,
                        "error": last_error,
                    })
                    sql_match = re.search(
                        r"(SELECT\s.+?)(?:;|$)", raw_fix,
                        re.IGNORECASE | re.DOTALL,
                    )
                    raw_sql = sql_match.group(1).strip() if sql_match else raw_fix.strip()
                    current_sql = _sanitize_sql(raw_sql)
                    metadata_log += f"[DB] Corrected SQL: {current_sql}\n"
                    print(f"[Finance DB Agent] 🔧 Corrected SQL: {current_sql[:100]}")
                except Exception as fix_err:
                    thinking_steps.append(f"SQL fix failed: {fix_err}")
                    break
            else:
                # Non-fixable error (e.g. database locked) — don't retry
                break

    # All retries exhausted
    thinking_steps.append(f"SQL failed after {_MAX_SQL_RETRIES} attempts: {last_error[:80]}")
    metadata_log += f"[DB] ❌ SQL failed after {_MAX_SQL_RETRIES} attempts: {last_error}\n"
    return [], [], metadata_log, last_error


@vera_agent("Finance DB Agent")
def run(state: GraphState) -> dict:
    """
    DB AGENT: Precision NL-to-SQL retrieval with self-correction.

    Flow:
        1. Auto-discover .db files in source_documents/{user_domain}/
        2. If schema query → return table listing
        3. Extract Entity → if found, restrict SQL to that entity.
        4. LLM generates SQL (Few-Shot Schema-Aware + Entity-Specific).
        5. Execute SQL with self-correction loop (up to 3 retries).
    """
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "finance")
    target_entity = state.get("target_entity", "GENERAL")
    entity_type = state.get("entity_type", "GENERAL")

    # --- Guard Clause: Fast-fail if intent doesn't need DB ---
    intent = state.get("intent", "")
    if intent not in ("db_query", "cross_reference", "spec_retrieval", ""):
        print(f"[Finance DB Agent] ⏭️ Fast-fail: intent='{intent}' is not DB-related")
        return {}

    thinking_steps = []
    metadata_log = state.get("metadata_log", "")

    # --- Step 1: Discover databases for this domain ---
    db_paths = discover_databases(user_domain)

    if not db_paths:
        msg = f"No databases found for domain '{user_domain}'."
        thinking_steps.append(f"Inspecting source_documents/{user_domain}/... {msg}")
        print(f"[Finance DB Agent] ⚠️ {msg}")
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
    print(f"[Finance DB Agent] Found {len(db_paths)} database(s) for '{user_domain}': {db_names}")

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

    # --- Step 3: Entity — use router's target_entity (no regex fallback) ---
    entity = state.get("target_entity", "").strip()
    entity_type = state.get("entity_type", "GENERAL").strip()

    # Simple check: if entity is empty or generic, mark as general query
    if not entity or entity.upper() == "GENERAL":
        entity = "GENERAL_QUERY"
    is_general = entity == "GENERAL_QUERY"

    thinking_steps.append(
        f"Analyzing query context... Target Entity: {entity} (type: {entity_type})"
    )
    print(f"[Finance DB Agent] 🎯 Target Entity: {entity} (type: {entity_type})")

    # Build entity type hint for schema-table mapping
    if entity_type and entity_type.upper() != "GENERAL":
        entity_type_hint = (
            f"The user is asking about a '{entity_type}'. "
            f"Map this to the most relevant table in the schema that "
            f"stores {entity_type} data."
        )
    else:
        entity_type_hint = "No specific entity type detected."

    # Construct specific instructions based on entity presence
    entity_instruction = ""
    if not is_general:
        entity_instruction = (
            f"IMPORTANT: The user is asking specifically about '{entity}'.\n"
            f"Your SQL MUST include a WHERE clause to filter for '{entity}'.\n"
            f"Use columns from the SCHEMA ABOVE that could match this entity.\n"
            f"DO NOT return data for other entities.\n"
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
                print(f"[Finance DB Agent] Schema scan error for {db_name}: {e}")

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
                    print(f"[Finance DB Agent] ✅ {db_name}: {len(rows)} rows (deterministic)")
            except Exception as e:
                print(f"[Finance DB Agent] ❌ {db_name}: {e}")

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

    # --- Deep mode / fallback: Few-Shot LLM-based SQL with Self-Correction ---
    chain = _SQL_PROMPT | config.llm | StrOutputParser()
    raw_sql = llm_invoke_with_retry(chain, {
        "schema": all_schemas,
        "question": question,
        "entity_instruction": entity_instruction,
        "entity_type_hint": entity_type_hint,
    })

    # Extract the SQL statement
    sql_match = re.search(r"(SELECT\s.+?)(?:;|$)", raw_sql, re.IGNORECASE | re.DOTALL)
    sql = sql_match.group(1).strip() if sql_match else raw_sql.strip()

    thinking_steps.append(f"Executing SQL: {sql[:100]}...")
    metadata_log += f"[DB] SQL: {sql}\n"
    print(f"[Finance DB Agent] Generated SQL: {sql}")

    # --- Step 4: Execute against all databases with self-correction ---
    all_results = []
    total_rows = 0

    for db_path in db_paths:
        db_name = db_path.split("/")[-1]

        columns, rows, metadata_log, last_err = _self_correct_sql(
            sql=sql,
            db_path=db_path,
            all_schemas=all_schemas,
            thinking_steps=thinking_steps,
            metadata_log=metadata_log,
        )

        # Table doesn't exist in this DB — skip silently to next
        if last_err and "no such table" in last_err.lower():
            print(f"[Finance DB Agent] ⏭️ {db_name}: table not found, trying next DB")
            continue

        if rows:
            result_text = format_results(columns, rows)
            all_results.append(f"[{db_name}]\n{result_text}")
            total_rows += len(rows)
            metadata_log += f"[DB] {db_name}: {len(rows)} rows returned\n"
            print(f"[Finance DB Agent] ✅ {db_name}: {len(rows)} rows")
        elif not columns and not rows:
            # _self_correct_sql exhausted retries — already logged
            pass
        else:
            metadata_log += f"[DB] {db_name}: No matching rows\n"

    # --- Step 5: Format for State Update ---
    combined_result = "\n\n".join(all_results) if all_results else "No matching data found in database."
    thinking_steps.append(f"Retrieved {total_rows} rows from database.")

    # Convert DB results to structured facts for the Auditor
    db_facts = []
    if all_results and total_rows > 0:
        # Optimization: If we have 1 row, extract individual columns as facts
        # This allows the Auditor to compare 'voltage_max' (DB) vs 'voltage_max' (Doc)
        # instead of comparing a big 'database_record' blob.
        for db_path in db_paths:
            db_name = os.path.basename(db_path)
            try:
                # Re-query specifically for fact extraction if needed, or use cached rows
                # For simplicity here, we re-parse the last successful result if total_rows is small
                columns, rows = execute_read_only(db_path, sql)
                if len(rows) == 1:
                    row = rows[0]
                    for i, col_name in enumerate(columns):
                        val = str(row[i])
                        if val.lower() not in ("none", "null", ""):
                            db_facts.append({
                                "entity": target_entity,
                                "attribute": col_name.lower(),
                                "value": val,
                                "source_type": "db",
                                "source_doc": db_name,
                                "date": state.get("latest_timestamp", "unknown"),
                                "confidence": "HIGH",
                            })
                else:
                    # Multi-row fallback: traditional blob
                    db_facts.append({
                        "entity": target_entity,
                        "attribute": "database_record",
                        "value": combined_result[:1000],
                        "source_type": "db",
                        "source_doc": db_name,
                        "date": state.get("latest_timestamp", "unknown"),
                        "confidence": "HIGH",
                    })
            except Exception:
                continue

    # Also add as a Document for RAG-only responders
    new_docs = []
    if total_rows > 0:
        new_docs = [Document(
            page_content=combined_result,
            metadata={"source": "database", "domain": user_domain, "type": "db_record"}
        )]

    # ====== 新增：智能早退判定 (Satisfaction Scoring) ======
    satisfaction_score = 0.0
    if total_rows == 1:
        satisfaction_score = 1.0
    elif total_rows > 1:
        satisfaction_score = 0.8
    
    should_short_circuit = False
    if state.get("intent") == "db_query" and satisfaction_score >= 0.8:
        should_short_circuit = True
        print(f"[Finance DB Agent] 🛑 高满意度 ({satisfaction_score})，触发早退信号。")

    # Per-step return: ONLY the tokens we added (reducers handle the merge)
    return {
        "documents": state.get("documents", []) + new_docs,
        "db_facts": db_facts,
        "db_data": combined_result,
        "is_resolved": should_short_circuit,
        "satisfaction_score": satisfaction_score,
        "metadata_log": metadata_log.replace(state.get("metadata_log", ""), "").strip(),
        "thought_process": thinking_steps,
        "_thinking": " | ".join(thinking_steps),
    }


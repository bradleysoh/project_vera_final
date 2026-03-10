"""
================================================================================
Project VERA — Database Utilities (Domain-Agnostic)
================================================================================

Shared utility for discovering and querying SQLite databases in any domain's
source_documents/ folder.  Every function is domain-agnostic — the caller
passes a domain name and this module resolves the paths automatically.

Usage:
    from shared.db_utils import discover_databases, get_all_schemas, execute_read_only

    dbs = discover_databases("semiconductor")
    schemas = get_all_schemas("semiconductor")
    cols, rows = execute_read_only(dbs[0], "SELECT * FROM product_specs LIMIT 5")

================================================================================
"""

import os
import glob
import sqlite3

# ---------------------------------------------------------------------------
# Project root (two levels up from shared/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "source_documents")


# ===========================================================================
# DATABASE DISCOVERY
# ===========================================================================

def _csv_to_sqlite(domain: str) -> list[str]:
    """
    Auto-convert any .csv files in source_documents/{domain}/ into
    SQLite databases with a '_auto.db' suffix.

    Skips conversion if the corresponding _auto.db already exists and
    is newer than the CSV.

    Returns:
        List of paths to newly created or existing auto-generated .db files.
    """
    import pandas as pd

    domain_dir = os.path.join(_SOURCE_DIR, domain)
    if not os.path.isdir(domain_dir):
        return []

    csv_files = glob.glob(os.path.join(domain_dir, "*.csv"))
    created = []

    for csv_path in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        db_path = os.path.join(domain_dir, f"{base_name}_auto.db")

        # Skip if DB already exists and is newer than the CSV
        if os.path.exists(db_path):
            csv_mtime = os.path.getmtime(csv_path)
            db_mtime = os.path.getmtime(db_path)
            if db_mtime >= csv_mtime:
                created.append(db_path)
                continue

        try:
            df = pd.read_csv(csv_path)
            # Deduplicate column names case-insensitively (SQLite is case-insensitive)
            cols = list(df.columns)
            seen: dict[str, int] = {}
            new_cols = []
            for c in cols:
                key = c.lower()
                if key in seen:
                    seen[key] += 1
                    new_cols.append(f"{c}_{seen[key]}")
                else:
                    seen[key] = 1
                    new_cols.append(c)
            df.columns = new_cols
            # Sanitize table name: replace spaces/dashes with underscores
            table_name = base_name.replace("-", "_").replace(" ", "_").lower()
            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()
            print(f"[DB Utils] 📦 Auto-converted CSV → SQLite: {csv_path} → {db_path} ({len(df)} rows)")
            created.append(db_path)
        except Exception as e:
            print(f"[DB Utils] ❌ CSV conversion failed for {csv_path}: {e}")

    return created


def discover_databases(domain: str) -> list[str]:
    """
    Find all .db / .sqlite / .sqlite3 files inside
    source_documents/{domain}/.

    Auto-converts any .csv files to SQLite first.

    Returns:
        Sorted list of absolute paths to database files.
    """
    # Auto-convert CSVs to SQLite before discovery
    _csv_to_sqlite(domain)

    domain_dir = os.path.join(_SOURCE_DIR, domain)
    if not os.path.isdir(domain_dir):
        return []

    patterns = ("*.db", "*.sqlite", "*.sqlite3")
    found: list[str] = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(domain_dir, "**", pat), recursive=True))

    return sorted(set(found))


# ===========================================================================
# SCHEMA DISCOVERY
# ===========================================================================

def get_schema(db_path: str, include_samples: bool = True) -> str:
    """
    Inspect a single database and return a human-readable schema description.

    For each table, shows:
      - Column names and types (via PRAGMA table_info)
      - 2 sample rows (if include_samples is True)

    Args:
        db_path: Absolute path to the SQLite file.
        include_samples: Whether to include sample data rows.

    Returns:
        Multi-line schema string, or empty string if file doesn't exist.
    """
    if not os.path.exists(db_path):
        return ""

    db_name = os.path.basename(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        conn.close()
        return f"[{db_name}] No tables found."

    parts = [f"Database: {db_name}"]
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_defs = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        parts.append(f"  Table '{table}': {col_defs}")

        if include_samples:
            col_names = [c[1] for c in cols]
            cursor.execute(f"SELECT * FROM {table} LIMIT 2")
            samples = cursor.fetchall()
            for row in samples:
                row_str = ", ".join(f"{col_names[i]}={row[i]}" for i in range(len(row)))
                parts.append(f"    [SCHEMA SAMPLE - DO NOT USE AS FILTER]: {row_str}")

    conn.close()
    return "\n".join(parts)


def get_all_schemas(domain: str) -> str:
    """
    Get schemas for ALL databases in a domain.

    Returns:
        Combined schema string for every .db file, or a 'no databases' message.
    """
    dbs = discover_databases(domain)
    if not dbs:
        return f"No databases found for domain '{domain}'."

    schemas = []
    for db_path in dbs:
        schema = get_schema(db_path)
        if schema:
            schemas.append(schema)

    return "\n\n".join(schemas) if schemas else f"No tables found in '{domain}' databases."


# ===========================================================================
# QUERY EXECUTION (READ-ONLY)
# ===========================================================================

def execute_read_only(db_path: str, sql: str) -> tuple[list[str], list[tuple]]:
    """
    Execute a SELECT query against a read-only database connection.

    Args:
        db_path: Absolute path to the SQLite file.
        sql: The SQL query to execute (must start with SELECT).

    Returns:
        (column_names, rows)

    Raises:
        ValueError: If the query is not a SELECT statement.
        sqlite3.Error: On database errors.
    """
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned.upper().startswith("SELECT"):
        raise ValueError(
            f"Only SELECT queries are allowed. Got: {cleaned[:50]}"
        )

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute(cleaned)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return columns, rows


def format_results(columns: list[str], rows: list[tuple]) -> str:
    """Format SQL results as a Markdown table string."""
    if not rows:
        return "No rows returned."

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)

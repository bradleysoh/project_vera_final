# 🧑‍💻 TEAM GUIDE — Project VERA Developer Guide

This guide explains how each team member can contribute their own **data** and **agent logic** to the VERA multi-agent system.

---

## Project Structure Overview

```
proj_vera/
├── shared/                               # ⛔ DO NOT MODIFY — shared infrastructure
│   ├── graph_state.py                    # GraphState TypedDict (state schema)
│   ├── config.py                         # LLM, VectorStore, RBAC, retry logic
│   ├── agent_base.py                     # @vera_agent decorator
│   └── dynamic_loader.py                 # Auto-discovers domain agent subfolders
│
├── agents_logic/                         # ✅ YOUR AGENTS GO HERE
│   ├── _template_agent.py                # Template — copy this to start
│   ├── router_agent.py                   # SHARED: Intent + Domain routing
│   ├── response_agent.py                 # SHARED: LLM response generation
│   ├── escalation_agent.py               # SHARED: Security + out-of-domain escalation
│   ├── semiconductor_agents/             # DOMAIN — auto-discovered
│   │   ├── tech_spec_agent.py
│   │   ├── compliance_agent.py
│   │   └── discrepancy_agent.py
│   └── medical_agents/                   # DOMAIN — auto-discovered
│       ├── tech_spec_agent.py
│       ├── compliance_agent.py
│       └── discrepancy_agent.py
│
├── source_documents/                     # ✅ YOUR DATA GOES HERE
│   ├── semiconductor/                    # Domain data
│   ├── medical/                          # Domain data (placeholder)
│   └── README.md                         # Data format instructions
│
├── app.py                                # Central orchestrator (dynamic loading)
├── ingestion.py                          # Data ingestion pipeline (domain-aware)
└── streamlit_app.py                      # Web UI
```

---

## 🔄 How Dynamic Loading Works

The system **automatically discovers** domain agents at startup:

1. Scans `agents_logic/` for `*_agents/` subfolders.
2. Imports each `.py` file; looks for a `run()` function.
3. Automatically maps agents to roles based on convention:
   - `db_agent.py` → `{domain}_db_query`
   - `official_docs_agent.py` → `{domain}_official`
   - `informal_docs_agent.py` → `{domain}_informal`
   - `discrepancy_agent.py` → `{domain}_discrepancy`

5. **Hybrid Routing Logic**: If no keywords are found, the system uses an **LLM intent fallback** to categorize the query based on your `domain_config.py` definitions.

**To add a new domain**, simply create a new folder and a `domain_config.py` file!

---

## 🔐 Enterprise Security Handrails

### Double-Filter Retrieval & Domain Isolation

Every retrieval call applies TWO filters simultaneously to ensure zero "Domain Bleed":

```python
filter = {
    "$and": [
        {"domain": user_domain},                    # Strict Domain Isolation
        {"access_level": {"$in": allowed_levels}}   # Role-based access
    ]
}
```

> [!IMPORTANT]
> The `user_domain` is treated as immutable once assigned to the state. Agents must NEVER override the user's domain selection based on query keywords.

### 🛡️ Information Lock (Grounding)

All Response Agents must follow the **Information Lock** protocol:
- **Strict Grounding**: Only use facts provided in the `extracted_facts` or `documents` state fields.
- **Fail-Safe**: If no relevant facts are found, return the standardized "Data Not Found" message.
- **No Hallucinations**: Never supplement with external LLM training data for technical specs.

---

## 📋 Template A: Adding a New Domain

```bash
# 1. Create the domain agent folder
mkdir agents_logic/medical_agents/
touch agents_logic/medical_agents/__init__.py

# 2. Add domain_config.py (CRITICAL)
# Define your domain's keywords, aliases, and metadata schema here.
# This powers the Surgical Router.

# 3. Implement the 4 Core Agents
# db_agent.py, official_docs_agent.py, informal_docs_agent.py, discrepancy_agent.py

# 4. Add data to source_documents/medical/
# Place .txt or .pdf files for RAG and .db files for SQL.

> [!TIP]
> **SQL Sanitization**: VERA automatically strips database prefixes (e.g., `products.db.table`) from generated SQL to ensure compatibility across cloud LLMs.
```

The domain will be **auto-discovered** — no code changes needed in `app.py`, `router_agent.py`, or `streamlit_app.py`.

---

## 📋 Template B: Data Preparation

### File Naming Convention

**Format**: `Domain_Type_Version_Access.[txt|pdf]`

| Part | Allowed Values | Example |
|------|---------------|---------|
| Domain | `Semi`, `Med`, or custom | `Semi` |
| Type | `Spec`, `Email`, `SOP`, `DB`, `DM`, `Doc` | `Spec` |
| Version | `v1`, `v2`, `v1.0`, etc. | `v4.2` |
| Access | `Public`, `Internal`, `Confidential` | `Public` |

### RBAC Auto-Tagging

| File Type | Default Access | Override? |
|-----------|---------------|-----------|
| `Email`, `DM` | `internal_only` | ✅ Yes |
| `DB` | `confidential` | ✅ Yes |
| `Spec`, `SOP`, `Doc` | `public` | ✅ Yes |

---

## 📋 Template C: Agent Code (Contract)

All agents MUST use the `@vera_agent` decorator and accept `GraphState`.

```python
from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.advanced_rag import extract_structured_facts

@vera_agent("Official Agent")
def run(state: GraphState) -> dict:
    # High-precision retrieval example
    facts = extract_structured_facts(
        state["question"],
        entity=state["target_entity"],
        attribute=state["target_attribute"],
        source_filter=["datasheet"]
    )
    return {"extracted_facts": facts}
```

### The 4 Required Agents per Domain:

| Agent File | Role | Goal |
| :--- | :--- | :--- |
| `db_agent.py` | SQL Expert | Natural language to SQL querying. Authority: **3 (Primary)**. |
| `official_docs_agent.py` | Specs Specialist | High-precision fact extraction from datasheets and SOPs. Authority: **2 (Gold Standard)**. |
| `informal_docs_agent.py` | Context Researcher | Context research in emails and memos. Authority: **1 (Conditional)**. |
| `discrepancy_agent.py` | Deterministic Auditor| Core logic for conflict detection and hierarchy enforcement. **No LLM calls**. |

---

## 🔍 Triangulation Discrepancy Engine

VERA uses a **deterministic authority hierarchy** to resolve conflicts between sources:

1. **Database Facts (Priority 3)**: Current production truth.
2. **Official Documents (Priority 2)**: Specifications, SOPs, and Manuals.
3. **Informal Communications (Priority 1)**: Emails, DMs, and Memos (Only overrides if date is newer).

### Semantic Attribute Normalization
To prevent false discrepancies (e.g., `voltage` mismatching `voltage_max`), the Discrepancy Agent applies **Semantic Core Matching**. It strips filler words (max, min, series) and compares the normalized cores.

---

## 🔧 GraphState Definition

```python
class GraphState(TypedDict):
    question: str             # The user's input query
    generation: str           # The LLM-generated response
    user_role: str            # "senior" or "junior"
    user_domain: str          # User's assigned domain (auto-discovered)
    documents: List[Document] # Retrieved documents from ChromaDB
    route: str                # "technical", "compliance", or "escalate"
    flagged: bool             # True if flagged (security OR out-of-domain)
    metadata_log: str         # Audit log for retrieval transparency
    retrieved_docs: dict      # Per-agent docs: {"tech": [...], "compliance": [...]}
    discrepancy_report: str   # Structured report from Case Agent
    next_agent: str           # Detected query domain for routing
    refinement_count: int     # Tracks discussion loop iterations
    max_refinements: int      # Configurable limit for discussion loop (default: 0)
    critique: str             # Feedback from Discrepancy Agent to Response Agent
```

---

## ⚠️ Rules

| ✅ DO | ❌ DON'T |
|-------|---------|
| Use `@vera_agent("Name")` decorator | Write raw print statements for logging |
| Import from `shared.config` | Create your own LLM instance |
| Enforce the **Information Lock** protocol | Allow LLM hallucinations or external info |
| Treat `user_domain` as **immutable** | Override domain based on query text |
| Name your folder `{domain}_agents/` | Use arbitrary folder names |
| Include all 4 agent files per domain | Skip `discrepancy_agent.py` |
| Pass `user_domain` to `retrieve_with_rbac()` | Hardcode domain names |
| Follow `Domain_Type_Version_Access.txt` | Use arbitrary file names |

---

## 🧪 Testing

```bash
# Setup conda environment
conda env create -f environment.yml
conda activate vera

# Run the system
python ingestion.py             # Domain-aware loading (.txt, .pdf)
python app.py                   # Full test suite (7 scenarios including CoWoS-S)
streamlit run streamlit_app.py  # Web UI
```

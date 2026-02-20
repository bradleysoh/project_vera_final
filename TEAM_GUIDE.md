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

1. Scans `agents_logic/` for `*_agents/` subfolders
2. Imports each `.py` file's `run()` function
3. Registers them as LangGraph nodes: `{domain}_{role}`

```
semiconductor_agents/tech_spec_agent.py  → node: semiconductor_retrieve_specs
semiconductor_agents/compliance_agent.py → node: semiconductor_retrieve_compliance
medical_agents/tech_spec_agent.py        → node: medical_retrieve_specs
```

**To add a new domain**, just create a new `{domain}_agents/` folder with the 3 required agents!

---

## 🔐 Multi-Domain RBAC

### Double-Filter Retrieval

Every retrieval call applies TWO filters simultaneously:

```python
filter = {
    "$and": [
        {"domain": user_domain},                    # Domain isolation
        {"access_level": {"$in": allowed_levels}}   # Role-based access
    ]
}
```

### Role Access Levels

| Role | Access Levels |
|------|--------------|
| `senior` | `public`, `internal_only`, `confidential` |
| `junior` | `public` only |

### Out-of-Domain Protection

If a user's `user_domain` does not match the query's detected domain:
- Router flags the query as **out-of-domain**
- Query is routed to the **Escalation Agent**
- Escalation message specifies the domain mismatch

Example: A semiconductor engineer asking "What are the FDA clinical trial requirements?" → escalated.

---

## 📋 Template A: Adding a New Domain

```bash
# 1. Create the domain agent folder
mkdir agents_logic/automotive_agents/
touch agents_logic/automotive_agents/__init__.py

# 2. Copy agents from an existing domain
cp agents_logic/semiconductor_agents/tech_spec_agent.py agents_logic/automotive_agents/
cp agents_logic/semiconductor_agents/compliance_agent.py agents_logic/automotive_agents/
cp agents_logic/semiconductor_agents/discrepancy_agent.py agents_logic/automotive_agents/

# 3. Edit each agent — change @vera_agent name and source_filter
# 4. Add data to source_documents/automotive/
# 5. Re-run ingestion and test
python ingestion.py && python app.py
```

The domain will be **auto-discovered** — no code changes needed in `app.py`, `router_agent.py`, or `streamlit_app.py`.

---

## 📋 Template B: Data Preparation

### File Naming Convention

**Format**: `Domain_Type_Version_Access.txt`

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

## 📋 Template C: Agent Code

### Required contract:

```python
from shared.graph_state import GraphState
from shared.agent_base import vera_agent
from shared.config import retrieve_with_rbac

@vera_agent("My Domain Agent Name")
def run(state: GraphState) -> dict:
    question = state["question"]
    user_role = state["user_role"]
    user_domain = state.get("user_domain", "my_domain")

    documents, metadata_log = retrieve_with_rbac(
        query=question,
        user_role=user_role,
        user_domain=user_domain,       # Domain isolation filter
        source_filter=["datasheet"],    # Source type filter
        k=4,
    )
    return {"documents": documents, "metadata_log": metadata_log}
```

### Each domain needs 3 agents:

| Agent File | Role | Node Name |
|-----------|------|-----------|
| `tech_spec_agent.py` | Retrieval (specs) | `{domain}_retrieve_specs` |
| `compliance_agent.py` | Retrieval (SOPs) | `{domain}_retrieve_compliance` |
| `discrepancy_agent.py` | Cross-agent comparison | `{domain}_check_discrepancy` |

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
| Name your folder `{domain}_agents/` | Use arbitrary folder names |
| Include all 3 agent files per domain | Skip `discrepancy_agent.py` |
| Pass `user_domain` to `retrieve_with_rbac()` | Hardcode domain names |
| Follow `Domain_Type_Version_Access.txt` | Use arbitrary file names |

---

## 🧪 Testing

```bash
# Setup conda environment
conda env create -f environment.yml
conda activate vera

# Run the system
python ingestion.py             # Domain-aware loading
python app.py                   # Full test suite (6 scenarios including out-of-domain)
streamlit run streamlit_app.py  # Web UI
```

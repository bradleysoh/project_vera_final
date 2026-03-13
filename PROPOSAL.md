# 🔍 PROJECT VERA — Project Proposal
> **Virtual Engineering Review Agent**
> *A Multi-Agent System for Technical Document Auditing & Compliance — Any Industry*

---

## 1. Executive Summary
Project VERA (Virtual Engineering Review Agent) is a production-grade, multi-agent AI system designed to automate technical document auditing, cross-reference checking, and compliance verification. It addresses the critical challenge of documentation drift in regulated industries by autonomously auditing live database records against official specifications and informal engineering communications (emails, memos).

The system features a **Surgical Router** for domain-isolated processing, an **Information Lock** protocol to eliminate hallucinations, and a deterministic **Triangulation Discrepancy Engine**. VERA supports local (Ollama) and cloud (Gemini, Groq) LLM backends, making it adaptable to diverse security and performance requirements.

---

## 2. Problem Statement
In technical industries, the gap between official specifications and actual engineering practices causes significant risk. Decisions made via informal channels (emails, chat) often override official datasheets but are not captured in formal systems.

**The "RTX-9000" Gap:**
- 📄 **Datasheet**: "Max voltage = 5.0V"
- 📧 **Internal Email**: "URGENT: Lowering to 3.3V due to heat failures"
- ❌ **Outcome**: Engineers testing at 5.0V cause product failures, unaware of the undocumented change.

---

## 3. Project Objectives
1. **O1: Multi-Agent Orchestration**: Implement a LangGraph-based workflow with specialized agents.
2. **O2: Surgical Routing**: Use LLM-NER and keyword heuristics to route queries to domain-specific clusters.
3. **O3: Triangulation Engine**: Automatically detect conflicts between SQL databases, official PDFs, and informal emails.
4. **O4: Security & RBAC**: Enforce Role-Based Access Control at the retrieval layer (ChromaDB metadata filters).
5. **O5: Information Lock**: Ensure zero hallucinations via strict grounding protocols.

---

## 4. System Architecture

VERA v2.5 utilizes a **Surgical Router** architecture that delegates tasks to domain-isolated agent clusters.

### Architecture Flow
1. **Surgical Router**: Filters query by domain and intent (NER: Entity, Attribute, Access Level).
2. **Domain Clusters**:
   - **DB Agent**: Queries structured SQL databases.
   - **Official Agent**: High-precision RAG from datasheets/SOPs.
   - **Informal Agent**: Context research from emails/memos.
3. **Triangulation Engine**: Cross-references all gathered facts using a deterministic authority hierarchy:
   - **DB Facts [3]** > **Official Docs [2]** > **Informal Comms [1]**.
4. **Response Generator**: Compiles final report with citations or triggers **Human Escalation** if security flags are tripped.

---

## 5. Security Implementation (RBAC)
VERA implements security at the retrieval layer, not just the prompt level.
- **Layer 1**: Router identifies sensitive intents (e.g., "internal emails") for junior users and short-circuits to **Escalation**.
- **Layer 2**: ChromaDB metadata filtering ensures junior users never see "internal_only" or "confidential" chunks.
- **Layer 3**: Immutable Domain Isolation prevents unauthorized "Domain Bleed."

---

## 6. Technology Stack
- **LangGraph**: Orchestration of multi-agent state machine.
- **LangChain**: RAG pipeline and prompt management.
- **ChromaDB**: Local vector store with metadata filtering.
- **Ollama**: 100% local LLM execution (Option A).
- **Google Gemini**: High-quality cloud inference (Option B).
- **Streamlit**: Interactive Web UI with agent execution traces.

---

## 7. Conclusion
Project VERA demonstrates a robust, enterprise-grade approach to technical document auditing. By combining surgical routing with deterministic triangulation, it provides a trustworthy compliance tool that eliminates the risk of "documentation drift" while maintaining strict security boundaries.

"""
================================================================================
Project VERA - Interactive Streamlit Chat Interface
Virtual Engineering Review Agent
================================================================================

Launch:
    streamlit run streamlit_app.py
================================================================================
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure project modules are importable
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Import VERA core from app.py
# ---------------------------------------------------------------------------
from app import build_graph
from shared.config import (
    LLM_BACKEND, OLLAMA_MODEL, GEMINI_API_KEY, GROQ_API_KEY,
    get_available_roles, get_current_backend, switch_backend,
)
from shared.graph_state import GraphState
from shared.dynamic_loader import get_available_domains
from shared.email_utils import send_alert_email, is_email_configured
from shared.system_logging import log_vera_step

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="VERA — Virtual Engineering Review Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — Premium dark theme
# ============================================================================
st.markdown("""
<style>
/* --- Global --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* --- Header --- */
.vera-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.08);
}
.vera-header h1 {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.vera-header p {
    color: rgba(255,255,255,0.7);
    font-size: 0.95rem;
    margin: 0;
}

/* --- Status badges --- */
.badge-row { display: flex; gap: 0.5rem; margin-top: 0.8rem; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
    backdrop-filter: blur(10px);
}
.badge-green  { background: rgba(64,192,87,0.25); color: #69db7c; border: 1px solid rgba(64,192,87,0.3); }
.badge-blue   { background: rgba(77,171,247,0.25); color: #74c0fc; border: 1px solid rgba(77,171,247,0.3); }
.badge-purple { background: rgba(155,89,182,0.25); color: #c084fc; border: 1px solid rgba(155,89,182,0.3); }
.badge-amber  { background: rgba(255,183,77,0.25); color: #ffd43b; border: 1px solid rgba(255,183,77,0.3); }

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #162335 100%);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #74c0fc;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 1.5rem;
}

/* --- Agent step cards --- */
.agent-step {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #4dabf7;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
}
.agent-step.route    { border-left-color: #4dabf7; }
.agent-step.retrieve { border-left-color: #69db7c; }
.agent-step.generate { border-left-color: #ffd43b; }
.agent-step.discrep  { border-left-color: #ff6b6b; }
.agent-step.escalate { border-left-color: #ff6b6b; }

/* --- Document cards --- */
.doc-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.doc-card .doc-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.5rem;
}
.doc-card .doc-title { font-weight: 600; font-size: 0.9rem; }
.doc-card .doc-content {
    font-size: 0.8rem; color: rgba(255,255,255,0.6);
    max-height: 200px; overflow-y: auto;
    white-space: pre-wrap;
}
.doc-summary {
    font-size: 0.85rem; color: rgba(255,255,255,0.7);
    background: rgba(0,0,0,0.2);
    padding: 0.8rem;
    border-radius: 6px;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* --- Escalation alert --- */
.escalation-alert {
    background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,107,107,0.05));
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.escalation-alert h3 { color: #ff6b6b; margin: 0 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INIT
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False


import re
_INVALID_XML_CHARS = re.compile(
    '[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
)

def sanitize_text(text) -> str:
    """
    Strip ALL XML-invalid control characters from text.
    Preserves newlines (\n), tabs (\t), and carriage returns (\r).
    Safely handles non-string inputs by converting to str first.
    """
    if text is None:
        return ""
    text = str(text)
    return _INVALID_XML_CHARS.sub('', text)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🔍 VERA Controls")

    # --- Role selector (dynamic) ---
    st.markdown("### 👤 User Role")
    available_roles = get_available_roles()
    if not available_roles:
        available_roles = ["senior"]  # fallback

    def format_role(r):
        if r == "senior": return "🔑 Senior Engineer"
        if r == "junior": return "👤 Junior Intern"
        return f"👤 {r.title()}"

    user_role = st.radio(
        "Select your role:",
        options=available_roles,
        format_func=format_role,
        index=0,
        help="Roles determine access levels (e.g., Senior=Full Access, Junior=Public Only).",
    )

    # --- Domain selector (dynamic) ---
    st.markdown("### 🏢 User Domain")
    available_domains = get_available_domains()
    if not available_domains:
        available_domains = ["semiconductor"]  # fallback
    user_domain = st.radio(
        "Select your domain:",
        options=available_domains,
        index=0,
        help="Your assigned domain. Out-of-domain queries will be escalated.",
    )

    # --- Discussion toggle ---
    st.markdown("### 💬 Agent Discussion")
    enable_discussion = st.checkbox(
        "Enable Collaboration",
        value=True,
        help="Allow agents to critique and refine answers if discrepancies are found.",
    )
    max_refinements = 0  # Refinement loop disabled — too slow on local 1B model

    # --- Backend selector ---
    st.markdown("### ⚙️ LLM Backend")
    backend_options = ["ollama"]
    if GEMINI_API_KEY:
        backend_options.append("gemini")
    if GROQ_API_KEY:
        backend_options.append("groq")

    # Initialize session state for backend tracking
    if "llm_backend" not in st.session_state:
        st.session_state.llm_backend = get_current_backend()

    def format_backend(b):
        if b == "gemini": return "☁️ Gemini API"
        if b == "groq": return "⚡ Groq (Fast)"
        return "🖥️ Ollama (Local)"

    def on_backend_change():
        """Callback when user changes backend radio — runs ONCE, no loop."""
        new_backend = st.session_state._backend_radio
        if new_backend != st.session_state.llm_backend:
            try:
                switch_backend(new_backend)
                st.session_state.llm_backend = new_backend
                st.session_state.graph = None
                st.session_state.graph_ready = False
                st.toast(f"✅ Switched to {format_backend(new_backend)}")
            except ValueError as e:
                st.error(str(e))

    current_idx = backend_options.index(
        st.session_state.llm_backend
    ) if st.session_state.llm_backend in backend_options else 0

    st.radio(
        "Select LLM:",
        options=backend_options,
        format_func=format_backend,
        index=current_idx,
        key="_backend_radio",
        on_change=on_backend_change,
        help="Switch between Ollama (local), Gemini (cloud), or Groq (fast cloud).",
    )

    if st.session_state.llm_backend == "ollama":
        st.caption(f"Model: `{OLLAMA_MODEL}`")

    # --- System status ---
    st.markdown("### 📊 System Status")
    st.success("✅ ChromaDB loaded (in-memory)")

    if st.session_state.graph_ready:
        st.success("✅ VERA agents ready")
    else:
        st.warning("⏳ Graph not compiled yet")

    # --- Example queries ---
    st.markdown("### 💡 Example Queries")

    if user_role == "senior":
        examples = [
            "What is the maximum voltage limit for the RTX-9000?",
            "What are the quality audit procedures, and any recent email changes?",
            "Show me the thermal specs for RTX-9000",
            "Were there any internal emails about skipping burn-in tests?",
            "Compare the RTX-9000 specification versions for any changes",
            "Check the production database for lots that skipped burn-in",
        ]
    else:
        examples = [
            "What is the maximum voltage limit for the RTX-9000?",
            "What are the wafer testing procedures?",
            "What compliance checks are needed before shipping?",
            "Show me the quarterly test summary from the database",
            "What does the RTX-9000 v1.0 specification say?",
        ]

    for ex in examples:
        if st.button(f"📝 {ex[:50]}{'…' if len(ex) > 50 else ''}", key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state.pending_query = ex

    # --- Actions ---
    st.markdown("### 🔧 Actions")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Re-compile Graph", use_container_width=True):
        st.session_state.graph = None
        st.session_state.graph_ready = False
        st.rerun()


# ============================================================================
# BUILD GRAPH (cached in session state)
# ============================================================================
if st.session_state.graph is None:
    with st.spinner("🔧 Compiling VERA agent graph..."):
        try:
            st.session_state.graph = build_graph()
            st.session_state.graph_ready = True
        except Exception as e:
            st.error(f"Failed to compile graph: {e}")

graph = st.session_state.graph


# ============================================================================
# HEADER
# ============================================================================
_backend = st.session_state.get("llm_backend", "ollama")
_badge_map = {
    "gemini": '<span class="badge badge-blue">☁️ Gemini</span>',
    "groq": '<span class="badge badge-amber">⚡ Groq</span>',
}
backend_badge = _badge_map.get(
    _backend, '<span class="badge badge-purple">🖥️ Ollama</span>'
)
role_badge = (
    '<span class="badge badge-green">🔑 Senior</span>'
    if user_role == "senior"
    else '<span class="badge badge-amber">👤 Junior</span>'
)

st.markdown(f"""
<div class="vera-header">
    <h1>🔍 VERA — Virtual Engineering Review Agent</h1>
    <p>Multi-Agent Document Auditing System with RBAC Security</p>
    <div class="badge-row">
        {role_badge}
        {backend_badge}
        <span class="badge badge-green">✅ DB Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# EMAIL DECISION HELPERS  (one per human-decision action)
# ============================================================================

def _send_accept_email(
    query: str, ai_response: str,
    discrepancy_report: str, role: str, domain: str,
):
    """Email sent when user clicks ✅ Accept AI Resolution."""
    body = (
        f"✅ DISCREPANCY RESOLVED — AI Resolution Accepted\n"
        f"{'━' * 50}\n\n"
        f"A discrepancy was detected by VERA and the user has ACCEPTED\n"
        f"the AI-proposed resolution. No further action is required\n"
        f"unless a follow-up audit is warranted.\n\n"
        f"User Role: {role}\n"
        f"Domain:    {domain}\n"
        f"Query:     {query}\n\n"
        f"{'─' * 50}\n"
        f"AI RESOLUTION (accepted)\n"
        f"{'─' * 50}\n"
        f"{ai_response[:1000]}\n\n"
        f"{'─' * 50}\n"
        f"ORIGINAL DISCREPANCY REPORT\n"
        f"{'─' * 50}\n"
        f"{discrepancy_report}\n"
    )
    ok = send_alert_email(f"Accepted — {domain}", body)
    if ok:
        st.toast("📧 Email sent: AI Resolution Accepted")
    else:
        st.warning("⚠️ Email not sent — check .env credentials.")


def _send_reject_email(
    query: str, ai_response: str,
    discrepancy_report: str, role: str, domain: str,
):
    """Email sent when user clicks ❌ Reject & Override."""
    body = (
        f"❌ AI RESOLUTION REJECTED — Manual Override Requested\n"
        f"{'━' * 50}\n\n"
        f"A discrepancy was detected by VERA but the user has REJECTED\n"
        f"the AI-proposed resolution and is requesting a manual override.\n"
        f"Human review and correction of the affected records is required.\n\n"
        f"User Role: {role}\n"
        f"Domain:    {domain}\n"
        f"Query:     {query}\n\n"
        f"{'─' * 50}\n"
        f"AI RESPONSE (rejected)\n"
        f"{'─' * 50}\n"
        f"{ai_response[:1000]}\n\n"
        f"{'─' * 50}\n"
        f"DISCREPANCY REPORT\n"
        f"{'─' * 50}\n"
        f"{discrepancy_report}\n\n"
        f"⚠️  ACTION REQUIRED: Please review the above discrepancy and\n"
        f"provide a corrected resolution or update the source records.\n"
    )
    ok = send_alert_email(f"Rejected & Override — {domain}", body)
    if ok:
        st.toast("📧 Email sent: Rejection & Override Requested")
    else:
        st.warning("⚠️ Email not sent — check .env credentials.")


def _send_escalate_email(
    query: str, ai_response: str,
    discrepancy_report: str, role: str, domain: str,
):
    """Email sent when user clicks 🛡️ Escalate to Safety Team."""
    body = (
        f"🚨 URGENT — ESCALATED TO SAFETY TEAM\n"
        f"{'━' * 50}\n\n"
        f"A discrepancy detected by VERA has been ESCALATED for formal\n"
        f"safety investigation. Immediate review by the safety team is\n"
        f"requested.\n\n"
        f"User Role: {role}\n"
        f"Domain:    {domain}\n"
        f"Query:     {query}\n\n"
        f"{'─' * 50}\n"
        f"AI RESPONSE (under review)\n"
        f"{'─' * 50}\n"
        f"{ai_response[:1000]}\n\n"
        f"{'─' * 50}\n"
        f"FULL DISCREPANCY REPORT\n"
        f"{'─' * 50}\n"
        f"{discrepancy_report}\n\n"
        f"🛡️  ACTION REQUIRED: Safety team — please initiate a formal\n"
        f"investigation into the above discrepancy and respond with\n"
        f"findings within the SLA window.\n"
    )
    ok = send_alert_email(f"URGENT: Safety Escalation — {domain}", body)
    if ok:
        st.toast("📧 Email sent: Escalated to Safety Team")
    else:
        st.warning("⚠️ Email not sent — check .env credentials.")


# ============================================================================
# CHAT HISTORY DISPLAY
# ============================================================================
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        # ---- User message ----
        if msg["role"] == "user":
            st.markdown(sanitize_text(msg["content"]))
        # ---- Assistant message ----
        else:
            st.markdown(sanitize_text(msg["content"]))

            # -- Escalation alert --
            if msg.get("escalated"):
                st.markdown("""
                <div class="escalation-alert">
                    <h3>🚨 Escalation Triggered</h3>
                    <p>This query was flagged for a potential access control violation.
                       A supervisor review has been requested.</p>
                </div>
                """, unsafe_allow_html=True)

            # Expandable panels
            col1, col2 = st.columns(2)

            # -- Retrieved documents --
            docs = msg.get("documents", [])
            if docs:
                with col1:
                    with st.expander(f"📄 Retrieved Documents ({len(docs)})", expanded=False):
                        for i, doc_info in enumerate(docs):
                            source_icon = {"datasheet": "📊", "email": "📧", "sop": "📋", "db_info": "🗄️", "document": "📑"}.get(doc_info["source"], "📄")
                            access_color = {
                                "public": "🟢",
                                "internal_only": "🟡",
                                "confidential": "🔴",
                            }.get(doc_info["access_level"], "⚪")

                            st.markdown(f"""
                            <div class="doc-card">
                                <div class="doc-header">
                                    <span class="doc-title">{source_icon} {doc_info.get('title', doc_info['source'].upper())}</span>
                                    <span>{access_color} {doc_info['access_level']}</span>
                                </div>
                                <div>ID: <code>{doc_info.get('doc_id', 'N/A')}</code></div>
                                <div class="doc-content">{sanitize_text(doc_info['preview'])}</div>
                            </div>
                            """, unsafe_allow_html=True)

            # -- Metadata / Audit log --
            if msg.get("metadata_log"):
                with col2:
                    with st.expander("📋 Metadata & Audit Log", expanded=False):
                        st.code(sanitize_text(msg["metadata_log"]), language="text")

            # -- Discrepancy report --
            if msg.get("discrepancy_report"):
                # Check if there's an actual discrepancy or just an aligned audit
                has_real_conflict = msg.get("has_discrepancy", False)
                
                if has_real_conflict:
                    st.error("🚨 **CONFLICT DETECTED — HUMAN REVIEW REQUIRED**")
                else:
                    st.info("✅ **AUDIT COMPLETE — NO CONFLICTS FOUND**")
                
                with st.expander("📋 View Discrepancy Report", expanded=has_real_conflict):
                    st.markdown(sanitize_text(msg["discrepancy_report"]))
                    
                    if has_real_conflict:
                        st.markdown("---")
                        st.write("**👨‍⚖️ Human Decision Required:**")

                        # Find the user query for this assistant message
                        _query = ""
                        for _prev_idx in range(idx - 1, -1, -1):
                            if st.session_state.messages[_prev_idx]["role"] == "user":
                                _query = st.session_state.messages[_prev_idx]["content"]
                                break

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("✅ Accept AI Resolution", key=f"btn_acc_{idx}"):
                                _send_accept_email(
                                    _query, msg["content"],
                                    msg["discrepancy_report"], user_role, user_domain,
                                )
                        with c2:
                            if st.button("❌ Reject & Override", key=f"btn_rej_{idx}"):
                                _send_reject_email(
                                    _query, msg["content"],
                                    msg["discrepancy_report"], user_role, user_domain,
                                )
                        with c3:
                            if st.button("🛡️ Escalate to Safety Team", key=f"btn_esc_{idx}"):
                                _send_escalate_email(
                                    _query, msg["content"],
                                    msg["discrepancy_report"], user_role, user_domain,
                                )

            # Show sources
            if msg.get("sources"):
                with st.expander(f"📚 References ({len(msg['sources'])})", expanded=True):
                    for doc_idx, doc in enumerate(msg["sources"]):
                        st.markdown(sanitize_text(f"**{doc_idx+1}. {doc['source'].upper()}**: {doc['title']}"))
                        summary = doc['content'][:500] + ("..." if len(doc['content']) > 500 else "")
                        st.markdown(f'<div class="doc-summary">{sanitize_text(summary)}</div>', unsafe_allow_html=True)

            # -- Agent trace --
            if msg.get("agent_trace"):
                with st.expander("🔗 Agent Execution Trace", expanded=False):
                    for step in msg["agent_trace"]:
                        css_class = step.get("css", "route")
                        st.markdown(
                            f'<div class="agent-step {css_class}">{step["icon"]} <b>{step["name"]}</b> — {step["detail"]}</div>',
                            unsafe_allow_html=True,
                        )


# ============================================================================
# CHAT INPUT HANDLER
# ============================================================================

def process_query(question: str, role: str, domain: str, max_refinements: int = 1):
    """Run a query through the VERA graph and return structured results."""
    initial_state = {
        "question": question,
        "generation": "",
        "user_role": role,
        "user_domain": domain,
        "documents": [],
        "route": "",
        "intent": "",
        "flagged": False,
        "metadata_log": "",
        "retrieved_docs": {},
        "db_result": "",
        "db_data": "",
        "discrepancy_report": "",
        "next_agent": "",
        "thought_process": [],
        "refinement_count": 0,
        "max_refinements": max_refinements,
        "critique": "",
        "retrieval_confidence": "",
        # Structured fact passing fields
        "target_entity": "",
        "entity_type": "",
        "target_attribute": "",
        "time_context": "",
        "official_facts": [],
        "informal_facts": [],
        "db_facts": [],
        "discrepancy_verdict": {},
        "official_data": [],
        "informal_data": [],
        "latest_timestamp": "",
    }

    final_state = initial_state.copy()
    
    # Use st.status to create a collapsible "Thinking..." container
    with st.status("🧠 VERA Agents Thinking...", expanded=True) as status:
        # Stream updates from the graph explicitly in "updates" mode
        for event in graph.stream(initial_state, stream_mode="updates"):
            # Debug: print event to console to see why updates might be None
            # print(f"DEBUG: Event from graph: {event}")
            
            # sanitize_text is now global

            for node_name, updates in event.items():
                if updates is None:
                    # Logging more details to terminal for back-end debugging
                    print(f"⚠️ Warning: Node '{node_name}' yielded None in stream event: {event}")
                    continue
                
                # Update UI Status with meaningful progress
                if "retrieve" in node_name:
                    status.update(label=f"🔍 Retrieved {len(updates.get('documents', []))} documents", state="running")
                    st.write(sanitize_text(f"Found {len(updates.get('documents', []))} sources..."))
                elif "generate_response" in node_name:
                    status.update(label="✏️ Drafting response...", state="running")
                    st.write(sanitize_text("Synthesizing answer..."))
                elif "check_discrepancy" in node_name:
                    if updates.get("critique"):
                        status.update(label="⚠️ Conflict detected - Refining...", state="running")
                        st.write(sanitize_text("Found discrepancy. Refining answer..."))
                    else:
                        status.update(label="✅ Quality Check passed", state="running")
                        st.write(sanitize_text("No material conflicts found."))
                    
                # Update our local final state
                final_state.update(updates)
        
        status.update(label="✅ Complete", state="complete", expanded=False)
                
    # --- Centralized Logging ---
    log_vera_step(final_state, domain)
                


    # Build structured output
    generation = final_state.get("generation", "No response generated.")
    flagged = final_state.get("flagged", False)
    route = final_state.get("route", "unknown")
    metadata_log = final_state.get("metadata_log", "")
    docs = final_state.get("documents", [])
    discrepancy_report = final_state.get("discrepancy_report", "")

    # Extract document info
    doc_infos = []
    for d in docs:
        doc_infos.append({
            "source": d.metadata.get("source", "unknown"),
            "title": d.metadata.get("title", "Untitled"),
            "content": d.page_content[:200]
        })

    # Build agent trace for UI
    agent_trace = []
    
    # 1. Router / Escalation
    agent_trace.append({
        "name": "Router Agent",
        "detail": f"Intent: {route} | Flagged: {flagged}",
        "icon": "🔀",
        "css": "route",
    })
    
    if flagged:
        agent_trace.append({
            "name": "Escalation Handler",
            "detail": "Query escalated to supervisor",
            "icon": "🚨",
            "css": "escalate",
        })
    else:
        # 2. Retrieval Agents
        if route == "technical":
            agent_trace.append({
                "name": "Technical Spec Agent",
                "detail": f"Retrieved {len(docs)} datasheet documents",
                "icon": "📊",
                "css": "retrieve",
            })
        else:
            agent_trace.append({
                "name": "Compliance Agent",
                "detail": f"Retrieved {len(docs)} compliance documents",
                "icon": "📋",
                "css": "retrieve",
            })
            
        # 3. Response Generator
        agent_trace.append({
            "name": "Response Generator",
            "detail": f"Generated {len(generation)} chars",
            "icon": "🤖",
            "css": "generate",
        })
        
        # 4. Discrepancy Detector
        if discrepancy_report:
            agent_trace.append({
                "name": "Discrepancy Detector",
                "detail": "⚠️ Conflict Detected!",
                "icon": "🔍",
                "css": "discrep",
            })
        else:
            agent_trace.append({
                "name": "Discrepancy Detector",
                "detail": "No conflicts found",
                "icon": "✅",
                "css": "generate",
            })

    # Determine if there's a real discrepancy (not just an aligned audit)
    discrepancy_verdict = final_state.get("discrepancy_verdict", {})
    has_discrepancy = False
    if discrepancy_verdict:
        status = discrepancy_verdict.get("overall_status", "ALIGNED")
        has_discrepancy = status == "DISCREPANCY"

    return {
        "role": "assistant",
        "content": generation,
        "sources": doc_infos,
        "metadata_log": metadata_log,
        "flagged": flagged,
        "route": route,
        "discrepancy_report": discrepancy_report,
        "has_discrepancy": has_discrepancy,
        "agent_trace": agent_trace,
        "thought_process": final_state.get("thought_process", []),
    }


# ---- Handle pending query from example buttons ----
if "pending_query" in st.session_state:
    pending = st.session_state.pending_query
    del st.session_state.pending_query
    # Append user message
    st.session_state.messages.append({"role": "user", "content": pending})
    # Process
    if graph:
        # Spinner removed; process_query uses st.status
        result = process_query(pending, user_role, user_domain, max_refinements)
        result["role"] = "assistant"
        st.session_state.messages.append(result)
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ VERA graph not ready. Please check the console for errors.",
        })
    st.rerun()

# ---- Handle manual chat input ----
if prompt := st.chat_input("Ask VERA a question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not graph:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ VERA graph not ready. Please check the console for errors.",
        })
        st.rerun()

    # Process query through VERA
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        # Spinner removed; process_query uses st.status
        start_time = time.time()
        result = process_query(prompt, user_role, user_domain, max_refinements)
        elapsed = time.time() - start_time

        st.markdown(sanitize_text(result["content"]))
        st.caption(f"⏱️ Processed in {elapsed:.1f}s")
        
        # Discrepancy Alert with Human Decision
        if result.get("discrepancy_report"):
            has_real_conflict = result.get("has_discrepancy", False)
            
            if has_real_conflict:
                st.error("🚨 **CONFLICT DETECTED — HUMAN REVIEW REQUIRED**")
            else:
                st.info("✅ **AUDIT COMPLETE — NO CONFLICTS FOUND**")
            
            with st.expander("📋 View Discrepancy Report", expanded=has_real_conflict):
                st.markdown(sanitize_text(result["discrepancy_report"]))
                
                if has_real_conflict:
                    st.markdown("---")
                    st.write("**👨‍⚖️ Human Decision Required:**")

                    msg_idx = len(st.session_state.messages)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("✅ Accept AI Resolution", key=f"btn_acc_{msg_idx}"):
                            _send_accept_email(
                                prompt,
                                result["content"],
                                result["discrepancy_report"],
                                user_role,
                                user_domain,
                            )
                    with c2:
                        if st.button("❌ Reject & Override", key=f"btn_rej_{msg_idx}"):
                            _send_reject_email(
                                prompt,
                                result["content"],
                                result["discrepancy_report"],
                                user_role,
                                user_domain,
                            )
                    with c3:
                        if st.button("🛡️ Escalate to Safety Team", key=f"btn_esc_{msg_idx}"):
                            _send_escalate_email(
                                prompt,
                                result["content"],
                                result["discrepancy_report"],
                                user_role,
                                user_domain,
                            )

        # Show thought process (DeepSeek-style)
        thought_process = result.get("thought_process", [])
        if thought_process:
            with st.expander(f"🧠 Agent Thinking ({len(thought_process)} steps)", expanded=False):
                for step in thought_process:
                    st.markdown(f"- {sanitize_text(step)}")

        # Show sources
        if result.get("sources"):
            with st.expander(f"📚 References ({len(result['sources'])})", expanded=True):
                for idx, doc in enumerate(result["sources"]):
                    st.markdown(sanitize_text(f"**{idx+1}. {doc['source'].upper()}**: {doc['title']}"))
                    summary = doc['content'][:500] + ("..." if len(doc['content']) > 500 else "")
                    st.markdown(f'<div class="doc-summary">{sanitize_text(summary)}</div>', unsafe_allow_html=True)

    # Save to history
    result["role"] = "assistant"
    st.session_state.messages.append(result)
    st.rerun()


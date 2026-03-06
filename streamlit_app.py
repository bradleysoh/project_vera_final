import os
import sys
import time
import datetime
import re
import streamlit as st
from dotenv import load_dotenv

# Import your custom Kaggle loader
from data_loader import load_all_industries

# ---------------------------------------------------------------------------
# Ensure project modules are importable
# ---------------------------------------------------------------------------
load_dotenv()

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
# HELPERS
# ============================================================================
_INVALID_XML_CHARS = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

def sanitize_text(text) -> str:
    if text is None: return ""
    text = str(text)
    return _INVALID_XML_CHARS.sub('', text)

def log_agent(agent: str, action: str):
    if "agent_trace_log" not in st.session_state:
        st.session_state.agent_trace_log = []
    st.session_state.agent_trace_log.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "action": action
    })

# ============================================================================
# PAGE CONFIG & CSS
# ============================================================================
st.set_page_config(
    page_title="VERA — Virtual Engineering Review Agent",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .vera-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.08);
    }
    .badge-row { display: flex; gap: 0.5rem; margin-top: 0.8rem; }
    .badge { padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .badge-green { background: rgba(64,192,87,0.25); color: #69db7c; }
    .badge-blue { background: rgba(77,171,247,0.25); color: #74c0fc; }
    .badge-amber { background: rgba(255,183,77,0.25); color: #ffd43b; }
    .agent-step { background: rgba(255,255,255,0.04); border-left: 3px solid #4dabf7; padding: 0.6rem; margin: 0.3rem 0; border-radius: 0 8px 8px 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & DATA INGESTION
# ============================================================================
if "messages" not in st.session_state: st.session_state.messages = []
if "graph" not in st.session_state: st.session_state.graph = None
if "graph_ready" not in st.session_state: st.session_state.graph_ready = False
if "agent_trace_log" not in st.session_state: st.session_state.agent_trace_log = []

# Automatically trigger the Kaggle data loader on first run
if "data_loaded" not in st.session_state:
    with st.spinner("📥 Ingesting industry datasets..."):
        try:
            load_all_industries()
            st.session_state.data_loaded = True
            st.toast("✅ Industry datasets ready!")
        except Exception as e:
            st.error(f"Data Load Error: {e}")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 👤 User Profile")
    user_role = st.radio("Role:", get_available_roles() or ["senior"])
    
    # Updated Domain Selector including your new industries
    st.markdown("### 🏢 User Domain")
    user_domain = st.radio(
        "Select your domain:",
        options=["semiconductor", "medical", "aerospace", "energy", "finance"],
        index=0,
        help="Select the specific industry dataset for VERA to audit."
    )
    
    st.markdown("### ⚙️ LLM Backend")
    backend_options = ["ollama"]
    if GEMINI_API_KEY: backend_options.append("gemini")
    if GROQ_API_KEY: backend_options.append("groq")
    
    if "llm_backend" not in st.session_state:
        st.session_state.llm_backend = get_current_backend()

    def on_backend_change():
        new_backend = st.session_state._backend_radio
        switch_backend(new_backend)
        st.session_state.llm_backend = new_backend
        st.session_state.graph = None
        st.session_state.graph_ready = False

    st.radio("Select LLM:", backend_options, key="_backend_radio", on_change=on_backend_change)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# GRAPH COMPILATION
# ============================================================================
if st.session_state.graph is None:
    try:
        st.session_state.graph = build_graph()
        st.session_state.graph_ready = True
    except Exception as e:
        st.error(f"Graph Error: {e}")

# ============================================================================
# CORE LOGIC: PROCESS QUERY
# ============================================================================
def process_query(question: str, role: str, domain: str):
    st.session_state.agent_trace_log = [] # Reset timeline for new query
    
    initial_state = {
        "question": question, 
        "user_role": role, 
        "user_domain": domain,
        "documents": [], 
        "generation": "", 
        "metadata_log": "", 
        "discrepancy_report": "",
        "route": "technical", 
        "flagged": False, 
        "thought_process": []
    }

    final_state = initial_state.copy()
    
    with st.status(f"🧠 VERA Auditing {domain.title()} Data...", expanded=True) as status:
        log_agent("Router Agent", f"Classified intent for {domain} domain")
        for event in st.session_state.graph.stream(initial_state, stream_mode="updates"):
            for node_name, updates in event.items():
                if updates:
                    if "retrieve" in node_name:
                        status.update(label=f"🔍 Searching {domain} sources...")
                        log_agent("Retrieval Agent", f"Accessed {domain} documents")
                    elif "generate" in node_name:
                        status.update(label="✏️ Drafting response...")
                    elif "discrepancy" in node_name:
                        status.update(label="⚖️ Auditing conflicts...")
                        log_agent("Discrepancy Agent", "Comparing cross-document values")
                    
                    final_state.update(updates)
        
        status.update(label="✅ Audit Complete", state="complete", expanded=False)

    return {
        "role": "assistant",
        "content": final_state.get("generation", "No output produced."),
        "documents": final_state.get("documents", []),
        "discrepancy_report": final_state.get("discrepancy_report", ""),
        "has_discrepancy": final_state.get("discrepancy_verdict", {}).get("overall_status") == "DISCREPANCY",
        "agent_trace_log": st.session_state.agent_trace_log.copy()
    }

# ============================================================================
# UI: HEADER & CHAT
# ============================================================================
st.markdown(f'<div class="vera-header"><h1>🔍 VERA</h1><p>Multi-Domain Agentic Review Agent</p></div>', unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("discrepancy_report"):
                with st.expander("📋 View Discrepancy Report"):
                    st.markdown(msg["discrepancy_report"])
            if msg.get("documents"):
                with st.expander(f"📚 References ({len(msg['documents'])})"):
                    for d in msg["documents"]:
                        st.markdown(f"**Source:** {d.metadata.get('source', 'N/A')} | **ID:** {d.metadata.get('doc_id', 'N/A')}")

# User Input
if prompt := st.chat_input(f"Ask VERA about {user_domain} data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        start_time = time.time()
        result = process_query(prompt, user_role, user_domain)
        elapsed = time.time() - start_time
        
        st.markdown(result["content"])
        st.caption(f"⏱️ Processed in {elapsed:.1f}s")
        st.session_state.messages.append(result)
        st.rerun()

# Execution Timeline in Sidebar
if st.session_state.agent_trace_log:
    with st.sidebar.expander("🔎 Agent Execution Trace", expanded=True):
        for step in st.session_state.agent_trace_log:
            st.write(f"**{step['time']}** | {step['agent']} → {step['action']}")
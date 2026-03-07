import os
from typing import List, TypedDict
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
# FIX: Using langchain_core to avoid the ModuleNotFoundError
from langchain_core.prompts import PromptTemplate 
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
from shared.config import get_embeddings
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

class GraphState(TypedDict):
    question: str
    user_role: str
    user_domain: str
    documents: List
    generation: str

# ============================================================================
# NODES
# ============================================================================

def retrieve_docs(state: GraphState):
    """Retrieves documents filtered by the selected domain."""
    domain = state.get("user_domain", "general").lower()
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name="vera_documents"
    )

    # Filtering by domain to ensure industry-specific accuracy
    search_results = vectorstore.similarity_search(
        state["question"], 
        k=4, 
        filter={"domain": domain} 
    )
    return {"documents": search_results}

def generate_answer(state: GraphState):
    """Feeds retrieved text into the LLM for a real answer (No more placeholders)."""
    docs = state["documents"]
    if not docs:
        return {"generation": "I couldn't find any relevant data in the files."}

    # Combine file content into context
    context = "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in docs])

    template = """You are VERA. Use the context below to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""

    prompt = PromptTemplate.from_template(template)
    # Ensure this model name matches your 'ollama list' output
    llm = Ollama(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest")
    
    chain = prompt | llm
    generation = chain.invoke({"context": context, "question": state["question"]})

    return {"generation": generation}

# ============================================================================
# GRAPH CONSTRUCTION (Fixes the ImportError)
# ============================================================================

def build_graph():
    """Compiles the workflow for Streamlit."""
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
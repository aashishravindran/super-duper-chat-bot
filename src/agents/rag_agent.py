"""
RAG pipeline subgraph: retrieve → rerank → generate.
Each step is its own node — easy to trace individually in LangSmith.
"""
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.config import get_llm
from src.rag.ingest import load_indexes
from src.rag.retriever import hybrid_retrieve
from src.rag.reranker import rerank
from src.state import State

# Lazy-loaded — populated on first RAG query so graph can be built/visualized
# before `scripts/ingest_resume.py` has been run.
_vectorstore = _bm25 = _texts = None


def _get_indexes():
    global _vectorstore, _bm25, _texts
    if _vectorstore is None:
        _vectorstore, _bm25, _texts = load_indexes()
    return _vectorstore, _bm25, _texts

SYSTEM_PROMPT = """You are a resume assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the resume."

Context:
{context}
"""


def retrieve_node(state: State) -> dict:
    query = state["messages"][-1].content
    vs, bm25, texts = _get_indexes()
    candidates = hybrid_retrieve(query, vs, bm25, texts, k=10)
    return {"rag_context": candidates}


def rerank_node(state: State) -> dict:
    query = state["messages"][-1].content
    top_chunks = rerank(query, state["rag_context"], top_k=4)
    return {"rag_context": top_chunks}


def generate_node(state: State) -> dict:
    query = state["messages"][-1].content
    context = "\n\n".join(c["text"] for c in state["rag_context"])
    prompt = SYSTEM_PROMPT.format(context=context)
    response = get_llm().invoke([SystemMessage(content=prompt), *state["messages"]])
    return {"messages": [response]}


def build_rag_subgraph():
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)

    return builder.compile()

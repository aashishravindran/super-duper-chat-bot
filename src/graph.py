"""
Top-level graph: wires the supervisor and two subgraphs together.

build_graph()             → no checkpointer; for LangGraph Studio (uses asyncio.run at import)
build_graph_with_memory() → async; for FastAPI — awaits MCP init inside the event loop
"""
import asyncio
import concurrent.futures

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from src.agents.assistant import call_assistant, route_assistant, verify_answer, route_verifier
from src.agents.github_agent import build_github_subgraph
from src.agents.rag_agent import build_rag_subgraph
from src.state import State


def _builder(github_subgraph) -> StateGraph:
    rag_subgraph = build_rag_subgraph()

    builder = StateGraph(State)
    builder.add_node("assistant", call_assistant)
    builder.add_node("github_agent", github_subgraph)
    builder.add_node("rag_agent", rag_subgraph)
    builder.add_node("verifier", verify_answer)

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        route_assistant,
        {"github_agent": "github_agent", "rag_agent": "rag_agent", "end": END},
    )

    builder.add_edge("github_agent", "verifier")
    builder.add_edge("rag_agent", "verifier")

    builder.add_conditional_edges(
        "verifier",
        route_verifier,
        {"github_agent": "github_agent", "rag_agent": "rag_agent", "end": END},
    )

    return builder


def _run_async(coro):
    """Run a coroutine safely regardless of whether an event loop is already running.
    Spawns a worker thread with its own loop — avoids the asyncio.run() restriction."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def build_graph():
    """LangGraph Studio — Studio may or may not have a running loop at import time."""
    github_subgraph = _run_async(build_github_subgraph())  # coroutine created here, run in thread
    return _builder(github_subgraph).compile()


async def build_graph_with_memory():
    """FastAPI — awaited inside uvicorn's event loop via lifespan."""
    github_subgraph = await build_github_subgraph()
    return _builder(github_subgraph).compile(checkpointer=MemorySaver())

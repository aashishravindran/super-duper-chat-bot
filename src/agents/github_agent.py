"""
GitHub ReAct subgraph: llm ↔ tools loop using StateGraph + ToolNode.
"""
import os
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import get_llm
from src.state import State

SYSTEM_PROMPT = SystemMessage(content="""You are a GitHub assistant for the user 'aashishravindran'.

When looking up repos or files, use list/get tools with owner='aashishravindran' rather than
search queries — GitHub's search API restricts user-scoped searches.

Steps for a typical repo question:
1. List repos for owner 'aashishravindran'
2. Get the specific repo or file contents directly

Always cite the repo name and file path in your answer.
""")

_MCP_CONFIG = {
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
        "transport": "stdio",
    }
}


@tool
def github_unavailable(query: str) -> str:
    """Placeholder used when the GitHub MCP server is not reachable."""
    return "GitHub MCP server is not configured. Set GITHUB_TOKEN and ensure Node.js is installed."


def _make_github_node(llm):
    """Return a node function with the LLM baked in — avoids closures inside async scope."""
    def call_github_llm(state: State) -> dict:
        return {"messages": [llm.invoke([SYSTEM_PROMPT] + state["messages"])]}
    call_github_llm.__name__ = "call_github_llm"
    return call_github_llm


async def build_github_subgraph():
    try:
        client = MultiServerMCPClient(_MCP_CONFIG)
        tools = await client.get_tools()
        if not tools:
            raise ValueError("No tools returned from MCP server")
    except Exception as e:
        print(f"[github_agent] MCP unavailable ({e}), using stub tool.")
        tools = [github_unavailable]

    llm = get_llm().bind_tools(tools)

    builder = StateGraph(State)
    builder.add_node("github_llm", _make_github_node(llm))
    builder.add_node("github_tools", ToolNode(tools))

    builder.add_edge(START, "github_llm")
    builder.add_conditional_edges(
        "github_llm",
        tools_condition,
        {"tools": "github_tools", END: END},
    )
    builder.add_edge("github_tools", "github_llm")

    return builder.compile()

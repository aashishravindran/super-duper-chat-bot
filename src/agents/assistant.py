"""
Supervisor node + router + verifier.

Routing: single LLM call with structured output → "github_agent" | "rag_agent" | "direct"
Verifier: separate node — checks if the subagent actually answered the query before END.
interrupt() only fires when the router is genuinely unsure (confidence == "low").
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import BaseModel
from typing import Literal

from src.config import get_llm
from src.state import State


# --- Structured output schemas ---

class RouteDecision(BaseModel):
    route: Literal["github_agent", "rag_agent", "resume_agent", "direct"]
    confidence: Literal["high", "low"]


class VerificationDecision(BaseModel):
    sufficient: bool  # did the agent response actually answer the query?


_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant. Decide which agent should handle the user's message.

- "github_agent"  → anything about code, repositories, projects, commits, files, or a specific repo by name
- "rag_agent"     → anything about the resume: work history, skills, education, experience, background
- "resume_agent"  → explicit request to rewrite, tailor, or update the resume for a job (includes a job description or URL)
- "direct"        → general conversation, greetings, clarifications, or anything else

Also set confidence:
- "high" → you are sure about the route
- "low"  → the query is ambiguous and a human should confirm

Examples:
  "Tell me about Sentinel"                          → github_agent, high
  "what repos do you have?"                         → github_agent, high
  "what's your work experience?"                    → rag_agent, high
  "rewrite my resume for this job: <url>"           → resume_agent, high
  "tailor my resume for a senior ML engineer role"  → resume_agent, high
  "hi there"                                        → direct, high
  "tell me about projects"                          → github_agent, low
"""),
    ("placeholder", "{messages}"),
])

_VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a quality checker. Given a user's query and an agent's response, decide if the response sufficiently answers the query.
sufficient=true  → the response addresses the query, even partially or with caveats
sufficient=false → the response is completely off-topic, refuses to answer, or returns an error

Default to sufficient=true when in doubt. Only return false if the response clearly failed."""),
    ("human", "Query: {query}\n\nAgent response: {answer}"),
])

_routing_llm = get_llm(temperature=0).with_structured_output(RouteDecision)
_verifier_llm = get_llm(temperature=0).with_structured_output(VerificationDecision)
_router = _ROUTER_PROMPT | _routing_llm
_verifier = _VERIFIER_PROMPT | _verifier_llm

_llm = get_llm()


# --- Node ---

async def call_assistant(state: State, config: RunnableConfig) -> dict:
    role = config.get("configurable", {}).get("role", "user")
    decision: RouteDecision = await _router.ainvoke({"messages": state["messages"]})

    # Non-admin users cannot access resume_agent
    if decision.route == "resume_agent" and role != "admin":
        response = await _llm.ainvoke([
            SystemMessage(content="You are a helpful assistant. Politely explain that resume rewriting is only available to the owner of this assistant, not to external visitors."),
            *state["messages"],
        ])
        return {"messages": [response], "active_agent": "end"}

    if decision.confidence == "low":
        user_response = interrupt({
            "question": "I'm not sure which agent to use — could you clarify?",
            "suggested_route": decision.route,
        })
        if user_response in ("github_agent", "rag_agent", "resume_agent"):
            decision.route = user_response

    if decision.route == "direct":
        response = await _llm.ainvoke([
            SystemMessage(content="You are a helpful personal assistant."),
            *state["messages"],
        ])
        return {"messages": [response], "active_agent": "end"}

    return {"active_agent": decision.route}


async def verify_answer(state: State) -> dict:
    """Critic node — re-routes once only if the subagent clearly failed (not just partial)."""
    attempts = state.get("route_attempts", 0)
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    query = human_msgs[-1].content
    answer = state["messages"][-1].content

    check = await _verifier.ainvoke({"query": query, "answer": answer})
    print(f"[verifier] sufficient={check.sufficient} attempts={attempts} query={query!r:.60}")

    if not check.sufficient and attempts < 1:
        return {"active_agent": state["active_agent"], "route_attempts": attempts + 1}

    return {"active_agent": "end", "route_attempts": 0}


# --- Edge functions ---

def route_assistant(state: State) -> str:
    return state.get("active_agent") or "end"


def route_verifier(state: State) -> str:
    return state.get("active_agent") or "end"

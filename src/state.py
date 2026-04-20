from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class State(MessagesState):
    """
    Extends MessagesState so LangGraph Studio recognises this as a chat graph
    and wraps user input as HumanMessage automatically.

    MessagesState already provides:
        messages: Annotated[list[AnyMessage], add_messages]
    """
    rag_context: list[dict] | None
    active_agent: str | None  # set before delegating; still set on return → synthesis signal
    route_attempts: int  # tracks retries to prevent infinite verification loops

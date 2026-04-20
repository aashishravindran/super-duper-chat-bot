"""
FastAPI layer: two endpoints.
  POST /chat          — start a conversation or resume after an interrupt
  POST /chat/cancel   — cancel a pending interrupt (user said "no")

Responses are Server-Sent Events (SSE) so the client sees tokens as they arrive.
"""
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.graph import build_graph_with_memory

GRAPH = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GRAPH
    GRAPH = await build_graph_with_memory()
    yield


app = FastAPI(title="Multi-Agent Assistant", lifespan=lifespan)


# --- Request / Response models ---

class ChatRequest(BaseModel):
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str | None = None
    resume: str | None = None


class ChatResponse(BaseModel):
    thread_id: str  # always returned so the client can continue the conversation


class CancelRequest(BaseModel):
    thread_id: str


# --- Endpoints ---

@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    config = {"configurable": {"thread_id": req.thread_id}}

    graph_input = (
        Command(resume=req.resume)
        if req.resume is not None
        else {"messages": [HumanMessage(content=req.message)]}
    )

    async def event_stream():
        # First event: always send the thread_id so client knows what to use next
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': req.thread_id})}\n\n"

        async for chunk in GRAPH.astream(
            graph_input, config, stream_mode=["messages", "custom"]
        ):
            mode, data = chunk

            if mode == "messages":
                token, meta = data
                if token.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': token.content, 'node': meta.get('langgraph_node', '')})}\n\n"

            elif mode == "custom":
                yield f"data: {json.dumps({'type': 'event', **data})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/cancel")
async def cancel(req: CancelRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    await GRAPH.ainvoke(Command(resume="cancel"), config)
    return {"status": "cancelled", "thread_id": req.thread_id}

"""
FastAPI layer: two endpoints.
  POST /chat          — start a conversation or resume after an interrupt
  POST /chat/cancel   — cancel a pending interrupt (user said "no")

Responses are Server-Sent Events (SSE) so the client sees tokens as they arrive.
Each SSE payload is a discriminated union on the `type` field.
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Union

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.graph import build_graph_with_memory

GRAPH = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global GRAPH
    GRAPH = await build_graph_with_memory()
    yield


app = FastAPI(title="Multi-Agent Assistant", lifespan=lifespan)


# ---------------------------------------------------------------------------
# SSE event models — discriminated union on `type`
# ---------------------------------------------------------------------------

class ThreadIdEvent(BaseModel):
    type: Literal["thread_id"] = "thread_id"
    thread_id: str


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str
    node: str


class LatencyEvent(BaseModel):
    type: Literal["latency"] = "latency"
    node: str
    latency_ms: int


class CustomEvent(BaseModel):
    type: Literal["event"] = "event"
    data: dict


class InterruptEvent(BaseModel):
    type: Literal["interrupt"] = "interrupt"
    question: str
    suggested_route: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


ChatEvent = Annotated[
    Union[ThreadIdEvent, TokenEvent, LatencyEvent, CustomEvent, InterruptEvent, ErrorEvent],
    Field(discriminator="type"),
]


def _sse(event: BaseModel) -> str:
    return f"data: {event.model_dump_json()}\n\n"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str | None = None
    resume: str | None = None


class CancelRequest(BaseModel):
    thread_id: str


class CancelResponse(BaseModel):
    status: Literal["cancelled"]
    thread_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    config = {"configurable": {"thread_id": req.thread_id}}

    graph_input = (
        Command(resume=req.resume)
        if req.resume is not None
        else {"messages": [HumanMessage(content=req.message)]}
    )

    async def event_stream():
        yield _sse(ThreadIdEvent(thread_id=req.thread_id))

        node_start_times: dict[str, float] = {}
        node_latencies: dict[str, int] = {}
        streamed_ids: set[str] = set()
        request_start = time.perf_counter()

        try:
            async for chunk in GRAPH.astream(
                graph_input, config, stream_mode=["messages", "custom", "updates"]
            ):
                mode, data = chunk

                if mode == "messages":
                    token, meta = data
                    msg_id = getattr(token, "id", None)
                    if msg_id and msg_id in streamed_ids:
                        continue
                    if msg_id:
                        streamed_ids.add(msg_id)
                    node = meta.get("langgraph_node", "")
                    if node and node not in node_start_times:
                        node_start_times[node] = time.perf_counter()
                    raw = token.content
                    if isinstance(raw, list):
                        content = " ".join(
                            block["text"] for block in raw
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        content = raw
                    if content:
                        yield _sse(TokenEvent(content=content, node=node))

                elif mode == "updates":
                    for node_name in data:
                        if node_name in node_start_times:
                            node_latencies[node_name] = round(
                                (time.perf_counter() - node_start_times[node_name]) * 1000
                            )

                elif mode == "custom":
                    if data.get("type") == "interrupt":
                        yield _sse(InterruptEvent(
                            question=data.get("question", ""),
                            suggested_route=data.get("suggested_route"),
                        ))
                    else:
                        yield _sse(CustomEvent(data=data))

            for node_name, latency_ms in node_latencies.items():
                yield _sse(LatencyEvent(node=node_name, latency_ms=latency_ms))

            total_ms = round((time.perf_counter() - request_start) * 1000)
            yield _sse(LatencyEvent(node="total", latency_ms=total_ms))

        except Exception as e:
            yield _sse(ErrorEvent(message=str(e)))

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/cancel", response_model=CancelResponse)
async def cancel(req: CancelRequest) -> CancelResponse:
    config = {"configurable": {"thread_id": req.thread_id}}
    await GRAPH.ainvoke(Command(resume="cancel"), config)
    return CancelResponse(status="cancelled", thread_id=req.thread_id)

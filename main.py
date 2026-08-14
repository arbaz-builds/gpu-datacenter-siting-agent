"""
GPU Datacenter Siting Agent — FastAPI entrypoint.

LangGraph-based agent that evaluates GPU data center sites, using
NVIDIA NIM for LLM inference and Mireye + EIA APIs for real site data.

Run locally:
    uvicorn main:app --reload

Run in production:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel

from config import DATABASE_URL
from graph import agent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# A single, long-lived Postgres connection pool and compiled graph, set up
# once at startup and reused across requests — opening a fresh Postgres
# connection on every request is slow and wastes connections under load.
_checkpointer_cm = None
_checkpointer = None
_compiled_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer_cm, _checkpointer, _compiled_agent

    _checkpointer_cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
    _checkpointer = await _checkpointer_cm.__aenter__()
    await _checkpointer.setup()
    _compiled_agent = agent.compile(checkpointer=_checkpointer)

    logger.info("[Startup] Postgres checkpointer ready, graph compiled.")
    yield

    await _checkpointer_cm.__aexit__(None, None, None)
    logger.info("[Shutdown] Postgres connection closed.")


app = FastAPI(
    title="GPU Datacenter Siting Agent",
    description=(
        "LangGraph agent that combines Mireye site data and EIA electricity "
        "pricing to REJECT / SHORTLIST / SELECT GPU data center sites."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "1"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


async def _invoke(query_text: str, thread_id: str = "1") -> dict:
    """
    Run one turn of the agent for the given thread and return the raw
    LangGraph result dict. Used by both the API and any local/manual
    testing scripts.
    """
    if _compiled_agent is None:
        raise RuntimeError(
            "Agent is not initialized yet — this should only happen if "
            "_invoke is called outside of the FastAPI lifespan (e.g. a "
            "standalone script). See README for a standalone usage example."
        )

    result = await _compiled_agent.ainvoke(
        {"messages": [HumanMessage(content=query_text)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result


@app.get("/health")
async def health() -> dict:
    """Basic liveness check — does not touch Postgres or any external API."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the GPU Datacenter Siting Agent.

    `thread_id` lets a conversation resume mid-analysis across requests
    (backed by Postgres checkpointing) — reuse the same thread_id for a
    follow-up message in the same session.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")

    try:
        result = await _invoke(request.query, request.thread_id)
    except Exception:
        logger.exception("[Chat] Agent invocation failed.")
        raise HTTPException(
            status_code=500,
            detail="The agent failed to process this request. Check server logs.",
        )

    final_message = result["messages"][-1]
    reply_text = getattr(final_message, "content", str(final_message))

    return ChatResponse(reply=reply_text, thread_id=request.thread_id)

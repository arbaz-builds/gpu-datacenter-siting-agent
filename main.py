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

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel

from config import DATABASE_URL
from graph import agent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="GPU Datacenter Siting Agent",
    description=(
        "LangGraph agent that combines Mireye site data and EIA electricity "
        "pricing to REJECT / SHORTLIST / SELECT GPU data center sites."
    ),
    version="1.0.0",
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
    LangGraph result dict.

    A fresh Postgres connection is opened and closed for this single call.
    This is intentionally NOT a long-lived pooled connection: managed
    free-tier Postgres instances (e.g. Render, Supabase free tier) close
    idle connections after a short period, which turned a single shared
    connection into "works at startup, fails on the first real request"
    (psycopg.OperationalError: SSL connection has been closed
    unexpectedly). A fresh connection per call is slightly slower but
    reliable — this agent is not high-throughput enough for that cost to
    matter.
    """
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as cp:
        await cp.setup()
        result = await agent.compile(checkpointer=cp).ainvoke(
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

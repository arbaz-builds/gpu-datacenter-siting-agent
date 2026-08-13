"""
GPU Center Location Agent
LangGraph-based agent that helps users search for GPU data centers,
using NVIDIA NIM for LLM inference and Mireye + EIA APIs for site data.
"""

import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import DATABASE_URL
from graph import agent



from fastapi import FastAPI


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def _invoke(query_text: str, thread_id: str = "1") -> dict:
    """Shared logic: fresh Postgres connection per call, run one turn,
    return the full result dict.
    """
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as cp:
        await cp.setup()
        result = await agent.compile(checkpointer=cp).ainvoke(
            {"messages": [HumanMessage(content=query_text)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    return result


app = FastAPI()

@app.post("/Chat")
async def Agent(query_text: str, thread_id: str = "1") -> dict:
    result = await _invoke(query_text, thread_id)
    return {"response": result}

     



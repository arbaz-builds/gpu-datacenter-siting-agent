"""
Shared graph state and structured-output schemas.
"""

from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class IntroDecision(BaseModel):
    reply: str = Field(
        description=(
            "A short conversational reply to the user. "
            "If the request is complete, acknowledge it briefly. "
            "If information is missing, ask only for the required GPU "
            "data center details."
        )
    )

    decision: Literal["gpu_search", "unclear"] = Field(
        description=(
            "'gpu_search' if the user has provided enough information to begin "
            "the GPU data center search. "
            "'unclear' if the location, GPU type, or intent is missing or ambiguous."
        )
    )

    RefindTopic: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED FORMAT: a single plain-text sentence, or null. "
            "NEVER a JSON object, dict, or list of fields. "
            "One flowing sentence restating the user's GPU data center "
            "search request in their own terms (location, GPU type, "
            "provider, budget — folded into normal prose, not separate keys). "
            "Example: 'H100 GPU data center site near Austin, TX, budget "
            "up to $50M, any cloud provider.' "
            "Set to null when decision is 'unclear'."
        ),
    )


class CandidateLocation(BaseModel):
    location: str = Field(description="Candidate city or address")
    reason: str = Field(description="Why this location was selected")


class PlannerOutput(BaseModel):
    planner: str = Field(description="Overall search plan")
    candidates: List[CandidateLocation] = Field(
        description="Candidate locations to evaluate"
    )


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    decision: str
    topic: str
    iteration_count: int
    planner: str
    candidate_locations: List[dict]

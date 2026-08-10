"""
Graph node functions: intro routing, decision-tool loop, data structuring,
and final answer generation.
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import LLM

from prompts import ANSWER_PROMPT, DECISION_TOOL_PROMPT, INTRO_SYSTEM, STRUCTURE_LLM_PROMPT,PLANNER_SYSTEM 

from state import IntroDecision, PlannerOutput, State

from tools import tools

logger = logging.getLogger(__name__)


# ============ INTRO ROUTER ===========

async def intro_router(state: State) -> dict:
    messages = [
        SystemMessage(content=INTRO_SYSTEM),
        *state["messages"][-6:],
    ]
    try:
        structured_llm = LLM.with_structured_output(IntroDecision, method="function_calling")
        output = await structured_llm.ainvoke(messages)
        logger.info(f"[IntroRouter] LLM output: {output!r}")
    except Exception as e:
        logger.exception(f"[IntroRouter] LLM call failed: {e}")
        return {
            "messages": [AIMessage(
                content="Sorry, I ran into an issue understanding that. "
                        "Could you tell me the location and GPU type you're looking for?"
            )],
            "topic": None,
            "decision": "unclear",
        }

    return {
        "messages": [AIMessage(content=output.reply)],
        "topic": output.RefindTopic,
        "decision": output.decision,
    }


def intro_router_condition(state: State):
    decision = state.get("decision")
    if decision == "gpu_search":
        return "LLMTOOL"
    return "END"


# ============ PLANNER ===========




async def planner_llm(state: State) -> dict:
    structured_llm = LLM.with_structured_output(PlannerOutput)

    # Use the user's actual last request, not an arbitrary slice of
    # recent messages (which may include intro-router chatter or shift
    # unpredictably as conversation history grows).
    user_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    user_content = user_message.content if user_message else ""

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=user_content),
    ]

    result = await structured_llm.ainvoke(messages)

    candidate_locations = [
        candidate.model_dump() for candidate in result.candidates
    ]

    # llm_tool_node only reads state["messages"] — fold the plan into a
    # message so it actually sees which candidates to fetch data for.
    summary_lines = "\n".join(
        f"- {c['location']}: {c['reason']}" for c in candidate_locations
    )
    plan_message = AIMessage(
        content=(
            f"Search plan: {result.planner}\n\n"
            f"Candidate locations to evaluate:\n{summary_lines}"
        )
    )

    return {
        "messages": [plan_message],
        "planner": result.planner,
        "candidate_locations": candidate_locations,
    }


# ============ DECISION-TOOL LOOP ===========

async def llm_tool_node(state: State) -> dict:
    candidate_locations = state.get("candidate_locations") or []

    if candidate_locations:
        candidates_block = "\n".join(
            f"- {c['location']}: {c['reason']}" for c in candidate_locations
        )
        context_message = HumanMessage(
            content=(
                "Candidate locations from the planner (fetch site data for "
                "each one that doesn't have data yet):\n" + candidates_block
            )
        )
    else:
        context_message = None

    messages = [SystemMessage(content=DECISION_TOOL_PROMPT)]
    if context_message:
        messages.append(context_message)
    messages.extend(state["messages"][-6:])

    resp = await LLM.bind_tools(tools).ainvoke(messages)

    return {
        "messages": [resp],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tool_or_structure(state: State) -> str:
    """After llm_tool: go to 'tools' if a tool call was made and we're
    still under the iteration limit. Otherwise all candidate data has
    been collected — go structure it before the final answer.

    NOTE: iteration_count increments once per llm_tool_node call (before
    the tool actually runs), so a cap of N here yields only N-1 completed
    tool rounds in practice. Capped at 5 to match planner_llm's max of
    5 candidates.
    """
    last = state["messages"][-1]
    iteration_count = state.get("iteration_count", 0)
    if isinstance(last, AIMessage) and last.tool_calls and iteration_count < 6:
        return "tools"
    return "structure"






# ============ STRUCTURE LLM ===========

async def structure_llm_node(state: State) -> dict:
    # The `tools` node (a prebuilt ToolNode) writes each result as a
    # ToolMessage into state["messages"]. If multiple candidates were
    # looked up (one tool call per candidate), there can be MULTIPLE
    # ToolMessages — collect all of them, not just the most recent one,
    # or earlier candidates silently get dropped from the final answer.
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

    if not tool_messages:
        logger.error("[StructureLLM] No ToolMessage found in state; nothing to structure.")
        raw_output = ""
    else:
        raw_output = "\n\n".join(str(m.content) for m in tool_messages)

    messages = [
        SystemMessage(content=STRUCTURE_LLM_PROMPT),
        HumanMessage(content=raw_output),
    ]
    result = await LLM.ainvoke(messages)

    try:
        json.loads(result.content)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "[StructureLLM] Output is not valid JSON; downstream LLM will "
            "receive it as-is. First 200 chars: %r",
            str(result.content)[:200],
        )

    return {"messages": [result]}


# ============ ANSWER NODE ===========

def has_tool_data(state: State) -> bool:
    """True if at least one tool call actually returned site data."""
    return any(isinstance(m, ToolMessage) for m in state["messages"])


async def answer_node(state: State) -> dict:
    if not has_tool_data(state):
        # We hit the iteration limit (or the LLM never called the tool)
        # without ever getting real site data. Do not let the LLM
        # invent a plausible-looking site report from nothing.
        logger.warning("[Answer] No site data was ever retrieved; refusing to fabricate a report.")
        return {
            "messages": [AIMessage(
                content=(
                    "I wasn't able to retrieve site data for this request "
                    "(no successful data lookup completed). Please confirm "
                    "the address and try again."
                )
            )]
        }

    resp = await LLM.ainvoke(
        [
            SystemMessage(content=ANSWER_PROMPT),
            *state["messages"][-6:],
        ]
    )

    try:
        json.loads(resp.content)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "[Answer] Final output is not valid JSON. First 200 chars: %r",
            str(resp.content)[:200],
        )

    return {"messages": [resp]}

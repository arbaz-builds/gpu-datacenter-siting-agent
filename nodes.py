"""
Graph node functions: intro routing, decision-tool loop, data structuring,
and final answer generation.
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import LLM

from prompts import (
    ANSWER_PROMPT,
    DECISION_TOOL_PROMPT,
    INTRO_SYSTEM,
    PLANNER_SYSTEM,
    STRUCTURE_LLM_PROMPT,
)

from state import DecisionOutput, IntroDecision, PlannerOutput, State

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


# ============ DECISION NODE ============

async def decision_node(state: State) -> dict:
    """
    Decide whether each candidate should be REJECTED, SHORTLISTED,
    or SELECTED based on the structured site data.
    """
    structured_data = state["messages"][-1].content

    decision_prompt = f"""
You are the final site-selection decision agent for a GPU data center.

You have been given structured physical-world and economic data
retrieved from Mireye and other data sources.

Your job is to make a BUSINESS DECISION, not just give a score.

For every candidate:

1. REJECT
   Use when a serious/blocking physical or economic constraint
   makes the candidate unsuitable.

2. SHORTLIST
   Use when the site is feasible but is not the strongest candidate.

3. SELECT
   Use for the strongest feasible candidate.

IMPORTANT RULES:
- If at least one candidate is feasible, select EXACTLY ONE candidate.
- If no candidate is feasible, selected_location must be null
  and all candidates must be REJECT.
- Never SELECT a candidate with a blocking issue.
- A high score must NOT override a critical blocking issue.
- Do not invent missing data.
- If important data is missing, mention it as a limitation.
- Base decisions only on the supplied data.
- Every decision must include clear reasons.
- REJECT decisions must include blocking_issues when applicable.
- SHORTLIST decisions should explain why they were not selected.
- SELECT decision should explain why it is the strongest feasible option.

SITE DATA:
{structured_data}
"""

    structured_llm = LLM.with_structured_output(DecisionOutput)
    result = await structured_llm.ainvoke([SystemMessage(content=decision_prompt)])

    decisions = [d.model_dump() for d in result.decisions]

    # --- Programmatic validation: never trust the prompt rule alone ---

    # Rule: a SELECT with blocking_issues is invalid — the prompt says
    # never to do this, but enforce it in code too.
    for d in decisions:
        if d["decision"] == "SELECT" and d.get("blocking_issues"):
            logger.error(
                "[Decision] SELECT has blocking issues for %s: %s",
                d["location"],
                d["blocking_issues"],
            )
            d["decision"] = "REJECT"

    selected = [
        d for d in decisions
        if d["decision"] == "SELECT" and not d.get("blocking_issues")
    ]

    if len(selected) > 1:
        logger.error(
            "[Decision] Invalid output: multiple SELECT decisions: %s",
            [d["location"] for d in selected],
        )
        # Keep the model from producing an invalid final state.
        # Keep the first SELECT only; remaining SELECTs become SHORTLIST.
        selected_location = selected[0]["location"]
        for d in decisions:
            if d["decision"] == "SELECT" and d["location"] != selected_location:
                d["decision"] = "SHORTLIST"
    else:
        selected_location = selected[0]["location"] if selected else None

        # Edge case: result.selected_location points to a location that
        # isn't actually marked SELECT in decisions — trust decisions, not
        # the separate field.
        if result.selected_location and result.selected_location != selected_location:
            logger.error(
                "[Decision] Mismatch: selected_location=%s but decisions show %s",
                result.selected_location,
                selected_location,
            )

    logger.info("[Decision] Selected location: %s", selected_location)

    return {
        "decision_results": decisions,
        "selected_location": selected_location,
    }


# ============ ACTION NODE ============

async def action_node(state: State) -> dict:
    """Convert the site-selection decision into concrete next actions."""
    decisions = state.get("decision_results", [])
    selected_location = state.get("selected_location")

    if not decisions:
        return {"actions": ["No site decision was produced."]}

    actions = []

    if selected_location:
        actions.append(
            f"START_PRELIMINARY_DUE_DILIGENCE: Begin preliminary due diligence for {selected_location}."
        )

    rejected = [d["location"] for d in decisions if d["decision"] == "REJECT"]
    if rejected:
        actions.append("EXCLUDE_REJECTED_SITES: " + ", ".join(rejected))

    shortlisted = [d["location"] for d in decisions if d["decision"] == "SHORTLIST"]
    if shortlisted:
        actions.append("KEEP_AS_BACKUP_OPTIONS: " + ", ".join(shortlisted))

    logger.info("[Action] Actions: %s", actions)
    return {"actions": actions}


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

    decision_summary = json.dumps(state.get("decision_results", []), indent=2)
    action_summary = json.dumps(state.get("actions", []), indent=2)

    final_context = HumanMessage(
        content=(
            f"FINAL SITE DECISIONS:\n{decision_summary}\n\n"
            f"ACTIONS:\n{action_summary}"
        )
    )

    resp = await LLM.ainvoke(
        [
            SystemMessage(content=ANSWER_PROMPT),
            *state["messages"][-6:],
            final_context,
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

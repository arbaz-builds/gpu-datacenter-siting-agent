"""
Builds and compiles the LangGraph StateGraph for the GPU Center Location Agent.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from nodes import (
    action_node,
    answer_node,
    decision_node,
    intro_router,
    intro_router_condition,
    llm_tool_node,
    planner_llm,
    structure_llm_node,
    tool_or_structure,
)
from state import State
from tools import tools

# Wrap tools so they can be used as a graph node
tool_node = ToolNode(tools)


# ============================================================
# GRAPH
# ============================================================

agent = StateGraph(State)

# Nodes
agent.add_node("intro_router", intro_router)
agent.add_node("planner", planner_llm)
agent.add_node("llm_tool", llm_tool_node)
agent.add_node("tools", tool_node)
agent.add_node("structure_llm", structure_llm_node)
agent.add_node("decision", decision_node)
agent.add_node("action", action_node)
agent.add_node("answer", answer_node)


# ============================================================
# START
# ============================================================

agent.add_edge(START, "intro_router")


# ============================================================
# INTRO ROUTER
# ============================================================

agent.add_conditional_edges(
    "intro_router",
    intro_router_condition,
    {
        "LLMTOOL": "planner",
        "END": END,
    },
)


# ============================================================
# PLANNER
# ============================================================

agent.add_edge("planner", "llm_tool")


# ============================================================
# LLM TOOL ROUTER
# Decide: still have candidates left to fetch data for? -> "tools"
# All tool calls done? -> go structure the collected data -> "structure_llm"
# ============================================================

agent.add_conditional_edges(
    "llm_tool",
    tool_or_structure,
    {
        "tools": "tools",
        "structure": "structure_llm",
    },
)


# ============================================================
# TOOLS
# After running a tool call, loop back to llm_tool — it may still
# have more candidates left to fetch data for.
# ============================================================

agent.add_edge("tools", "llm_tool")


# ============================================================
# STRUCTURE LLM
# Runs ONCE, only after llm_tool has finished all tool calls.
# Normalizes the collected site data, then hands off to the
# decision node, which makes an explicit REJECT/SHORTLIST/SELECT
# call for each candidate.
# ============================================================

agent.add_edge("structure_llm", "decision")


# ============================================================
# DECISION
# Converts structured site data into a business decision per
# candidate (REJECT / SHORTLIST / SELECT, with exactly one SELECT
# among feasible candidates).
# ============================================================

agent.add_edge("decision", "action")


# ============================================================
# ACTION
# Deterministically converts the decision into concrete next
# steps. No LLM decision-making happens here — only mapping.
# ============================================================

agent.add_edge("action", "answer")


# ============================================================
# FINAL ANSWER
# ============================================================

agent.add_edge("answer", END)

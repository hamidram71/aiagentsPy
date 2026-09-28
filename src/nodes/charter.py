"""Charter worker: same ReAct shape as the analyst, one tool, visual output."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.config import settings
from src.nodes.analyst import step_block
from src.errors import errors_block, safe_node
from src.llm import get_llm
from src.messages import has_tool_calls, window
from src.prompts import CHARTER_SYSTEM
from src.state import AgentState
from src.tools.charts import CHART_TOOLS

SAVED_PREFIX = "saved chart: "


@safe_node("charter")
def charter_node(state: AgentState) -> dict:
    """Ask for a chart, then record the files that were actually written.

    Note that the chart paths come from the *tool output*, not from what the
    model claims it did. A model will happily tell you it saved a file that
    does not exist; the tool's return value is the only trustworthy source.
    """
    turns = state.get("worker_turns") or 0
    if turns >= settings.max_tool_rounds:
        # Same per-dispatch cap as the analyst. Whatever was drawn is kept.
        return {
            "messages": [AIMessage(content="Charter: turn allowance used up.")],
            "step_count": 1,
            "charts": _new_charts(state),
            "chart_attempted": True,
            "worker_turns": turns + 1,
        }

    llm = get_llm().bind_tools(CHART_TOOLS)

    findings = state.get("findings") or []
    findings_text = "\n".join(f"- {f['text']}" for f in findings) or "(none)"
    system = (
        CHARTER_SYSTEM
        + step_block(state)
        + f"\n\nFindings to illustrate:\n{findings_text}\n"
        + errors_block(state)
    )

    reply: AIMessage = llm.invoke(
        [SystemMessage(content=system), *window(state["messages"], keep=10)]
    )

    if has_tool_calls(reply):
        # Record charts drawn by EARLIER calls of this loop right now. Only
        # the closing reply used to record them, and the budget can cut the
        # loop before that reply: the PNG was on disk, the report said
        # "(none)", and the Charts section was removed.
        return {
            "messages": [reply],
            "step_count": 1,
            "charts": _new_charts(state),
            "chart_attempted": True,
            "worker_turns": turns + 1,
        }

    return {
        "messages": [reply],
        "step_count": 1,
        "charts": _new_charts(state),
        "chart_attempted": True,
        "worker_turns": turns + 1,
    }


def _new_charts(state: AgentState) -> list[str]:
    """Charts this turn produced -- not ones an earlier charter turn recorded.

    _saved_charts scans the whole transcript, so a second charter turn used
    to report the first turn's chart again. The state reducer dropped the
    duplicate, but the CLI printed it: on a real run revenue_by_region.png
    was announced twice, which reads as "it drew the same chart twice".
    """
    known = set(state.get("charts") or [])
    return [c for c in _saved_charts(state) if c not in known]


def _saved_charts(state: AgentState) -> list[str]:
    """Harvest every path a make_chart call actually produced."""
    paths: list[str] = []
    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            content = str(message.content)
            if content.startswith(SAVED_PREFIX):
                # First line only. The tool also reports what it drew from,
                # and a parser that took "the rest of the message" turned
                # that note into part of the file name the report linked to.
                first_line = content[len(SAVED_PREFIX) :].splitlines()[0]
                paths.append(first_line.strip())
    return paths


def route_from_charter(state: AgentState) -> str:
    """Same bounded tool loop as the analyst -- see route_from_analyst."""
    if state.get("failed_node") == "charter":
        return "supervisor"
    if state.get("step_count", 0) >= settings.max_steps:
        return "supervisor"
    last = state["messages"][-1]
    return "charter_tools" if has_tool_calls(last) else "supervisor"

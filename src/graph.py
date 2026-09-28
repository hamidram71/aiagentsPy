"""Graph assembly.

    START
      v
    loader                (deterministic: read + profile the CSV)
      v
    planner               (structured LLM call -> list of steps)
      v
    approval?             (interrupt: a human approves or sends it back)
      v
    supervisor  <-----------------------------+
      |  |  |                                 |
      |  |  +--> reporter --------------------+
      |  +-----> charter  <--> charter_tools  |
      +--------> analyst  <--> analyst_tools  |
      |                                       |
      +--> END (only once a report exists) ---+

Read the edges, not the nodes, to understand an agent: the nodes are just
functions, the edges are the behaviour.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import PROJECT_ROOT, settings
from src.nodes.analyst import analyst_node, route_from_analyst
from src.nodes.approval import approval_node
from src.nodes.charter import charter_node, route_from_charter
from src.nodes.loader import loader_node
from src.nodes.planner import planner_node
from src.nodes.reporter import reporter_node
from src.nodes.supervisor import route_from_supervisor, supervisor_node
from src.state import AgentState
from src.tools.analysis import ANALYSIS_TOOLS
from src.tools.charts import CHART_TOOLS


# --------------------------------------------------------------------------
# small routing functions
# --------------------------------------------------------------------------


def route_after_loader(state: AgentState) -> str:
    """A CSV we cannot read is a dead end -- fail fast and loudly."""
    return "END" if state.get("failed_node") == "loader" else "planner"


def route_after_planner(state: AgentState) -> str:
    """A failed plan is retried within the failure budget, like the supervisor.

    It used to go straight to END: one timeout on the very first model call
    ended the run with no report, exit code 0, and a --resume hint that did
    nothing, because the graph had already finished.
    """
    if state.get("failed_node") == "planner":
        if state.get("failure_count", 0) <= settings.max_failures:
            return "planner"
        return "END"
    return "approval" if settings.require_approval else "supervisor"


def route_after_approval(state: AgentState) -> str:
    """Rejected plan -> back to the planner with the human's feedback."""
    return "supervisor" if state.get("approved") else "planner"


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the agent. Pass a checkpointer to enable memory + interrupts."""
    builder = StateGraph(AgentState)

    builder.add_node("loader", loader_node)
    builder.add_node("planner", planner_node)
    builder.add_node("approval", approval_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("charter", charter_node)
    builder.add_node("reporter", reporter_node)

    # ToolNode turns every tool_call in the last AIMessage into a ToolMessage.
    # handle_tool_errors=True means a raising tool returns its error to the
    # model instead of killing the graph -- the model then usually fixes it.
    builder.add_node("analyst_tools", ToolNode(ANALYSIS_TOOLS, handle_tool_errors=True))
    builder.add_node("charter_tools", ToolNode(CHART_TOOLS, handle_tool_errors=True))

    builder.add_edge(START, "loader")

    builder.add_conditional_edges(
        "loader", route_after_loader, {"planner": "planner", "END": END}
    )
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"approval": "approval", "supervisor": "supervisor", "planner": "planner", "END": END},
    )
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"supervisor": "supervisor", "planner": "planner"},
    )
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "analyst": "analyst",
            "charter": "charter",
            "reporter": "reporter",
            # A failed supervisor turn retries itself, bounded by max_failures.
            "supervisor": "supervisor",
            "FINISH": END,
            "END": END,
        },
    )
    builder.add_conditional_edges(
        "analyst",
        route_from_analyst,
        {"analyst_tools": "analyst_tools", "supervisor": "supervisor"},
    )
    builder.add_edge("analyst_tools", "analyst")

    builder.add_conditional_edges(
        "charter",
        route_from_charter,
        {"charter_tools": "charter_tools", "supervisor": "supervisor"},
    )
    builder.add_edge("charter_tools", "charter")

    builder.add_edge("reporter", "supervisor")

    return builder.compile(checkpointer=checkpointer)


def make_checkpointer() -> tuple[BaseCheckpointSaver, sqlite3.Connection | None]:
    """A SQLite checkpointer, or an in-memory one when CHECKPOINT_DB=:memory:.

    SQLite is what makes `interrupt()` survive a process restart: the pause
    lives in the database, not in RAM. Swap in `PostgresSaver` for anything
    multi-process.
    """
    if settings.checkpoint_db in {":memory:", "", "none"}:
        return InMemorySaver(), None

    db_path = Path(settings.checkpoint_db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from langgraph.checkpoint.sqlite import SqliteSaver

    # check_same_thread=False: LangGraph may touch the connection from a
    # worker thread when you use the async or streaming APIs.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn), conn

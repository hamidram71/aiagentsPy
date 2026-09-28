"""The graph state and its reducers.

A LangGraph node returns a *partial* state. LangGraph then merges that partial
into the running state. How it merges depends on the reducer attached to each
key with `Annotated[...]`:

    messages: Annotated[list, add_messages]  -> append (and de-duplicate by id)
    findings: Annotated[list, append_unique] -> append, skipping duplicates
    profile:  dict                           -> plain overwrite (no reducer)

Getting reducers right is most of the skill in LangGraph. If two nodes write
the same key with no reducer and they run in parallel, you get an
InvalidUpdateError -- that is the framework telling you the merge is ambiguous.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

WorkerName = Literal["analyst", "charter", "reporter", "FINISH"]


def append_unique(left: list[Any], right: list[Any]) -> list[Any]:
    """Append items from `right` that are not already in `left`.

    Useful for findings: two workers may independently notice the same thing
    and we do not want the report to repeat it.
    """
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


class Finding(TypedDict):
    """One atomic result produced by a worker."""

    source: str  # which node produced it
    text: str  # human-readable statement
    evidence: str  # the numbers / code output behind the statement


class AgentState(TypedDict, total=False):
    # --- conversation ------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]

    # --- inputs ------------------------------------------------------------
    csv_path: str
    question: str
    # This run's own folder for charts, the report and the cleaned copy.
    # Charts are named by the model ("revenue_by_region.png"), so with one
    # shared outputs/ folder, two people running at once overwrote each
    # other's charts -- and a report could show a colleague's chart.
    run_dir: str

    # --- produced by the loader node --------------------------------------
    profile: dict[str, Any]

    # --- the cleaned working copy (save_cleaned_data) ---------------------
    # Path of the dataset every tool reads once the analyst has cleaned the
    # data; absent until then, and tools fall back to csv_path. Last write
    # wins, which is right: a second cleaning builds on the first.
    data_path: str
    # One line per cleaning action, in order, so the report can say exactly
    # what was changed. `operator.add` because the history is the point.
    cleaning_log: Annotated[list[str], operator.add]

    # --- produced by the supervisor ---------------------------------------
    plan: list[str]
    next_worker: WorkerName
    step_count: Annotated[int, operator.add]
    # Which plan step the supervisor just dispatched, and which steps have
    # actually produced a finding. Without this the supervisor can see the
    # plan and it can see the findings, but nothing connects the two -- so it
    # re-issues work it already completed. Observed on a real run: step 1
    # produced a correct finding, and the very next turn asked for step 1
    # again with the reason "(Not yet done.)".
    current_step: int
    addressed_steps: Annotated[list[int], append_unique]
    # Turns the current worker has taken on the step it was dispatched to.
    # No reducer on purpose: last write wins, so the supervisor can reset it
    # to 0 on every dispatch. (operator.add could only ever go up.)
    worker_turns: int
    # The instruction the supervisor gave with the current dispatch, so a
    # worker that runs out of turns can be asked about THIS step, not about
    # the user's whole question.
    current_instruction: str
    # One entry per analyst dispatch: the plan step it was sent to. Counting
    # them is how the supervisor knows a step has had its tries.
    attempts: Annotated[list[int], operator.add]

    # --- produced by the workers ------------------------------------------
    findings: Annotated[list[Finding], append_unique]
    charts: Annotated[list[str], append_unique]
    # Set once the charter has had its turn, whether or not it managed to
    # draw anything -- so the "every report gets a chart" rule can force the
    # charter exactly once and never loop on a charter that keeps failing.
    chart_attempted: bool
    report_path: str

    # --- human-in-the-loop -------------------------------------------------
    approved: bool
    human_feedback: str

    # --- error handling ----------------------------------------------------
    errors: Annotated[list[str], operator.add]
    # Total node failures in this run. Named `failure_count`, not
    # `retry_count`, because `operator.add` can only ever go up: there is no
    # way to "reset it on success" with this reducer, so it is a whole-run
    # budget and the name should say so. Calling it a retry count would be a
    # lie that costs someone an afternoon.
    failure_count: Annotated[int, operator.add]
    # Sticky-until-cleared marker of the node that just failed. `safe_node`
    # sets it on failure and clears it on success; the route_* functions read
    # it. See src/errors.py for why clearing matters.
    failed_node: str


def initial_state(csv_path: str, question: str, run_dir: str | None = None) -> AgentState:
    """Build the state we hand to `graph.invoke()`."""
    extra: dict = {"run_dir": run_dir} if run_dir else {}
    return {
        **extra,
        "messages": [],
        "csv_path": csv_path,
        "question": question,
        "plan": [],
        "findings": [],
        "charts": [],
        "errors": [],
        "failure_count": 0,
        "step_count": 0,
        "failed_node": "",
        "current_step": 0,
        "addressed_steps": [],
    }

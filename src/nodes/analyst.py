"""Analyst worker: a ReAct loop that computes numbers and reports findings.

The loop is three graph elements, and it is worth being able to draw it from
memory because almost every tool-using agent is this shape:

    analyst --(tool_calls present)--> analyst_tools --> analyst
    analyst --(no tool_calls)-------> supervisor

`ToolNode` executes every tool call in the last AIMessage and appends one
`ToolMessage` per call. Because `messages` uses the `add_messages` reducer,
those appends merge into the shared transcript automatically.
"""

from __future__ import annotations

from typing import NamedTuple

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.config import settings
from src.budget import analysis_limit
from src.internal_prompts import WRAP_UP_SYSTEM
from src.errors import errors_block, safe_node
from src.llm import get_llm
from src.messages import has_tool_calls, parse_tagged, window
from src.prompts import ANALYST_SYSTEM, FINDING_EXTRACTOR_SYSTEM
from src.schemas import FindingOut
from src.state import AgentState
from src.tools.analysis import ANALYSIS_TOOLS


@safe_node("analyst")
def analyst_node(state: AgentState) -> dict:
    """One turn of the analyst: call tools, emit a finding, or wrap up.

    Wrap-up happens when this dispatch has used its MAX_TOOL_ROUNDS turns.
    Then no more tools: the finding is extracted, as a typed object, from
    the last tool output the analyst received. A real run spent its entire
    budget on the first two of five plan steps -- the analyst simply kept
    checking -- and the question was never answered.
    """
    turns = state.get("worker_turns") or 0
    # The last turn the budget allows is a wrap-up too. Otherwise a model that
    # "verifies once more" gets cut off after its tool call, and everything
    # this step already computed is dropped: on a test run the monthly totals
    # had been calculated, and the report still said the trend was unmeasured.
    budget_ends_now = turns > 0 and state.get("step_count", 0) + 1 >= analysis_limit(state)
    if turns >= settings.max_tool_rounds or budget_ends_now:
        update = _wrap_up(state)
    else:
        update = _analyst_turn(state)
    update["worker_turns"] = turns + 1
    return update


class ExtractionFailed(NamedTuple):
    """The extraction call itself failed -- distinct from "no finding".

    Both used to come back as None. So a model error during extraction (or,
    in the test suite, a script that no longer matched the flow) vanished
    without a trace: the step simply produced nothing. Best-effort still --
    it never fails the node -- but it is written down.
    """

    error: str


def _extraction_failed(failure: ExtractionFailed) -> dict:
    return {
        "messages": [AIMessage(content="Analyst: extracting a finding failed; see errors.")],
        "step_count": 1,
        # Recorded, not counted: `errors` without `failure_count`, because the
        # node itself did its job; one best-effort call did not.
        "errors": [f"analyst: finding extraction failed: {failure.error}"],
    }


def _wrap_up(state: AgentState) -> dict:
    """The step's turns are used up: salvage a finding from what is there.

    The first version of this reused _extract_finding with the reply text
    "(the tool-call allowance for this step is used up)". That extractor's
    prompt says a finding exists only when the analyst's *reply* states
    something about the numbers -- and that reply states nothing. So it
    answered "no finding" by construction, every time, and on a real run a
    validation step was dispatched three times for one result. It also got
    the user's overall question instead of the step, and a formula check
    ("0 rows over $0.50") is rightly not an answer to "which region drives
    revenue". A helper reused for a question it was not written for.

    Here the question is the one that matters: given THIS step and THESE
    tool outputs, is the step answered?
    """
    extracted = _wrap_up_finding(state)
    if isinstance(extracted, ExtractionFailed):
        return _extraction_failed(extracted)
    if extracted is None:
        return {
            "messages": [
                AIMessage(content="Analyst: turn allowance for this step used up; "
                                  "the tool outputs did not answer the step.")
            ],
            "step_count": 1,
        }
    statement, evidence = extracted
    current = state.get("current_step") or 0
    return {
        "messages": [
            AIMessage(content=f"Analyst: turn allowance for this step used up. "
                              f"FINDING: {statement}")
        ],
        "step_count": 1,
        "addressed_steps": [current] if current else [],
        "findings": [
            {"source": "analyst", "text": statement,
             "evidence": (evidence or _tool_outputs_since_dispatch(state))[:1500]}
        ],
    }


def _analyst_turn(state: AgentState) -> dict:
    """One ordinary turn: either call tools, or emit a finding."""
    llm = get_llm().bind_tools(ANALYSIS_TOOLS)

    system = ANALYST_SYSTEM + step_block(state) + errors_block(state)
    reply: AIMessage = llm.invoke(
        [SystemMessage(content=system), *window(state["messages"])]
    )

    # Still working: hand the tool calls to ToolNode via the conditional edge.
    # Every turn costs a step, including tool-loop turns -- see the note on
    # route_from_analyst below.
    if has_tool_calls(reply):
        return {"messages": [reply], "step_count": 1}

    # Done. Record a finding only if the model actually stated one.
    #
    # The tempting version of this line is
    #     statement = parse_tagged(text, "FINDING") or text.strip()
    # and it is wrong. With the fallback, a refusal ("I cannot answer that
    # with these columns"), a thinking-out-loud message, or an empty reply all
    # become a Finding with source="analyst" -- which then satisfies the
    # supervisor's "do we have findings yet?" guard and gets handed to the
    # reporter, whose prompt treats findings as established fact. That is a
    # pipeline that manufactures evidence. An unlabelled reply stays in the
    # transcript, where the supervisor can see it and decide what to do.
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    statement = parse_tagged(text, "FINDING")
    evidence = parse_tagged(text, "EVIDENCE")

    if not statement:
        # The text format was not followed. Rather than discarding a turn
        # that may well contain a real result, ask for the same information
        # as a typed object -- the thing this project already trusts for
        # routing. A model that ignores "reply in this exact shape" will
        # still fill in a schema, because the schema is enforced rather than
        # requested.
        extracted = _extract_finding(state, text)
        if isinstance(extracted, ExtractionFailed):
            return {**_extraction_failed(extracted), "messages": [reply]}
        if extracted is None:
            return {"messages": [reply], "step_count": 1}
        statement, evidence = extracted

    evidence = evidence or _tool_outputs_since_dispatch(state)
    current = state.get("current_step") or 0
    return {
        "messages": [reply],
        "step_count": 1,
        # Attribute the finding to the step the supervisor dispatched, so the
        # supervisor can see that step is finished and move on.
        "addressed_steps": [current] if current else [],
        "findings": [
            {"source": "analyst", "text": statement, "evidence": evidence[:1500]}
        ],
    }


def _extract_finding(state: AgentState, reply_text: str) -> tuple[str, str] | None:
    """Second chance: pull a finding out of an unlabelled reply, as a schema.

    Returns (statement, evidence), or None when there is genuinely nothing.
    Grounded in the tool output the model actually received, so this is not
    a licence to invent -- `has_finding` must be False unless a tool produced
    a number. A failure here is not fatal: no finding, same as before.
    """
    # This step's outputs, not "the most recent tool output in the whole
    # transcript": when the analyst answered step 2 in prose without calling
    # a tool, that was step 1's region table, and step 2 could be marked done
    # with step 1's numbers. No output for this step means nothing backs the
    # prose, so there is nothing to extract.
    tool_output = _tool_outputs_since_dispatch(state)
    if not tool_output.strip():
        return None
    asked = state.get("current_instruction") or state.get("question", "")

    try:
        result: FindingOut = get_llm().with_structured_output(FindingOut).invoke(
            [
                SystemMessage(content=FINDING_EXTRACTOR_SYSTEM),
                (
                    "human",
                    f"The analyst was asked: {asked}\n\n"
                    f"Tool outputs for this step, oldest first:\n{tool_output[-2000:]}\n\n"
                    f"What the analyst then wrote:\n{reply_text[:1500] or '(nothing)'}",
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - extraction is best-effort
        return ExtractionFailed(f"{type(exc).__name__}: {exc}")

    if not result.has_finding or not result.statement.strip():
        return None
    return result.statement.strip(), result.evidence.strip()


def step_block(state: AgentState) -> str:
    """The step being worked on, for the system prompt.

    The supervisor's instruction also sits in the transcript, but window()
    keeps only the most recent messages, and with several parallel tool calls
    per turn it fell out by the fourth turn: the analyst went on working with
    only the user's overall question in view, and its answer was still
    credited to the dispatched step. The system prompt is never trimmed.
    """
    instruction = state.get("current_instruction")
    return f"\n\nThe step you are working on now: {instruction}\n" if instruction else ""


def _tool_outputs_since_dispatch(state: AgentState, max_chars: int = 3000) -> str:
    """Every tool output the analyst received on the current step, in order."""
    outputs: list[str] = []
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and str(message.content).startswith("Supervisor ->"):
            break
        if isinstance(message, ToolMessage):
            outputs.append(str(message.content))
    outputs.reverse()
    text = "\n\n---\n\n".join(outputs)
    # Keep the most recent part if it is long: that is where the answer is.
    return text[-max_chars:] if len(text) > max_chars else text


def _wrap_up_finding(state: AgentState) -> tuple[str, str] | None:
    """Ask, as a schema, whether this step's tool outputs answer the step."""
    outputs = _tool_outputs_since_dispatch(state)
    if not outputs.strip():
        return None
    step = state.get("current_instruction") or state.get("question", "")
    try:
        result: FindingOut = get_llm().with_structured_output(FindingOut).invoke(
            [
                SystemMessage(content=WRAP_UP_SYSTEM),
                ("human", f"The step: {step}\n\nTool outputs, oldest first:\n{outputs}"),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, like _extract_finding
        return ExtractionFailed(f"{type(exc).__name__}: {exc}")
    if not result.has_finding or not result.statement.strip():
        return None
    return result.statement.strip(), result.evidence.strip()


def _last_tool_output(state: AgentState) -> str:
    """Fallback evidence: whatever the most recent tool returned."""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            return str(message.content)
    return ""


def route_from_analyst(state: AgentState) -> str:
    """Tool loop, or back to the supervisor.

    The step budget has to be enforced *here*, not only in the supervisor.
    `analyst <-> analyst_tools` is a closed loop that never passes through
    the supervisor, so a supervisor-only budget does not bound it at all: a
    model that keeps calling tools will run until LangGraph's recursion limit
    (default ~10000 -- thousands of paid calls). Whenever you add a cycle to
    a graph, ask which counter breaks it, and if the answer is "none", that
    cycle is unbounded.
    """
    if state.get("failed_node") == "analyst":
        return "supervisor"

    if state.get("step_count", 0) >= analysis_limit(state):
        return "supervisor"

    last = state["messages"][-1]
    return "analyst_tools" if has_tool_calls(last) else "supervisor"

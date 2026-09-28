"""Supervisor node: the router at the centre of the multi-agent pattern.

Why a supervisor instead of a fixed chain? Because the right number of
analysis steps depends on what the data turns out to look like, and only the
model can judge that at runtime. The supervisor is the one place allowed to
make that judgement, which keeps the workers simple and single-purpose.

Two guardrails, both non-negotiable in any looping agent:
  * a hard step budget, so a confused model cannot burn your API credits;
  * a retry budget, so a permanently failing tool ends the run with a partial
    report instead of spinning forever.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from src.config import settings
from src.budget import (
    MAX_ATTEMPTS_PER_STEP,
    analysis_limit,
    chart_owed,
    room_for_another_try,
)
from src.errors import errors_block, safe_node
from src.llm import get_llm
from src.prompts import SUPERVISOR_SYSTEM
from src.schemas import Route
from src.state import AgentState


def _progress(state: AgentState) -> str:
    findings = state.get("findings") or []
    charts = state.get("charts") or []
    plan = state.get("plan") or []
    done = set(state.get("addressed_steps") or [])

    # Marking each step [done]/[pending] is the whole fix for a supervisor
    # that repeats itself. A plain list of steps plus a separate list of
    # findings leaves the model to infer the mapping, and it infers wrong.
    lines = ["Plan:"]
    for i, step in enumerate(plan, 1):
        lines.append(f"  {i}. [{'done' if i in done else 'pending'}] {step}")
    lines.append(f"\nFindings so far ({len(findings)}):")
    lines += [f"  - [{f['source']}] {f['text']}" for f in findings] or ["  (none yet)"]
    lines.append(f"\nCharts so far ({len(charts)}):")
    lines += [f"  - {c}" for c in charts] or ["  (none yet)"]
    lines.append(
        f"\nReport written: {'yes -> ' + state['report_path'] if state.get('report_path') else 'no'}"
    )
    lines.append(f"Steps used: {state.get('step_count', 0)} / {settings.max_steps}")
    return "\n".join(lines)


CHART_INSTRUCTION = (
    "Before the report: draw a chart for each part of the question that has "
    "a finding -- a bar chart for a comparison between groups, a line chart "
    "with time_bucket='month' for a trend over time. At most two charts."
)


@safe_node("supervisor")
def supervisor_node(state: AgentState) -> dict:
    """Decide which worker acts next."""
    step = state.get("step_count", 0)
    failures = state.get("failure_count", 0)

    # --- guardrails, checked before we spend a token -----------------------
    if state.get("report_path"):
        return {
            "next_worker": "FINISH",
            "step_count": 1,
            "messages": [AIMessage(content="Supervisor: report exists, finishing.")],
        }

    # The reporter is the only way out of the two bail-out branches below, so
    # a reporter that cannot run at all would otherwise mean an infinite
    # supervisor <-> reporter loop. Every escape hatch needs its own escape
    # hatch: check that the node you fall back to can actually succeed.
    if state.get("failed_node") == "reporter":
        return {
            "next_worker": "FINISH",
            "step_count": 1,
            "messages": [
                AIMessage(content="Supervisor: the reporter itself failed; stopping.")
            ],
        }

    out_of_budget = step >= settings.max_steps
    out_of_failures = failures > settings.max_failures
    if out_of_budget or out_of_failures:
        why = "step budget exhausted" if out_of_budget else "too many failures"
        return {
            "next_worker": "reporter",
            "step_count": 1,
            "messages": [
                AIMessage(content=f"Supervisor: {why}; writing a partial report.")
            ],
        }

    # Analysis has used its share and a chart is still owed: spend the
    # reserved steps on the chart. Checked in code, before any model call,
    # because the budget guardrail above used to send the run straight to
    # the reporter and skip the chart rule entirely.
    if chart_owed(state) and step >= analysis_limit(state):
        return {
            "next_worker": "charter",
            "current_instruction": CHART_INSTRUCTION,
            "worker_turns": 0,
            "step_count": 1,
            "messages": [
                AIMessage(
                    content="Supervisor -> charter: analysis budget used; "
                    f"{CHART_INSTRUCTION} (reserved steps for the chart)"
                )
            ],
        }

    # --- the actual decision ----------------------------------------------
    llm = get_llm().with_structured_output(Route)
    prompt = (
        f"Question: {state['question']}\n\n{_progress(state)}\n"
        f"{errors_block(state)}\n"
        "Choose the next worker."
    )
    route: Route = llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM), ("human", prompt)]
    )

    # Enforce the invariants the prompt only *asks* for. Never trust a model
    # to respect a rule you can check in code.
    findings = state.get("findings") or []
    if route.next == "FINISH" and not state.get("report_path"):
        route.next, route.instruction = "reporter", "Write the report from the findings."
    if route.next in {"reporter", "charter"} and not findings:
        route.next, route.instruction = "analyst", state["question"]

    done = set(state.get("addressed_steps") or [])
    plan = state.get("plan") or []
    attempts = list(state.get("attempts") or [])
    _enforce_plan(route, state, plan, done, attempts, findings, step)

    # Every report with findings gets a chart. The prompt asks for this, and
    # a real model ignored it on two runs in a row: it went from the last
    # finding straight to the reporter. So code decides. Exactly once --
    # `chart_attempted` is set by the charter whether or not it succeeded,
    # so a charter that cannot draw anything costs one turn, not a loop.
    if (
        route.next == "reporter"
        and settings.require_chart
        and findings
        and not state.get("charts")
        and not state.get("chart_attempted")
    ):
        route.next = "charter"
        route.instruction = CHART_INSTRUCTION
        route.reason = "every report gets at least one chart"

    content = (
        f"Supervisor -> {route.next} (step {route.step_index}): "
        f"{route.instruction} ({route.reason})"
    )
    return {
        "next_worker": route.next,
        "current_step": route.step_index,
        "current_instruction": route.instruction,
        "attempts": [route.step_index] if route.next == "analyst" else [],
        "worker_turns": 0,
        "step_count": 1,
        "messages": [AIMessage(content=content)],
    }


def route_from_supervisor(state: AgentState) -> str:
    """Conditional edge: read the decision the supervisor put in the state.

    A failure here means the router itself broke, so `next_worker` is stale.
    Salvage what we have (or stop) rather than acting on an old decision --
    but note this only fires for the invocation that actually failed, because
    `safe_node` clears the flag on success.
    """
    if state.get("failed_node") == "supervisor":
        if state.get("failure_count", 0) > settings.max_failures:
            return "reporter" if (state.get("findings") or []) else "END"
        # Within budget: let the supervisor try again rather than giving up
        # on the run because of one 429.
        return "supervisor"
    return state.get("next_worker", "analyst")


def _enforce_plan(route, state, plan, done, attempts, findings, step) -> None:
    """Keep the run on the plan: every analysis step gets its turn, at most twice.

    Three rules, all ones the prompt already asks for and a real run broke:

    * The analyst is never sent to a step that is done, or that has already
      been tried MAX_ATTEMPTS_PER_STEP times -- the next step that can still
      be tried goes instead. (The prompt says "change approach or move on";
      the model re-sent one validation step three times.)
    * Neither the reporter nor the charter runs while an analysis step is
      still pending and the budget allows another try. On a real run the
      supervisor sent the "revenue by region" and "monthly trend" steps to
      the CHARTER: two pictures, no numbers, and the report had nothing to
      say about the question's two halves.
    * The last plan step is the report (the planner always ends with it), so
      it is not an analysis step.
    * step_index 0 means ad-hoc work outside the plan (the code itself uses
      it to send the analyst at the whole question). That is allowed -- only
      a *repeated* ad-hoc dispatch counts as used up.

    Analyst-only for the "done step" part. The first version of that rule
    applied to every worker but the reporter -- and the charter's whole job
    is to draw steps that are *done*, so "chart the trend (step 3, done)" was
    rewritten to the next pending step. A rule written for one worker,
    applied to all.
    """
    analysis_steps = range(1, len(plan))
    pending = [i for i in analysis_steps if i not in done]
    retryable = [i for i in pending if attempts.count(i) < MAX_ATTEMPTS_PER_STEP]

    def send_analyst_to(i: int, reason: str) -> None:
        route.next, route.step_index = "analyst", i
        route.instruction, route.reason = plan[i - 1], reason

    if route.next == "analyst":
        used_up = (
            route.step_index in done
            or (route.step_index != 0 and route.step_index not in analysis_steps)
            or attempts.count(route.step_index) >= MAX_ATTEMPTS_PER_STEP
        )
        if used_up and retryable:
            send_analyst_to(retryable[0], "next plan step that can still be tried")
        elif used_up:
            # With or without findings. Without, this used to fall through
            # and send the analyst out again -- "at most twice per step" did
            # not hold, and the rest of the budget went on repeats. The
            # reporter writes a no-findings report without a model call.
            route.next = "reporter"
            route.instruction = "Write the report from the findings."
            route.reason = "every analysis step is done or was tried twice"
    elif route.next in {"reporter", "charter"} and retryable and room_for_another_try(state):
        send_analyst_to(retryable[0], "plan step not analysed yet")

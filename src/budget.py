"""How the step budget is shared out. No LLM, no I/O -- just arithmetic.

One number, MAX_STEPS, bounds the whole run. A real run showed what happens
when nothing else divides it: gpt-5.6-terra spent the ENTIRE budget on the
first two plan steps (checking duplicates, then the revenue formula), the
supervisor dispatched the analyst exactly twice, and region, trend and chart
never got a turn. The report answered neither half of the question.

Two rules divide it now:

  * per dispatch: a worker gets at most MAX_TOOL_ROUNDS turns for the step
    it was sent to do (see analyst_node), so one step cannot starve the rest;
  * at the end: while a chart is still owed, the last CHART_RESERVE steps
    are kept for the charter, because a budget that runs out first skips
    the chart rule entirely -- which is exactly what happened.
"""

from __future__ import annotations

from src.config import settings

# Steps kept back for the charter, counted so its whole turn fits:
#   1  the analyst's turn that crosses the analysis limit (the edge cuts it
#      *after* that turn, so it can overshoot by one)
#   1  the supervisor dispatching the charter
#   2  two make_chart calls -- CHART_INSTRUCTION asks for up to two charts
#      (a comparison and a trend); the first version reserved room for one
#   1  the charter's closing reply
CHART_RESERVE = 5

# Dispatches one plan step may get before the supervisor moves on. A real run
# sent the same validation step three times; after two, a third try is
# rarely different and always costs a full turn allowance.
MAX_ATTEMPTS_PER_STEP = 2


# Steps one analyst try needs to produce anything: the supervisor's dispatch,
# one tool call, one reply. Sending an analyst with fewer left only buys a turn
# that the budget cuts off mid-thought.
MIN_STEPS_FOR_A_TRY = 3


def room_for_another_try(state: dict) -> bool:
    return analysis_limit(state) - state.get("step_count", 0) >= MIN_STEPS_FOR_A_TRY


def chart_owed(state: dict) -> bool:
    """A chart is required, there is something to chart, and none exists yet."""
    return bool(
        settings.require_chart
        and (state.get("findings") or [])
        and not (state.get("charts") or [])
        and not state.get("chart_attempted")
    )


def analysis_limit(state: dict) -> int:
    """Step count at which analysis stops, so what is reserved stays free."""
    return settings.max_steps - (CHART_RESERVE if chart_owed(state) else 0)

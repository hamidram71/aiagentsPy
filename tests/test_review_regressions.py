"""Regression tests for the full-code review (version 0.13.0).

One test per confirmed finding that is not covered elsewhere. Each was
checked by breaking the fix on purpose and watching this file fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import src.dataset as dataset
from src.graph import build_graph
from src.llm import get_llm
from src.schemas import Plan, Route
from src.state import initial_state
from src.tools.analysis import group_aggregate, save_cleaned_data, value_counts
from src.tools.charts import make_chart
from tests.fake_llm import default_script
from tests.test_graph import _install, assert_script_matched

CSV = "data/sales.csv"
S = {"csv_path": CSV}


@pytest.fixture(autouse=True)
def fresh_caches():
    get_llm.cache_clear()
    dataset.load_df.cache_clear()
    yield
    get_llm.cache_clear()
    dataset.load_df.cache_clear()


def _csv(tmp_path: Path, frame: pd.DataFrame, name: str = "d.csv") -> str:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


# --- tools and data --------------------------------------------------------------


def test_rows_with_an_empty_group_key_are_a_group_not_dropped() -> None:
    """#1: 9 orders without a channel -- 40,498.07 of revenue -- vanished from
    "revenue per channel", under a header saying "3 of 3 groups"."""
    out = group_aggregate.func(group_by=["channel"], value_column="revenue", agg="sum", state=S)
    assert "channel=(blank) -> 40,498.07" in out
    assert "of 4 groups" in out


def test_small_rates_keep_their_difference(tmp_path) -> None:
    """#4: 0.4% and 0.1% both printed as 0.00."""
    frame = pd.DataFrame({"plant": ["A"] * 10 + ["B"] * 10, "rate": [0.004] * 10 + [0.001] * 10})
    out = group_aggregate.func(group_by=["plant"], value_column="rate", agg="mean",
                               state={"csv_path": _csv(tmp_path, frame)})
    assert "plant=A -> 0.004" in out and "plant=B -> 0.001" in out


def test_value_counts_says_what_it_left_out() -> None:
    """#5: 15 of 88 customers looked like the whole list."""
    out = value_counts.func(column="customer_id", state=S)
    assert out.splitlines()[0].startswith("customer_id: 88 distinct values")
    assert "73 more values not shown" in out.splitlines()[-1]


def test_a_blank_value_is_named_in_value_counts() -> None:
    out = value_counts.func(column="channel", state=S)
    assert "(blank): 9" in out


def test_cleaning_that_adds_rows_is_refused() -> None:
    """#6: `concat([df, df])` was saved, and every later total doubled."""
    command = save_cleaned_data.func(code="result = pd.concat([df, df])", description="x",
                                     state=S, tool_call_id="c")
    assert "data_path" not in command.update
    assert "never adds them" in command.update["messages"][0].content


def test_a_changed_column_type_is_reported() -> None:
    command = save_cleaned_data.func(
        code="df['order_date'] = df['order_date'].dt.strftime('%Y-%m-%d')\nresult = df",
        description="dates as text", state=S, tool_call_id="c",
    )
    assert "column types changed: order_date" in command.update["cleaning_log"][0]


def test_a_date_column_with_many_blanks_is_still_a_date(tmp_path) -> None:
    """#20: the parse rate counted blank cells as failures, so >10% blanks
    meant "not a date" -- and a cleaned copy with NaT lost its date type."""
    frame = pd.read_csv(CSV)
    frame.loc[frame.sample(frac=0.25, random_state=1).index, "order_date"] = None
    loaded = dataset.load_df(_csv(tmp_path, frame))
    assert pd.api.types.is_datetime64_any_dtype(loaded["order_date"])


@pytest.mark.parametrize(("column", "values"), [
    ("month_name", ["Jan", "Feb", "Mar"]),   # became year-1 dates
    ("order_time", ["09:15", "14:30", "17:05"]),  # got today's date attached
])
def test_text_that_is_not_a_calendar_date_stays_text(tmp_path, column, values) -> None:
    """#26."""
    frame = pd.DataFrame({column: values * 10, "v": range(30)})
    loaded = dataset.load_df(_csv(tmp_path, frame, f"{column}.csv"))
    assert not pd.api.types.is_datetime64_any_dtype(loaded[column])
    assert loaded[column].iloc[0] == values[0]


def test_the_profile_holds_only_json_values() -> None:
    """#30: pandas Timestamps in profile["head"] went into the checkpoint, and
    LangGraph refused to rebuild them: warnings on every --resume."""
    profile = dataset.profile_df(dataset.load_df(CSV))
    json.dumps(profile)  # raises on a Timestamp
    assert isinstance(profile["head"][0]["order_date"], str)


# --- charts ----------------------------------------------------------------------


def test_a_monthly_bar_chart_is_drawn(isolate_outputs) -> None:
    """#13: every bar chart with time_bucket failed -- bar plots reject `marker`."""
    out = make_chart.invoke({"kind": "bar", "x": "order_date", "y": "revenue", "agg": "sum",
                             "time_bucket": "month", "title": "t", "filename": "m.png", "state": S})
    assert out.startswith("saved chart: "), out
    assert "12 monthly points" in out
    assert (isolate_outputs / "m.png").exists()


def test_an_empty_column_is_not_drawn_as_a_line_of_zeros(tmp_path, isolate_outputs) -> None:
    """#27: all-empty values summed to 0 per month: "peak = 0.00"."""
    frame = pd.read_csv(CSV)
    frame["revenue"] = float("nan")
    out = make_chart.invoke({"kind": "line", "x": "order_date", "y": "revenue", "agg": "sum",
                             "time_bucket": "month", "title": "t", "filename": "e.png",
                             "state": {"csv_path": _csv(tmp_path, frame)}})
    assert out.startswith("ERROR") and "no values" in out


def test_a_bar_chart_reports_the_numbers_it_drew_and_its_blank_group(isolate_outputs) -> None:
    """#28: a bar chart silently kept the top 15 and returned no numbers."""
    out = make_chart.invoke({"kind": "bar", "x": "channel", "y": "revenue", "agg": "sum",
                             "title": "t", "filename": "c.png", "state": S})
    assert "4 of 4 channel groups shown" in out
    assert "(blank) = 40,498.07" in out


def test_small_values_get_readable_axis_labels(isolate_outputs) -> None:
    """A rate chart (0.02 ... 0.10) had every tick labelled "0"."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.tools.charts import _format_value_axis

    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [0.04, 0.10])
    _format_value_axis(ax)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)
    assert any(label not in ("0", "") and label != "−0" for label in labels[1:]), labels
    assert len(set(labels)) > 2, f"ticks collapsed to one value: {labels}"


# --- graph flow ------------------------------------------------------------------


def _run(script, monkeypatch, override_settings, thread, **settings_overrides):
    override_settings(require_approval=False, **settings_overrides)
    model = _install(script, monkeypatch)
    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, "Which region drives revenue, and is the trend up?"),
        config={"configurable": {"thread_id": thread}},
    )
    assert_script_matched(final)
    return model, final


def test_a_plan_without_a_report_step_still_gets_every_step_analysed(
    monkeypatch, override_settings
) -> None:
    """#7: with the plan ["region", "trend"], "trend" counted as the report
    step and was never analysed."""
    _model, final = _run(
        [
            Plan(steps=["Total revenue per region", "Monthly revenue trend"], rationale="two halves"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            Route(next="analyst", instruction="trend", reason="s", step_index=2),
            AIMessage(content="FINDING: trend is up\nEVIDENCE: monthly sums"),
            Route(next="reporter", instruction="write", reason="done", step_index=3),
            AIMessage(content="# Report"),
        ],
        monkeypatch, override_settings, "no-report-step", require_chart=False,
    )
    assert final["plan"][-1] == "Write the report"
    assert [f["text"] for f in final["findings"]] == ["North leads", "trend is up"]


def test_a_revised_plan_replaces_the_rejected_one(monkeypatch, override_settings) -> None:
    """#25: both plans stayed in the workers' context, unmarked."""
    override_settings(require_approval=True, require_chart=False)
    rejected = Plan(steps=["Average discount per product", "Write the report"], rationale="OLD")
    _install([rejected, *default_script()], monkeypatch)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "revise"}}
    graph.invoke(initial_state(CSV, "Which region drives revenue?"), config)
    graph.invoke(Command(resume="only regions, please"), config)
    final = graph.invoke(Command(resume="approve"), config)

    plans = [m for m in final["messages"] if str(m.content).startswith("Plan (")]
    assert len(plans) == 1, [str(p.content)[:60] for p in plans]
    assert "revised after feedback: only regions, please" in plans[0].content
    assert "discount" not in plans[0].content


def test_one_planner_timeout_does_not_end_the_run(monkeypatch, override_settings) -> None:
    """#10: a single failure at the planner went straight to END."""
    _model, final = _run([TimeoutError("Request timed out."), *default_script()],
                         monkeypatch, override_settings, "planner-retry")
    assert final.get("report_path")
    assert final["failure_count"] == 1


def test_the_analyst_always_sees_its_step(monkeypatch, override_settings) -> None:
    """#11: with parallel tool calls the dispatch message fell out of the
    window by the fourth turn; the instruction now lives in the system prompt."""
    model, _final = _run(
        [
            Plan(steps=["Revenue by region", "Report"], rationale="r"),
            Route(next="analyst", instruction="STEP-ONE: revenue per region", reason="s", step_index=1),
            AIMessage(content="FINDING: North leads\nEVIDENCE: x"),
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch, override_settings, "step-block", require_chart=False,
    )
    analyst_system = next(str(p[0].content) for p in model.seen_prompts
                          if "meticulous data analyst" in str(p[0].content))
    assert "The step you are working on now: STEP-ONE: revenue per region" in analyst_system


def test_the_two_tries_cap_holds_without_findings(monkeypatch, override_settings) -> None:
    """#24: with nothing retryable and no findings the analyst was sent out
    again and again; now the no-findings report is written."""
    from src.nodes.supervisor import supervisor_node

    override_settings(require_approval=False, require_chart=False)
    _install([Route(next="analyst", instruction="again", reason="y", step_index=1)], monkeypatch)
    update = supervisor_node({
        "question": "q", "csv_path": CSV, "plan": ["Region", "Report"],
        "addressed_steps": [], "attempts": [1, 1], "findings": [],
        "step_count": 5, "failure_count": 0,
    })
    assert update["next_worker"] == "reporter"


def test_charts_are_kept_when_the_charter_hits_its_turn_allowance(
    monkeypatch, override_settings, isolate_outputs
) -> None:
    """The other exit of the charter's loop: the per-step turn cap."""
    from tests.test_cleaning_and_charts import _bar

    _model, final = _run(
        [
            Plan(steps=["Revenue by region", "Report"], rationale="r"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            AIMessage(content="FINDING: North leads\nEVIDENCE: x"),
            Route(next="charter", instruction="bar", reason="s", step_index=1),
            _bar("capped.png", "c1"),  # turn 1 of 1; the next charter turn is the cap
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch, override_settings, "charter-cap", max_tool_rounds=1,
    )
    assert [Path(c).name for c in final["charts"]] == ["capped.png"]


# --- the cleaning log is computed, not remembered ---------------------------------

FIX_AND_DEDUPE = (
    'df.loc[df["order_id"] == "SO-1012", "revenue"] = 4326.98\n'
    'result = df.drop_duplicates("order_id")'
)


def test_the_cleaning_log_states_what_really_changed_even_if_the_model_misremembers() -> None:
    """A real report listed the removed duplicates as ... SO-1150, SO-1188.
    SO-1188 occurs once; the fifth duplicate is SO-1260. The ids came from
    the model's own description, which nothing checked."""
    command = save_cleaned_data.func(
        code=FIX_AND_DEDUPE,
        description="dropped duplicates SO-1003, SO-1040, SO-1077, SO-1150, SO-1188",
        state=S, tool_call_id="c",
    )
    log = command.update["cleaning_log"][0]
    verified = log.split("Verified from the data:")[1]
    assert "removed 5 rows (SO-1003, SO-1040, SO-1077, SO-1150, SO-1260)" in verified
    assert "SO-1188" not in verified
    assert "changed 1 value (SO-1012 revenue: 432,700.00 -> 4,326.98)" in verified


def test_a_reset_index_is_admitted_not_guessed() -> None:
    command = save_cleaned_data.func(
        code="result = df.drop_duplicates().reset_index(drop=True)",
        description="x", state=S, tool_call_id="c",
    )
    assert "cannot be listed" in command.update["cleaning_log"][0]


def test_the_reporter_is_told_to_trust_the_verified_part(monkeypatch, override_settings) -> None:
    from tests.test_cleaning_and_charts import _call

    model, _final = _run(
        [
            Plan(steps=["Clean and compare regions", "Report"], rationale="r"),
            Route(next="analyst", instruction="clean", reason="s", step_index=1),
            _call("save_cleaned_data", {"code": FIX_AND_DEDUPE,
                                        "description": "dropped SO-1188 and others"}, "c1"),
            AIMessage(content="FINDING: North leads\nEVIDENCE: x"),
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch, override_settings, "verified-log", require_chart=False,
    )
    prompt = str(model.seen_prompts[-1][-1])
    assert "ONLY from the verified part" in prompt
    assert "SO-1260" in prompt

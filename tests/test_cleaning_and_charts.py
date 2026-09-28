"""Cleaned data that every tool sees, and charts drawn from it.

The real run that motivated this file produced a correct report -- North
48.44%, SO-1012 corrected, trend +11,434/month -- and would have produced a
wrong chart, because the chart tool read the raw file: duplicates, the
432,700 typo and all. The report and its own chart would have disagreed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.dataset import load_df
from src.graph import build_graph
from src.llm import get_llm
from src.schemas import Plan, Route
from src.state import initial_state
from src.tools.charts import make_chart
from tests.fake_llm import scripted_model
from tests.test_graph import assert_script_matched

CSV = "data/sales.csv"
QUESTION = "Which region drives revenue, and is the trend going up?"

# The cleaning the real agent chose: drop 5 duplicates, fix one typo.
CLEANING = (
    'df.loc[df["order_id"] == "SO-1012", "revenue"] = 4326.98\n'
    'result = df.drop_duplicates("order_id")'
)

LLM_CONSUMERS = (
    "src.llm",
    "src.nodes.planner",
    "src.nodes.supervisor",
    "src.nodes.analyst",
    "src.nodes.charter",
    "src.nodes.reporter",
)


@pytest.fixture(autouse=True)
def clear_caches():
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


def _install(script, monkeypatch):
    model = scripted_model(script)
    for module in LLM_CONSUMERS:
        monkeypatch.setattr(f"{module}.get_llm", lambda *a, **k: model)
    return model


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _tool_output(final: dict, call_id: str) -> str:
    return next(
        str(m.content)
        for m in final["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == call_id
    )


# --- dates -------------------------------------------------------------------


def test_date_columns_are_really_parsed() -> None:
    """The pandas 3 regression: `dtype == "object"` stopped matching text,
    so no date was ever parsed, while the planner prompt promised the model
    that they were. Put the old comparison back in src/dataset.py and this
    fails on pandas 3."""
    df = load_df(CSV)
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"]), df["order_date"].dtype


# --- the whole path, as a real run takes it ------------------------------------


def test_cleaning_once_reaches_every_later_tool_and_the_chart(
    monkeypatch, override_settings
) -> None:
    override_settings(require_approval=False)
    model = _install(
        [
            Plan(steps=["Clean the data, then compare regions", "Report"],
                 rationale="validate, then answer"),
            Route(next="analyst", instruction="clean the data", reason="first", step_index=1),
            _call("save_cleaned_data",
                  {"code": CLEANING, "description": "dropped duplicate order_id rows; "
                   "set SO-1012 revenue to its formula value 4326.98"}, "clean"),
            _call("group_aggregate",
                  {"group_by": ["region"], "value_column": "revenue", "agg": "sum"}, "agg"),
            AIMessage(content="FINDING: North leads with 1,355,772.53\nEVIDENCE: sum by region"),
            Route(next="charter", instruction="monthly revenue line", reason="trend", step_index=2),
            _call("make_chart",
                  {"kind": "line", "x": "order_date", "y": "revenue", "agg": "sum",
                   "time_bucket": "month", "title": "Revenue rises through 2025",
                   "filename": "monthly_revenue.png"}, "chart"),
            AIMessage(content="CHART: monthly_revenue.png - rising"),
            Route(next="reporter", instruction="write it", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )

    graph = build_graph(checkpointer=InMemorySaver())
    final = graph.invoke(initial_state(CSV, QUESTION),
                         config={"configurable": {"thread_id": "clean"}})

    # 1. the cleaning was saved, and says what it did
    assert final.get("data_path"), "save_cleaned_data must record the working copy"
    assert "425 -> 420 rows" in final["cleaning_log"][0]

    # 2. a LATER tool, with no cleaning code of its own, saw the cleaned data:
    #    South is 452,830 after the fix, 881,203 before it.
    aggregated = _tool_output(final, "agg")
    assert "452,830" in aggregated, aggregated
    assert "881,203" not in aggregated

    # 3. the chart was drawn from the cleaned data, one point per month
    charted = _tool_output(final, "chart")
    assert "drawn from cleaned working data" in charted
    assert "12 monthly points" in charted
    # The raw file peaks in November at 679,477.51 -- the SO-1012 typo. The
    # cleaned data peaks in December. The label above only says which data
    # was *asked for*; this is the check that it was actually *drawn*.
    assert "peak 2025-12 = 363,111.86" in charted, charted
    assert "679,477" not in charted

    # 4. the path the report links to is a real file (no note glued onto it)
    assert final["charts"], "the charter must record the chart"
    assert Path(final["charts"][0]).exists(), final["charts"][0]

    # 5. the reporter was told what was cleaned
    reporter_prompt = str(model.seen_prompts[-1][-1].content)
    assert "Data cleaning applied" in reporter_prompt
    assert "SO-1012" in reporter_prompt


# --- save_cleaned_data refuses what is almost certainly a mistake -------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("result = len(df)", "must be a DataFrame"),
        ('result = df.drop(columns=["revenue"])', "lost columns"),
        ('result = df[df["region"] == "West"]', "almost always a bug"),
    ],
)
def test_suspicious_cleaning_is_refused_and_nothing_is_saved(code, expected) -> None:
    from src.tools.analysis import save_cleaned_data

    command = save_cleaned_data.func(
        code=code, description="x", state={"csv_path": CSV}, tool_call_id="c1"
    )

    assert "data_path" not in command.update, "a refused cleaning must not be saved"
    assert expected in command.update["messages"][0].content


# --- the chart tool ------------------------------------------------------------


def _chart(**kwargs) -> str:
    args = {"kind": "line", "x": "order_date", "y": "revenue", "agg": "sum",
            "title": "t", "filename": "c.png", "state": {"csv_path": CSV}}
    args.update(kwargs)
    return make_chart.invoke(args)


def test_a_trend_line_without_a_bucket_warns_about_its_grain() -> None:
    out = _chart()
    assert out.startswith("saved chart: ")
    assert "WARNING" in out and "time_bucket='month'" in out


def test_time_bucket_refuses_a_column_that_is_not_a_date() -> None:
    out = _chart(x="region", time_bucket="month")
    assert out.startswith("ERROR") and "date column" in out


# --- the supervisor must not rewrite the charter's instruction ----------------


def test_the_charter_may_draw_a_step_that_is_already_done(
    monkeypatch, override_settings
) -> None:
    """Charting is about finished steps. The done-step redirect is for the
    analyst only; applied to the charter it replaced "chart the trend" with
    the next pending plan step. Revert the `route.next == "analyst"` check in
    src/nodes/supervisor.py and this fails."""
    from src.nodes.supervisor import supervisor_node

    override_settings(require_approval=False)
    _install(
        [Route(next="charter", instruction="line chart of monthly revenue",
               reason="show trend", step_index=1)],
        monkeypatch,
    )
    state = {
        "question": QUESTION,
        "csv_path": CSV,
        "plan": ["Monthly trend", "Write the report"],
        "addressed_steps": [1],
        "findings": [{"source": "analyst", "text": "trend is up", "evidence": "slope"}],
        "step_count": 1,
        "failure_count": 0,
    }

    update = supervisor_node(state)

    assert update["next_worker"] == "charter"
    assert "line chart of monthly revenue" in update["messages"][0].content
    assert "Write the report" not in update["messages"][0].content


# --- every report with findings gets a chart ------------------------------------


def test_a_model_that_skips_the_charter_is_sent_there_anyway(
    monkeypatch, override_settings
) -> None:
    """The real run: SUPERVISOR_SYSTEM asked for charts, and the model went
    from the last finding straight to the reporter -- twice. Remove the
    require_chart block in src/nodes/supervisor.py and this fails."""
    override_settings(require_approval=False)
    _install(
        [
            Plan(steps=["Revenue by region", "Report"], rationale="r"),
            Route(next="analyst", instruction="revenue per region", reason="start", step_index=1),
            _call("group_aggregate",
                  {"group_by": ["region"], "value_column": "revenue", "agg": "sum"}, "agg"),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            # The model's own choice: skip the charter.
            Route(next="reporter", instruction="write it", reason="done", step_index=2),
            # What the code makes happen instead.
            _call("make_chart",
                  {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum",
                   "title": "North leads", "filename": "by_region.png"}, "chart"),
            AIMessage(content="CHART: by_region.png - North is tallest"),
            Route(next="reporter", instruction="write it", reason="done", step_index=2),
            AIMessage(content="# Report\n\n## Charts\n![North leads](by_region.png)\n"),
        ],
        monkeypatch,
    )

    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "force"}}
    )
    assert_script_matched(final)

    assert final["charts"], "the supervisor must route to the charter before the report"
    assert final["chart_attempted"] is True


@pytest.mark.parametrize(
    ("extra_state", "setting", "expected"),
    [
        ({}, True, "charter"),                              # the rule fires
        ({"chart_attempted": True}, True, "reporter"),      # but only once
        ({"charts": ["/x/a.png"]}, True, "reporter"),       # a chart exists
        ({}, False, "reporter"),                            # REQUIRE_CHART=false
    ],
)
def test_the_chart_rule_fires_once_and_only_when_needed(
    extra_state, setting, expected, monkeypatch, override_settings
) -> None:
    from src.nodes.supervisor import supervisor_node

    override_settings(require_approval=False, require_chart=setting)
    _install([Route(next="reporter", instruction="w", reason="r", step_index=1)], monkeypatch)
    state = {
        "question": QUESTION,
        "csv_path": CSV,
        "plan": ["Revenue by region", "Report"],
        "addressed_steps": [1],
        "findings": [{"source": "analyst", "text": "North leads", "evidence": "sum"}],
        "step_count": 1,
        "failure_count": 0,
        **extra_state,
    }

    assert supervisor_node(state)["next_worker"] == expected


# --- an empty Charts section is removed by code ---------------------------------


def test_an_empty_charts_section_is_removed() -> None:
    """Exactly what the model wrote on the real run, despite the prompt."""
    from src.nodes.reporter import drop_section

    report = (
        "# Title\n\n## What the data shows\n- North leads\n\n"
        "## Charts\nNo charts available.\n\n"
        "## Caveats\n- cleaned\n"
    )
    out = drop_section(report, "Charts")

    assert "## Charts" not in out and "No charts available" not in out
    assert "## What the data shows\n- North leads" in out
    assert "## Caveats\n- cleaned" in out, "the section after it must survive"


def test_drop_section_handles_a_last_section_and_a_missing_one() -> None:
    from src.nodes.reporter import drop_section

    assert drop_section("# T\n\n## Charts\nnone\n", "Charts").strip() == "# T"
    assert drop_section("# T\n\n## Caveats\n- x\n", "Charts") == "# T\n\n## Caveats\n- x\n"


def test_the_written_report_has_no_empty_charts_section(
    monkeypatch, override_settings
) -> None:
    """The function above is tested on its own; this checks the reporter
    actually uses it, on the file that lands in outputs/. Without this, the
    call could be deleted and every test would still pass -- which is exactly
    what a mutation run showed."""
    override_settings(require_approval=False, require_chart=False)
    _install(
        [
            Plan(steps=["Revenue by region", "Report"], rationale="r"),
            Route(next="analyst", instruction="revenue per region", reason="s", step_index=1),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            Route(next="reporter", instruction="write", reason="d", step_index=2),
            AIMessage(content="# T\n\n## Answer\nNorth.\n\n## Charts\nNo charts available.\n\n"
                              "## Caveats\n- none\n"),
        ],
        monkeypatch,
    )

    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "nochart"}}
    )
    assert_script_matched(final)
    written = Path(final["report_path"]).read_text(encoding="utf-8")

    assert "## Charts" not in written
    assert "## Caveats" in written


# --- the step budget is shared out, not first-come-first-served -----------------


def test_one_step_cannot_take_more_than_its_share_of_turns(
    monkeypatch, override_settings
) -> None:
    """After MAX_TOOL_ROUNDS turns on one step, the analyst stops calling
    tools and the finding is extracted from the last tool output. Remove the
    cap in analyst_node and the analyst makes a third tool-bound call, which
    this script does not allow for."""
    from src.schemas import FindingOut

    override_settings(require_approval=False, require_chart=False, max_tool_rounds=2)
    model = _install(
        [
            Plan(steps=["Validate revenue", "Report"], rationale="r"),
            Route(next="analyst", instruction="validate", reason="first", step_index=1),
            _call("column_stats", {"column": "revenue"}, "c1"),
            _call("column_stats", {"column": "units"}, "c2"),
            # turn allowance used up: wrap-up extracts, no third tool call
            FindingOut(has_finding=True, statement="units range from 2 to 68",
                       evidence="column_stats(units)"),
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )

    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "cap"}}
    )
    assert_script_matched(final)

    analyst_calls = [b for b in model.tool_bindings if "run_pandas" in b]
    assert len(analyst_calls) == 2, "exactly MAX_TOOL_ROUNDS tool-bound turns"
    assert final["findings"][0]["text"] == "units range from 2 to 68"
    assert final["addressed_steps"] == [1], "the salvaged finding closes the step"


def test_a_run_that_runs_out_of_budget_still_gets_its_chart(
    monkeypatch, override_settings
) -> None:
    """The real run: the analyst kept checking, the budget ran out, and the
    budget guardrail sent the run straight to the reporter -- no chart, and
    half the question unanswered. Now the last CHART_RESERVE steps are kept
    for the charter. Remove the reserve branch in supervisor_node and this
    fails."""
    override_settings(require_approval=False, max_steps=9, max_tool_rounds=10)
    _install(
        [
            Plan(steps=["Revenue by region", "Validate everything", "Report"], rationale="r"),
            Route(next="analyst", instruction="revenue per region", reason="s", step_index=1),
            _call("group_aggregate",
                  {"group_by": ["region"], "value_column": "revenue", "agg": "sum"}, "a1"),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            Route(next="analyst", instruction="check everything", reason="s", step_index=2),
            # the analyst would go on checking forever; the analysis limit cuts it
            _call("column_stats", {"column": "revenue"}, "a2"),
            # reserved steps: the code sends the charter, no model decision
            _call("make_chart", {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum",
                                 "title": "North leads", "filename": "r.png"}, "ch"),
            AIMessage(content="CHART: r.png - North is tallest"),
            Route(next="reporter", instruction="write", reason="done", step_index=3),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )

    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "reserve"}}
    )
    assert_script_matched(final)

    assert final["charts"], "the reserved steps must produce the chart"
    assert final.get("report_path")
    assert final["step_count"] <= 9 + 1, "and the budget still holds"
    assert any("reserved steps" in str(m.content) for m in final["messages"])


def test_every_dispatch_starts_with_a_fresh_turn_allowance(
    monkeypatch, override_settings
) -> None:
    """The allowance is per plan step, not per run. If the supervisor did
    not reset `worker_turns`, step 2 would inherit step 1's used-up count
    and wrap up before making a single tool call."""
    override_settings(require_approval=False, require_chart=False, max_tool_rounds=2)
    _install(
        [
            Plan(steps=["Region", "Trend", "Report"], rationale="r"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            _call("group_aggregate",
                  {"group_by": ["region"], "value_column": "revenue", "agg": "sum"}, "r1"),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),  # turn 2 of 2
            Route(next="analyst", instruction="trend", reason="s", step_index=2),
            _call("column_stats", {"column": "revenue"}, "t1"),  # needs a fresh allowance
            AIMessage(content="FINDING: revenue mean is 7,652\nEVIDENCE: column_stats"),
            Route(next="reporter", instruction="write", reason="done", step_index=3),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )

    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "fresh"}}
    )
    assert_script_matched(final)

    assert [f["text"] for f in final["findings"]] == [
        "North leads", "revenue mean is 7,652",
    ]


# --- the wrap-up asks about the step, not the whole question -------------------


def test_the_wrap_up_is_asked_about_this_step_and_all_its_outputs(
    monkeypatch, override_settings
) -> None:
    """The first wrap-up sent the user's overall question and a "reply" that
    said nothing, to an extractor told to find nothing unless the reply
    states something. It answered "no finding" every time, and on a real run
    one validation step was dispatched three times."""
    from src.schemas import FindingOut

    override_settings(require_approval=False, require_chart=False, max_tool_rounds=2)
    model = _install(
        [
            Plan(steps=["Check revenue against the formula", "Report"], rationale="r"),
            Route(next="analyst", instruction="compare revenue with the formula; "
                  "list rows off by more than $0.50", reason="validate", step_index=1),
            _call("run_pandas", {"code": "result = len(df)"}, "w1"),
            _call("column_stats", {"column": "revenue"}, "w2"),
            FindingOut(has_finding=True, statement="0 rows differ by more than $0.50",
                       evidence="max diff 0.27"),
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )

    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "wrap"}}
    )
    assert_script_matched(final)

    wrap_up = next(
        str(p[-1][1] if isinstance(p[-1], tuple) else p[-1].content)
        for p in model.seen_prompts
        if "Tool outputs, oldest first" in str(p[-1])
    )
    assert "compare revenue with the formula" in wrap_up, "the STEP, not the question"
    assert "425" in wrap_up, "the first tool output (len(df)) is included too"
    assert "What the analyst then wrote" not in wrap_up
    assert final["findings"][0]["text"] == "0 rows differ by more than $0.50"


# --- the plan is finished before the report ------------------------------------


def _supervise(route, monkeypatch, override_settings, **state_overrides):
    from src.nodes.supervisor import supervisor_node

    override_settings(require_approval=False, require_chart=False)
    _install([route], monkeypatch)
    state = {
        "question": QUESTION,
        "csv_path": CSV,
        "plan": ["Validate", "Revenue by region", "Monthly trend", "Report"],
        "addressed_steps": [1],
        "attempts": [1],
        "findings": [{"source": "analyst", "text": "validated", "evidence": "0 rows"}],
        "step_count": 5,
        "failure_count": 0,
    }
    state.update(state_overrides)
    return supervisor_node(state)


@pytest.mark.parametrize("worker", ["reporter", "charter"])
def test_no_report_and_no_chart_while_a_plan_step_is_pending(
    worker, monkeypatch, override_settings
) -> None:
    """The real run: "revenue by region" and "monthly trend" were sent to the
    charter, then the report was written with neither analysed."""
    update = _supervise(Route(next=worker, instruction="x", reason="y", step_index=2),
                        monkeypatch, override_settings)
    assert update["next_worker"] == "analyst"
    assert update["current_step"] == 2
    assert update["attempts"] == [2], "the dispatch is counted"


def test_a_step_tried_twice_is_skipped_for_the_next_one(monkeypatch, override_settings) -> None:
    update = _supervise(Route(next="analyst", instruction="validate again", reason="y",
                              step_index=2),
                        monkeypatch, override_settings, attempts=[1, 2, 2])
    assert update["next_worker"] == "analyst"
    assert update["current_step"] == 3, "step 2 had its two tries"


def test_when_nothing_is_left_to_try_the_report_is_written(monkeypatch, override_settings) -> None:
    update = _supervise(Route(next="analyst", instruction="more", reason="y", step_index=3),
                        monkeypatch, override_settings,
                        addressed_steps=[1, 2], attempts=[1, 2, 3, 3])
    assert update["next_worker"] == "reporter"


def test_no_redirect_when_the_budget_cannot_fit_another_try(monkeypatch, override_settings) -> None:
    update = _supervise(Route(next="reporter", instruction="write", reason="y", step_index=4),
                        monkeypatch, override_settings, step_count=28)  # 30 - 28 < 3
    assert update["next_worker"] == "reporter"


def test_ad_hoc_work_outside_the_plan_is_allowed(monkeypatch, override_settings) -> None:
    update = _supervise(Route(next="analyst", instruction="ad hoc", reason="y", step_index=0),
                        monkeypatch, override_settings)
    assert update["next_worker"] == "analyst"
    assert update["current_step"] == 0


def test_a_second_charter_turn_does_not_announce_the_first_chart_again() -> None:
    from src.nodes.charter import _new_charts

    tool = ToolMessage("saved chart: /o/a.png\ndrawn from cleaned working data", tool_call_id="1")
    tool2 = ToolMessage("saved chart: /o/b.png\ndrawn from cleaned working data", tool_call_id="2")
    state = {"messages": [tool, tool2], "charts": ["/o/a.png"]}
    assert _new_charts(state) == ["/o/b.png"]


def test_each_charter_turn_reports_only_the_charts_it_drew(
    monkeypatch, override_settings
) -> None:
    """What the CLI prints is each node's update. Two charter turns must
    announce a.png, then b.png -- not a.png, then a.png and b.png."""
    override_settings(require_approval=False)
    _install(
        [
            Plan(steps=["Revenue by region", "Report"], rationale="r"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            Route(next="charter", instruction="bar", reason="s", step_index=1),
            _call("make_chart", {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum",
                                 "title": "a", "filename": "a.png"}, "ca"),
            AIMessage(content="CHART: a.png - bars"),
            Route(next="charter", instruction="line", reason="s", step_index=1),
            _call("make_chart", {"kind": "line", "x": "order_date", "y": "revenue",
                                 "agg": "sum", "time_bucket": "month",
                                 "title": "b", "filename": "b.png"}, "cb"),
            AIMessage(content="CHART: b.png - line"),
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "twice"}}

    announced = [
        [Path(c).name for c in update["charter"]["charts"]]
        for update in graph.stream(initial_state(CSV, QUESTION), config=config,
                                   stream_mode="updates")
        if "charter" in update and update["charter"].get("charts")
    ]

    assert_script_matched(graph.get_state(config).values)
    assert announced == [["a.png"], ["b.png"]]


# --- the review's graph findings, as they must behave now ------------------------

GA = {"group_by": ["region"], "value_column": "revenue", "agg": "sum"}


def _bar(filename: str, call_id: str) -> AIMessage:
    return _call("make_chart", {"kind": "bar", "x": "region", "y": "revenue", "agg": "sum",
                                "title": "North leads", "filename": filename}, call_id)


def test_charts_drawn_before_a_budget_cut_are_kept(monkeypatch, override_settings) -> None:
    """Review R4. The charter recorded charts only in its closing reply, so a
    budget cut before that reply lost them: PNGs on disk, "(none)" in the
    report, Charts section removed. Now every charter turn records what the
    previous calls drew. Here the fourth call is cut; three charts survive."""
    override_settings(require_approval=False, max_steps=8)
    model = _install(
        [
            Plan(steps=["Revenue by region", "Report"], rationale="r"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            _call("group_aggregate", GA, "a1"),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            # reserve reached: the code sends the charter, which keeps drawing
            _bar("c1.png", "c1"), _bar("c2.png", "c2"), _bar("c3.png", "c3"),
            _bar("c4.png", "c4"),  # cut here by the budget, never executed
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )
    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "r4"}}
    )
    assert_script_matched(final)

    assert sorted(Path(c).name for c in final["charts"]) == ["c1.png", "c2.png", "c3.png"]
    reporter_prompt = str(model.seen_prompts[-1][-1])
    assert "c3.png" in reporter_prompt


def test_a_prose_answer_without_a_tool_is_not_credited_with_old_numbers(
    monkeypatch, override_settings
) -> None:
    """Review R8. With no tool call on step 2, the extractor was handed step
    1's region table as "the most recent tool output" -- and step 2 could be
    marked done with step 1's numbers. Now there is nothing to extract, the
    step stays open, and the second try does the work."""
    override_settings(require_approval=False, require_chart=False)
    model = _install(
        [
            Plan(steps=["Revenue by region", "Monthly trend", "Report"], rationale="r"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            _call("group_aggregate", GA, "a1"),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            Route(next="analyst", instruction="monthly trend", reason="s", step_index=2),
            AIMessage(content="The trend looks upward overall."),  # no tool, no label
            Route(next="analyst", instruction="monthly trend, with numbers", reason="retry",
                  step_index=2),
            _call("run_pandas", {"code": "result = df.set_index('order_date').revenue"
                                         ".resample('MS').sum().iloc[[0, -1]]"}, "t1"),
            AIMessage(content="FINDING: revenue rose from 228,553 to 363,112\nEVIDENCE: monthly sums"),
            Route(next="reporter", instruction="write", reason="done", step_index=3),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )
    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "r8"}}
    )
    assert_script_matched(final)

    assert [f["text"] for f in final["findings"]] == [
        "North leads", "revenue rose from 228,553 to 363,112",
    ]
    assert final["addressed_steps"] == [1, 2]
    assert not any("unstructured reply" in str(p[0].content) for p in model.seen_prompts), (
        "the extractor must not be asked when no tool ran for the step"
    )


def test_work_done_before_a_budget_cut_still_becomes_a_finding(
    monkeypatch, override_settings
) -> None:
    """Review R10. A thorough model computed the monthly totals, went to
    "verify once more", and the budget cut it off: the trend was computed and
    the report still said it was not measured. The last turn the budget
    allows is now a wrap-up over what the step already computed."""
    from src.schemas import FindingOut

    override_settings(require_approval=False, max_steps=12)
    model = _install(
        [
            Plan(steps=["Revenue by region", "Monthly trend", "Report"], rationale="r"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            _call("group_aggregate", GA, "a1"),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            Route(next="analyst", instruction="monthly trend", reason="s", step_index=2),
            _call("run_pandas", {"code": "result = df.set_index('order_date').revenue"
                                         ".resample('MS').sum()"}, "t1"),
            _call("run_pandas", {"code": "result = 1"}, "t2"),  # "verifying once more"
            # the budget allows one more turn: a wrap-up, not a third call
            FindingOut(has_finding=True, statement="monthly revenue rose from 228,553.07 to 363,111.86",
                       evidence="t1 monthly sums"),
            _bar("region.png", "c1"),
            AIMessage(content="CHART: region.png - North is tallest"),
            Route(next="reporter", instruction="write", reason="done", step_index=3),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )
    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "r10"}}
    )
    assert_script_matched(final)

    assert "monthly revenue rose from 228,553.07 to 363,111.86" in [f["text"] for f in final["findings"]]
    assert final["addressed_steps"] == [1, 2]
    wrap = next(str(p[-1][1]) for p in model.seen_prompts if "Tool outputs, oldest first" in str(p[-1]))
    assert "228553.07" in wrap, "the wrap-up saw the monthly totals the step computed"
    assert final["charts"], "and the chart reserve still held"


def test_a_failed_extraction_is_written_down_not_swallowed(monkeypatch, override_settings) -> None:
    """Both "no finding" and "the extraction call failed" used to come back as
    None, so a failing extractor left no trace at all."""
    override_settings(require_approval=False, require_chart=False, max_tool_rounds=1)
    _install(
        [
            Plan(steps=["Revenue by region", "Report"], rationale="r"),
            Route(next="analyst", instruction="region", reason="s", step_index=1),
            _call("group_aggregate", GA, "a1"),
            RuntimeError("503 extractor unavailable"),  # the wrap-up call fails
            # step 1 is still open, so "write the report" becomes a second try
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="FINDING: North leads\nEVIDENCE: sum by region"),
            Route(next="reporter", instruction="write", reason="done", step_index=2),
            AIMessage(content="# Report"),
        ],
        monkeypatch,
    )
    final = build_graph(checkpointer=InMemorySaver()).invoke(
        initial_state(CSV, QUESTION), config={"configurable": {"thread_id": "xf"}}
    )
    assert_script_matched(final)
    assert final["findings"][0]["text"] == "North leads", "the second try still works"
    assert any("finding extraction failed" in e and "503" in e for e in final["errors"])
    assert (final.get("failure_count") or 0) == 0, "recorded, not counted against the budget"

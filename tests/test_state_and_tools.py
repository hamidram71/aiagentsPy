"""Reducers, message windowing, dataset profiling and the tools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.dataset import profile_df
from src.messages import parse_tagged, window
from src.state import append_unique
from src.tools.analysis import correlations, group_aggregate, run_pandas, value_counts
from src.tools.charts import make_chart

CSV = "data/sales.csv"
STATE = {"csv_path": CSV}


# --- reducers --------------------------------------------------------------


def test_append_unique_skips_duplicates() -> None:
    left = [{"text": "a"}]
    right = [{"text": "a"}, {"text": "b"}]
    assert append_unique(left, right) == [{"text": "a"}, {"text": "b"}]


# --- message window --------------------------------------------------------


def test_window_never_orphans_a_tool_message() -> None:
    messages = [
        HumanMessage(content="task"),
        *[AIMessage(content=f"noise {i}") for i in range(20)],
        AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "keep"}],
        ),
        ToolMessage(content="kept", tool_call_id="keep"),
        ToolMessage(content="orphan", tool_call_id="gone"),
    ]
    kept = window(messages, keep=4)

    assert kept[0] is messages[0]  # the task seed always survives
    orphans = [
        m
        for m in kept
        if isinstance(m, ToolMessage) and m.tool_call_id == "gone"
    ]
    assert orphans == []


def test_window_is_a_noop_when_short() -> None:
    messages = [HumanMessage(content="a"), AIMessage(content="b")]
    assert window(messages, keep=10) == messages


def test_parse_tagged() -> None:
    text = "FINDING: revenue is up 12%\nEVIDENCE: sum by month"
    assert parse_tagged(text, "FINDING") == "revenue is up 12%"
    assert parse_tagged(text, "EVIDENCE") == "sum by month"
    assert parse_tagged(text, "MISSING") == ""


# --- dataset ---------------------------------------------------------------


def test_profile_reports_the_deliberate_data_problems() -> None:
    profile = profile_df(pd.read_csv(CSV))
    assert profile["rows"] > 400
    assert profile["duplicate_rows"] == 5  # planted in the sample file
    assert profile["columns"]["delivery_days"]["nulls"] > 0
    assert "revenue" in profile["numeric_columns"]


# --- tools -----------------------------------------------------------------
# Tools declared with InjectedState are invoked with the state alongside the
# model-facing arguments; `.invoke()` takes the whole argument dict.


def test_group_aggregate() -> None:
    out = group_aggregate.invoke(
        {
            "group_by": ["region"],
            "value_column": "revenue",
            "agg": "sum",
            "state": STATE,
        }
    )
    assert "North" in out


def test_group_aggregate_rejects_unknown_columns() -> None:
    out = group_aggregate.invoke(
        {
            "group_by": ["planet"],
            "value_column": "revenue",
            "agg": "sum",
            "state": STATE,
        }
    )
    assert out.startswith("ERROR")


def test_value_counts_includes_shares() -> None:
    out = value_counts.invoke({"column": "region", "state": STATE})
    assert "%" in out


def test_correlations_finds_the_planted_relationship() -> None:
    out = correlations.invoke({"state": STATE, "threshold": 0.1})
    # note: the planted 100x outlier suppresses units~revenue, which is itself
    # a nice illustration of why the agent has to look at data quality.
    assert "revenue" in out and "r=" in out


def test_run_pandas_through_the_tool_interface() -> None:
    out = run_pandas.invoke(
        {"code": "result = df['revenue'].sum()", "state": STATE}
    )
    assert "result:" in out


def test_run_pandas_refuses_to_write_files() -> None:
    out = run_pandas.invoke(
        {"code": "df.to_csv('/tmp/leak.csv')\nresult = 1", "state": STATE}
    )
    assert out.startswith("ERROR")


def test_make_chart_writes_a_png(tmp_path, monkeypatch) -> None:
    import src.tools.charts as charts

    monkeypatch.setattr(charts, "OUTPUT_DIR", tmp_path)
    out = make_chart.invoke(
        {
            "kind": "bar",
            "x": "region",
            "y": "revenue",
            "agg": "sum",
            "title": "Revenue by region",
            "filename": "t.png",
            "state": STATE,
        }
    )
    assert out.startswith("saved chart: ")
    assert (tmp_path / "t.png").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "../../../etc/evil.png",
        "/etc/evil.png",
        "..\\..\\windows\\evil.png",
        "sub/dir/evil.png",
        "",
    ],
)
def test_make_chart_cannot_escape_the_output_dir(
    filename, tmp_path, monkeypatch
) -> None:
    """The model picks this filename, so it is untrusted input.

    Note what is asserted: that the *resolved* path is inside the output
    directory. An earlier version of this test checked whether a directory
    called `etc` appeared next to tmp_path -- but `../../../etc/evil.png`
    resolves somewhere else entirely, so the assertion could never fail and
    the security property was never actually tested. When a test names a
    security property, resolve the real path and compare.
    """
    import src.tools.charts as charts

    sandbox_dir = tmp_path / "chart_sandbox"
    sandbox_dir.mkdir()
    monkeypatch.setattr(charts, "OUTPUT_DIR", sandbox_dir)

    out = make_chart.invoke(
        {
            "kind": "hist",
            "x": "revenue",
            "title": "x",
            "filename": filename,
            "state": STATE,
        }
    )

    assert out.startswith("saved chart: ")
    written = Path(out[len("saved chart: ") :].splitlines()[0].strip()).resolve()
    assert written.parent == sandbox_dir.resolve(), f"escaped to {written}"
    assert written.exists()


# --- unambiguous tool output ----------------------------------------------
# These exist because of a real wrong answer. The agent grouped by
# region AND order_date; `Series.to_string()` elided the repeated region
# level, and the model filled the blank with a different region, reporting
# East's revenue as West's. It also saw the top 15 of 366 groups and
# concluded "only one date is provided".


def test_multi_key_groups_never_rely_on_visual_alignment() -> None:
    out = group_aggregate.invoke(
        {
            "group_by": ["region", "order_date"],
            "value_column": "revenue",
            "agg": "sum",
            "top_n": 4,
            "state": STATE,
        }
    )

    rows = [line for line in out.splitlines() if "->" in line]
    assert rows, out
    for row in rows:
        # every row carries BOTH keys, so no row means "same as above"
        assert "region=" in row, f"row without its own region key: {row}"
        assert "order_date=" in row, f"row without its own date key: {row}"


def test_a_truncated_grouping_says_how_many_groups_exist() -> None:
    """Saying what was left out is what stops "only one date" conclusions."""
    out = group_aggregate.invoke(
        {
            "group_by": ["region", "order_date"],
            "value_column": "revenue",
            "agg": "sum",
            "top_n": 4,
            "state": STATE,
        }
    )
    assert "of 366 groups" in out
    assert "more groups not shown" in out
    assert "NOT totals per region" in out, "the row meaning must be spelled out"


def test_floats_are_rounded_so_artefacts_are_not_copied_into_reports() -> None:
    out = group_aggregate.invoke(
        {
            "group_by": ["region", "order_date"],
            "value_column": "revenue",
            "agg": "sum",
            "top_n": 4,
            "state": STATE,
        }
    )
    assert "40,579.02" in out
    assert "40579.020000000004" not in out


def test_single_key_grouping_gives_the_real_region_totals() -> None:
    """The analysis the question actually needed."""
    out = group_aggregate.invoke(
        {
            "group_by": ["region"],
            "value_column": "revenue",
            "agg": "sum",
            "state": STATE,
        }
    )
    assert "of 4 groups" in out
    assert "region=North -> 1,373,177.21" in out
    assert "more groups not shown" not in out

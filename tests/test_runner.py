"""src/runner.py: what the CLI and the web app share.

The web app runs several people's analyses at once, so the things tested
here are the ones that break only when runs overlap: each run's files in its
own folder, meta.json telling the truth about where a run stands, and the
usage file getting exactly one line per finished run.
"""

from __future__ import annotations

import contextvars
import csv
import threading
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src import runner
from src.graph import build_graph
from src.llm import get_llm
from tests.fake_llm import default_script, scripted_model
from tests.test_graph import LLM_CONSUMERS, assert_script_matched

CSV = Path("data/sales.csv")
QUESTION = "Which region drives revenue?"


def _install(model, monkeypatch) -> None:
    for module in LLM_CONSUMERS:
        monkeypatch.setattr(f"{module}.get_llm", lambda *a, **k: model)


def _with_usage(script: list) -> list:
    """The default script, with token counts on every plain model reply."""
    out = []
    for item in script:
        if isinstance(item, AIMessage):
            item = item.model_copy(update={"usage_metadata": {
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
            }})
        out.append(item)
    return out


def test_a_run_moves_from_approval_to_done_and_is_recorded(
    tmp_path, monkeypatch, override_settings
) -> None:
    override_settings(require_approval=True)
    get_llm.cache_clear()
    _install(scripted_model(_with_usage(default_script())), monkeypatch)
    runs = tmp_path / "runs"
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "r1"}}

    payload = runner.start_payload(CSV, QUESTION, "r1", user="hamid", runs_dir=runs)
    folder = runs / "r1"
    assert runner.read_meta(folder)["status"] == "running"

    seen: list[str] = []
    first = runner.advance(graph, payload, config, on_update=lambda n, u: seen.append(n))
    assert first.status == "approval"
    assert first.approval["plan"] == ["Total revenue per region", "Write the report"]
    assert runner.read_meta(folder)["status"] == "waiting for approval"
    assert "planner" in seen

    done = runner.advance(graph, Command(resume="approve"), config)
    assert_script_matched(done.values)
    assert done.status == "done"

    # Everything the run wrote is inside its own folder.
    assert Path(done.report_path).parent == folder
    assert [Path(c).parent for c in done.values["charts"]] == [folder]

    meta = runner.read_meta(folder)
    assert meta["status"] == "done"
    assert meta["user"] == "hamid"
    assert meta["charts"] == ["revenue_by_region.png"]
    # 5 plain replies in the script x 100 input / 10 output tokens each.
    assert meta["usage"] == {"input_tokens": 500, "output_tokens": 50}
    assert done.usage == {"input_tokens": 500, "output_tokens": 50}

    with (tmp_path / "usage.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # One line per advance: the planning part (the fake planner reports no
    # tokens) and the rest. Summed per user, they give the run's total.
    assert [r["part"] for r in rows] == ["up to the plan approval", "finished"]
    assert runner.usage_by_user(tmp_path / "usage.csv") == {
        "hamid": {"input_tokens": 500, "output_tokens": 50}
    }

    assert [m["thread_id"] for m in runner.list_runs(runs_dir=runs)] == ["r1"]
    assert runner.list_runs(runs_dir=runs, user="someone else") == []


def test_copy_input_keeps_the_data_inside_the_run(tmp_path) -> None:
    """An uploaded file lives in a temp location; the run must not depend on it."""
    payload = runner.start_payload(CSV, QUESTION, "r2", runs_dir=tmp_path, copy_input=True)
    assert Path(payload["csv_path"]).parent == tmp_path / "r2"
    assert Path(payload["csv_path"]).read_bytes() == CSV.read_bytes()
    assert runner.read_meta(tmp_path / "r2")["csv"] == "sales.csv"


def test_two_runs_at_once_keep_their_own_charts(
    tmp_path, monkeypatch, override_settings
) -> None:
    """Both scripts save a chart called revenue_by_region.png. With one shared
    outputs/ folder, the second overwrote the first -- and the first report
    showed a colleague's chart. Each run now writes into its own folder."""
    override_settings(require_approval=False)
    get_llm.cache_clear()

    # One scripted model per run; which one a call gets depends on the run
    # it belongs to. LangGraph carries context variables into its threads.
    current = contextvars.ContextVar("model")
    for module in LLM_CONSUMERS:
        monkeypatch.setattr(f"{module}.get_llm", lambda *a, **k: current.get())

    runs = tmp_path / "runs"
    graph = build_graph(checkpointer=InMemorySaver())
    barrier = threading.Barrier(2)
    results: dict[str, runner.Outcome] = {}
    failures: list[BaseException] = []

    def one(thread_id: str) -> None:
        try:
            current.set(scripted_model(default_script()))
            payload = runner.start_payload(CSV, QUESTION, thread_id, runs_dir=runs)
            barrier.wait(timeout=10)
            config = {"configurable": {"thread_id": thread_id}}
            results[thread_id] = runner.advance(graph, payload, config)
        except BaseException as exc:  # noqa: BLE001 -- reported below
            failures.append(exc)

    threads = [threading.Thread(target=one, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not failures, failures

    for thread_id in ("a", "b"):
        outcome = results[thread_id]
        assert_script_matched(outcome.values)
        chart = Path(outcome.values["charts"][0])
        assert chart == runs / thread_id / "revenue_by_region.png"
        assert chart.exists()
        assert Path(outcome.report_path).parent == runs / thread_id

    with (tmp_path / "usage.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_describe_returns_lines_and_prints_nothing(capsys) -> None:
    lines = runner.describe("planner", {"plan": ["a", "b"]})
    assert lines == ["plan:", "  1. a", "  2. b"]
    assert runner.describe("analyst", {"errors": ["boom"]}) == ["! boom"]
    assert capsys.readouterr().out == ""


def test_the_cli_prints_the_same_lines(capsys) -> None:
    import main

    main.describe("planner", {"plan": ["a"]})
    main.describe("reporter", {"report_path": "x.md"})
    assert capsys.readouterr().out.splitlines() == [
        "  [planner] plan:",
        "        1. a",
        "  [reporter] report -> x.md",
    ]

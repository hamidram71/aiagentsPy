"""src/jobs.py: runs in background threads, for the web app."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src import jobs, runner
from src.graph import build_graph
from src.llm import get_llm
from tests.fake_llm import default_script, scripted_model
from tests.test_graph import LLM_CONSUMERS, _refusal, assert_script_matched

CSV = Path("data/sales.csv")
QUESTION = "Which region drives revenue?"


def _install(script, monkeypatch) -> None:
    get_llm.cache_clear()
    model = scripted_model(script)
    for module in LLM_CONSUMERS:
        monkeypatch.setattr(f"{module}.get_llm", lambda *a, **k: model)


def _wait(job: jobs.Job, seconds: float = 60) -> jobs.Job:
    deadline = time.time() + seconds
    while job.active:
        assert time.time() < deadline, f"job still running: {job.lines[-3:]}"
        time.sleep(0.05)
    return job


@pytest.fixture
def manager():
    return jobs.JobManager(build_graph(checkpointer=InMemorySaver()))


def test_a_run_pauses_for_approval_and_finishes_after_it(
    manager, monkeypatch, override_settings
) -> None:
    override_settings(require_approval=True)
    _install(default_script(), monkeypatch)
    payload = runner.start_payload(CSV, QUESTION, "j1")

    job = _wait(manager.start("j1", payload))
    assert job.status == "approval"
    assert job.outcome.approval["plan"][0] == "Total revenue per region"
    assert any(line.startswith("[planner]") for line in job.lines)

    job = _wait(manager.start("j1", Command(resume="approve")))
    assert job.status == "done", job.error
    assert_script_matched(job.outcome.values)
    assert Path(job.outcome.report_path).exists()
    # The progress from before the approval is still there.
    assert any(line.startswith("[planner]") for line in job.lines)
    assert any(line.startswith("[reporter]") for line in job.lines)
    assert runner.read_meta(runner.run_folder("j1"))["status"] == "done"


def test_a_failed_run_says_why_and_is_marked_failed(
    manager, monkeypatch, override_settings
) -> None:
    override_settings(require_approval=False)
    script = default_script()
    _install([script[0], script[1], _refusal()], monkeypatch)

    job = _wait(manager.start("j2", runner.start_payload(CSV, QUESTION, "j2")))

    from src.errors import FatalModelError

    assert job.status == "failed"
    assert isinstance(job.error, FatalModelError)
    meta = runner.read_meta(runner.run_folder("j2"))
    assert meta["status"] == "failed"
    assert "FatalModelError" in meta["error"]

    # Fixed, it continues from the checkpoint -- the analyst, not the start.
    _install(script[2:], monkeypatch)
    job = _wait(manager.start("j2", None))
    assert job.status == "done", job.error
    assert job.outcome.report_path


def test_stop_ends_the_run_after_the_current_step(
    manager, monkeypatch, override_settings
) -> None:
    override_settings(require_approval=False)
    _install(default_script(), monkeypatch)

    # Hold the run inside its first progress report until stop() is called,
    # so the test does not race a run that might finish first.
    reached, release = threading.Event(), threading.Event()
    real = runner.describe

    def held(node, update):
        reached.set()
        release.wait(10)
        return real(node, update)

    monkeypatch.setattr(runner, "describe", held)
    job = manager.start("j3", runner.start_payload(CSV, QUESTION, "j3"))
    assert reached.wait(10)
    manager.stop("j3")
    release.set()

    assert _wait(job).status == "stopped"
    assert runner.read_meta(runner.run_folder("j3"))["status"] == "stopped"
    assert not job.outcome, "a stopped run has not reached a report"
    # Tokens spent before the stop are still counted.
    usage = (runner.USAGE_FILE).read_text(encoding="utf-8")
    assert ",stopped early," in usage


def test_the_same_run_cannot_be_started_twice_at_once(
    manager, monkeypatch, override_settings
) -> None:
    override_settings(require_approval=False)
    _install(default_script(), monkeypatch)
    release = threading.Event()
    real = runner.describe
    monkeypatch.setattr(runner, "describe", lambda n, u: (release.wait(10), real(n, u))[1])

    job = manager.start("j4", runner.start_payload(CSV, QUESTION, "j4"))
    try:
        with pytest.raises(RuntimeError, match="already running"):
            manager.start("j4", None)
        assert manager.active_count() == 1
    finally:
        release.set()
    assert _wait(job).status == "done"
    assert manager.active_count() == 0

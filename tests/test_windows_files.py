"""Windows file semantics, on any system.

On Windows a file that is open cannot be replaced or deleted: `os.replace`
raises PermissionError ("[WinError 5] Zugriff verweigert"). Linux and macOS
allow it, so this suite, run on Linux, never saw the bug it found on a real
Windows machine: the web page read meta.json while a run thread replaced
it, the thread died, and the page showed "Working ..." forever.

These tests make Linux behave like Windows where it matters.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src import jobs, runner
from src.graph import build_graph
from tests.test_jobs import CSV, QUESTION, _install, _wait
from tests.fake_llm import default_script


@pytest.fixture
def windows_like(monkeypatch):
    """Path.replace fails while anyone is reading the target, as on Windows."""
    readers: dict[Path, int] = {}
    guard = threading.Lock()
    real_read, real_replace = Path.read_text, Path.replace

    def read_text(self, *args, **kwargs):
        with guard:
            readers[self] = readers.get(self, 0) + 1
        try:
            time.sleep(0.002)  # a read takes a moment, like a real one
            return real_read(self, *args, **kwargs)
        finally:
            with guard:
                readers[self] -= 1

    def replace(self, target):
        with guard:
            if readers.get(Path(target), 0):
                raise PermissionError(13, "Zugriff verweigert (simulated WinError 5)")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "replace", replace)
    return readers


@pytest.mark.parametrize("retries", [runner._WINDOWS_RETRIES, 1], ids=["lock+retry", "lock alone"])
def test_reading_and_writing_meta_at_once_never_fails(
    tmp_path, windows_like, monkeypatch, retries
) -> None:
    """The exact situation of the failure: the page reads, the run writes.
    Run once with retries switched off, so the lock is shown to be enough
    on its own inside one process (and the retry is not hiding a gap)."""
    monkeypatch.setattr(runner, "_WINDOWS_RETRIES", retries)
    runner.write_meta(tmp_path, {"status": "running", "n": 0})
    errors: list[BaseException] = []
    empty_reads = []

    def writer() -> None:
        try:
            for n in range(150):
                runner.update_meta(tmp_path, n=n)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        for _ in range(150):
            if not runner.read_meta(tmp_path):
                empty_reads.append(1)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader),
               threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    assert not empty_reads, "a reader must never see the run as missing"
    assert runner.read_meta(tmp_path)["n"] == 149
    assert not list(tmp_path.glob("*.tmp")), "no temp files left behind"


def test_a_file_held_by_another_program_is_waited_for(tmp_path, monkeypatch) -> None:
    """Across PROCESSES the lock cannot help (an antivirus scan, the CLI):
    a short retry does."""
    real = Path.replace
    failures = iter([True, True, True])

    def flaky(self, target):
        if next(failures, False):
            raise PermissionError(13, "Zugriff verweigert")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    runner.write_meta(tmp_path, {"status": "done"})
    assert runner.read_meta(tmp_path) == {"status": "done"}


def test_a_run_finishes_even_if_meta_json_cannot_be_written(
    monkeypatch, override_settings
) -> None:
    """If meta.json stays locked, the run still ends -- it must never be left
    'running' forever, and a finished analysis must not become a failure."""
    override_settings(require_approval=False)
    _install(default_script(), monkeypatch)
    manager = jobs.JobManager(build_graph(checkpointer=InMemorySaver()))
    payload = runner.start_payload(CSV, QUESTION, "locked")

    def locked(folder, meta):
        raise PermissionError(13, "Zugriff verweigert")

    monkeypatch.setattr(runner, "write_meta", locked)
    job = _wait(manager.start("locked", payload))

    assert job.status == "done", job.error
    assert Path(job.outcome.report_path).exists()


def test_a_failure_before_the_run_starts_is_reported_not_hidden(
    monkeypatch, override_settings
) -> None:
    """Anything that goes wrong in the run thread ends the job as failed."""
    override_settings(require_approval=False)
    manager = jobs.JobManager(build_graph(checkpointer=InMemorySaver()))

    def broken(*args, **kwargs):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(runner, "advance", broken)
    job = _wait(manager.start("early", runner.start_payload(CSV, QUESTION, "early")), 10)
    assert job.status == "failed"
    assert "disk gone" in str(job.error)

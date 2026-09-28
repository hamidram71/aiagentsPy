"""Runs in the background, for the web app.

A web page must not run the analysis itself. Streamlit re-runs the page
script on every click, and a click in the middle of an analysis would abort
the script -- and the analysis with it. Closing the browser tab would too.

So the page only *starts* a run here and then looks at it. The run itself
lives in its own thread, owned by this module, for as long as the app
process lives: a colleague can close the tab and come back to a finished
report. No Streamlit import in this file, so all of it is tested directly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import runner


class Stopped(Exception):
    """Raised inside a run to stop it at the next finished step."""


@dataclass
class Job:
    thread_id: str
    status: str = "running"  # running | approval | done | failed | stopped
    lines: list[str] = field(default_factory=list)
    outcome: runner.Outcome | None = None
    error: BaseException | None = None
    started: float = field(default_factory=time.time)
    stop_requested: threading.Event = field(default_factory=threading.Event)

    @property
    def active(self) -> bool:
        return self.status == "running"


class JobManager:
    """Start runs, continue them after approval, and report on them."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, thread_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(thread_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(job.active for job in self._jobs.values())

    def start(self, thread_id: str, payload: Any) -> Job:
        """Run the graph for `thread_id` from `payload` in a new thread.

        `payload` is a new run's initial state, a `Command(resume=...)` with
        the plan decision, or None to continue from the checkpoint.
        """
        with self._lock:
            current = self._jobs.get(thread_id)
            if current and current.active:
                # Two advances on one thread at once would interleave two
                # continuations of the same checkpoint. Refuse the second.
                raise RuntimeError(f"run {thread_id} is already running")
            # Keep the progress lines across the approval pause: the page
            # shows the whole run, not just the part after the click.
            job = Job(thread_id=thread_id, lines=list(current.lines) if current else [])
            self._jobs[thread_id] = job

        worker = threading.Thread(
            target=self._work, args=(job, payload), name=f"run-{thread_id}", daemon=True
        )
        worker.start()
        return job

    def stop(self, thread_id: str) -> None:
        """Ask a run to stop after the step it is on. It can be continued."""
        job = self.get(thread_id)
        if job and job.active:
            job.stop_requested.set()
            job.lines.append("stopping after the current step ...")

    # --- the thread ------------------------------------------------------------

    def _work(self, job: Job, payload: Any) -> None:
        config = {"configurable": {"thread_id": job.thread_id}}

        def on_update(node: str, update: dict) -> None:
            job.lines.extend(f"[{node}] {line}" for line in runner.describe(node, update))
            if job.stop_requested.is_set():
                raise Stopped()

        folder = runner.run_folder(job.thread_id)
        # EVERYTHING inside the try, the first meta.json write included. It
        # used to sit before the try: on Windows it raised a PermissionError,
        # the thread died, and the job said "running" forever -- the page
        # showed "Working ..." with a clock that never stopped.
        try:
            self._set_meta(folder, status="running", error="")
            outcome = runner.advance(self.graph, payload, config, on_update=on_update)
        except Stopped:
            job.status = "stopped"
            self._set_meta(folder, status="stopped")
            return
        except BaseException as exc:  # noqa: BLE001 -- shown on the page
            job.error = exc
            job.status = "failed"  # first: the page must see it even if the next line fails
            self._set_meta(folder, status="failed", error=f"{type(exc).__name__}: {exc}")
            return
        job.outcome = outcome
        job.status = outcome.status  # "approval" or "done"

    @staticmethod
    def _set_meta(folder: Path, **changes: Any) -> None:
        """Best effort: meta.json is the history list's view of the run. The
        job object is the truth while the app runs, so a failed write here
        must never take the run down with it."""
        try:
            runner.update_meta(folder, **changes)
        except OSError as exc:
            logging.getLogger(__name__).warning("could not update %s/meta.json: %s", folder, exc)

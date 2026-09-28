"""Running the agent for a front end. The CLI and the web app both use this.

One place for the things every front end needs and must get the same way:
where a run's files live, how a run is started and continued past the plan
approval, how its progress is described, and how many tokens it used.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler

from src.config import OUTPUT_DIR, settings
from src.state import initial_state

RUNS_DIR = OUTPUT_DIR / "runs"
USAGE_FILE = OUTPUT_DIR / "usage.csv"

# Several people can run at once (the web app serves each browser tab in its
# own thread). Appending to one CSV from two threads can interleave lines.
_usage_lock = threading.Lock()


def new_thread_id() -> str:
    return f"run-{uuid.uuid4().hex[:8]}"


def run_folder(thread_id: str, runs_dir: Path | None = None) -> Path:
    return (runs_dir or RUNS_DIR) / thread_id


# --- starting a run -------------------------------------------------------------


def start_payload(
    csv_path: str | Path,
    question: str,
    thread_id: str,
    user: str = "",
    runs_dir: Path | None = None,
    copy_input: bool = False,
) -> dict:
    """Create the run's folder and meta.json, and return the graph's input.

    `copy_input` keeps a copy of the CSV inside the run folder -- the web app
    needs it, because an uploaded file has no home of its own, and a run
    that is resumed tomorrow must still find its data.
    """
    folder = run_folder(thread_id, runs_dir)
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(csv_path).resolve()
    if copy_input:
        target = folder / f"input_{source.name}"
        shutil.copyfile(source, target)
        source = target

    write_meta(folder, {
        "thread_id": thread_id,
        "user": user,
        "question": question,
        "csv": Path(csv_path).name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "model": f"{settings.provider} / {settings.resolved_model()}",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    })
    return initial_state(str(source), question, run_dir=str(folder))


# --- advancing a run --------------------------------------------------------------


class TokenCounter(BaseCallbackHandler):
    """Adds up the tokens of every model call made during one advance().

    langchain's own UsageMetadataCallbackHandler counts a call only when the
    reply names its model -- a reply without `model_name` was silently not
    counted at all. Tokens are tokens; this counts every reply that has them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # tools and nodes may run in threads
        self.totals = {"input_tokens": 0, "output_tokens": 0}

    def on_llm_end(self, response, **kwargs: Any) -> None:
        for generations in response.generations or []:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                with self._lock:
                    for key in self.totals:
                        self.totals[key] += int(usage.get(key, 0) or 0)


@dataclass
class Outcome:
    """Where a run stopped: waiting for plan approval, or finished."""

    status: str  # "approval" | "done"
    values: dict
    approval: dict | None = None
    usage: dict = field(default_factory=dict)

    @property
    def report_path(self) -> str | None:
        return self.values.get("report_path")


def advance(
    graph,
    payload: Any,
    config: dict,
    on_update: Callable[[str, dict], None] | None = None,
) -> Outcome:
    """Run the graph until it pauses for approval or finishes.

    `payload` is the initial state for a new run, a `Command(resume=...)`
    after an approval, or None to resume from the checkpoint.
    """
    usage = TokenCounter()
    run_config = {**config, "callbacks": [*(config.get("callbacks") or []), usage]}

    request = None
    try:
        for update in graph.stream(payload, config=run_config, stream_mode="updates"):
            for node, data in (update or {}).items():
                if node == "__interrupt__":
                    request = data[0].value
                    continue
                if on_update:
                    on_update(node, data or {})
    except BaseException:
        # A run that fails or is stopped half-way has still spent tokens,
        # and the one API key has still paid for them. Count them.
        folder = graph.get_state(config).values.get("run_dir")
        if folder:
            _record_safely(Path(folder), {}, dict(usage.totals), "stopped early")
        raise

    values = graph.get_state(config).values
    tokens = dict(usage.totals)
    folder = values.get("run_dir")
    status = "approval" if request else "done"
    if folder:
        _record_safely(Path(folder), values, tokens, status)
    return Outcome(status=status, values=values, approval=request, usage=tokens)


def _record_safely(folder: Path, values: dict, tokens: dict, status: str) -> None:
    """_record, but a file that cannot be written only costs a log line.

    The run's result is in the checkpoint and its report is on disk; a
    locked meta.json or usage.csv (Windows, an open Excel window on
    usage.csv) must not turn a finished analysis into a failed one.
    """
    try:
        _record(folder, values, tokens, status)
    except OSError as exc:
        logging.getLogger(__name__).warning("could not record the run in %s: %s", folder, exc)


USAGE_COLUMNS = ["time", "user", "thread", "model", "part", "input_tokens", "output_tokens"]


def _record(folder: Path, values: dict, tokens: dict, status: str) -> None:
    """Update meta.json, and add this advance's tokens to outputs/usage.csv.

    One API key pays for everyone's runs, so the usage file says who used
    how much -- tokens, not money: prices change and are not ours to guess.
    One line per advance (up to the approval, after it, a stopped attempt),
    so a run that never finishes is counted too. Add up by user for totals.
    """
    with _meta_lock:
        meta = read_meta(folder)
        used = meta.get("usage") or {"input_tokens": 0, "output_tokens": 0}
        for key in ("input_tokens", "output_tokens"):
            used[key] = int(used.get(key, 0)) + tokens[key]
        meta["usage"] = used
        if status in {"approval", "done"}:
            meta.update({
                "status": "waiting for approval" if status == "approval" else "done",
                "report_path": values.get("report_path") or "",
                "findings": len(values.get("findings") or []),
                "charts": [Path(c).name for c in values.get("charts") or []],
            })
        write_meta(folder, meta)

    part = {"approval": "up to the plan approval", "done": "finished"}.get(status, status)
    # Next to the runs folder: outputs/runs/<thread>/ -> outputs/usage.csv.
    usage_file = folder.parent.parent / USAGE_FILE.name
    with _usage_lock:
        new_file = not usage_file.exists()
        with usage_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(USAGE_COLUMNS)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"), meta.get("user", ""),
                meta.get("thread_id", ""), meta.get("model", ""), part,
                tokens["input_tokens"], tokens["output_tokens"],
            ])


def usage_by_user(usage_file: Path | None = None) -> dict[str, dict[str, int]]:
    """Total tokens per user, from outputs/usage.csv."""
    path = usage_file or USAGE_FILE
    totals: dict[str, dict[str, int]] = {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                user = totals.setdefault(row.get("user") or "(no name)", {"input_tokens": 0, "output_tokens": 0})
                for key in user:
                    user[key] += int(row.get(key) or 0)
    except OSError:
        return {}
    return totals


# --- meta.json and history --------------------------------------------------------


# One lock for every meta.json read and write in this process. On Windows a
# file that another thread has open cannot be replaced: the web page reading
# meta.json while a run thread saved it gave "PermissionError: [WinError 5]
# Zugriff verweigert" -- and killed the run thread. Linux allows both at
# once, which is why no test here saw it. RLock: update_meta holds it while
# calling read_meta and write_meta.
_meta_lock = threading.RLock()
_WINDOWS_RETRIES = 40  # x 50 ms: another PROCESS (an antivirus scan, the CLI) may hold the file


def write_meta(folder: Path, meta: dict) -> None:
    with _meta_lock:
        # A temp name of its own per write, so two writers can never rename
        # each other's file away.
        tmp = folder / f"meta.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(_WINDOWS_RETRIES):
            try:
                tmp.replace(folder / "meta.json")  # atomic: never half a file
                return
            except PermissionError:
                if attempt == _WINDOWS_RETRIES - 1:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(0.05)


def read_meta(folder: Path) -> dict:
    with _meta_lock:
        for _ in range(_WINDOWS_RETRIES):
            try:
                return json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            except PermissionError:  # being replaced by another process right now
                time.sleep(0.05)
            except (OSError, ValueError):
                return {}
        return {}


def update_meta(folder: Path, **changes: Any) -> None:
    """Change some fields of meta.json, with no other write in between."""
    with _meta_lock:
        meta = read_meta(folder)
        if meta:
            meta.update(changes)
            write_meta(folder, meta)


def list_runs(limit: int = 20, user: str | None = None, runs_dir: Path | None = None) -> list[dict]:
    """The most recent runs, newest first -- optionally only one user's."""
    root = runs_dir or RUNS_DIR
    if not root.exists():
        return []
    metas = []
    for folder in root.iterdir():
        meta = read_meta(folder) if folder.is_dir() else {}
        if meta and (user is None or meta.get("user") == user):
            metas.append({**meta, "folder": str(folder)})
    metas.sort(key=lambda m: m.get("created", ""), reverse=True)
    return metas[:limit]


# --- describing progress ----------------------------------------------------------


def describe(node: str, update: dict) -> list[str]:
    """One readable line (or a few) per node update. No printing: the caller
    decides where the lines go -- the terminal, or the web page."""
    lines: list[str] = []
    if node == "loader" and "profile" in update:
        profile = update["profile"]
        lines.append(f"loaded {profile['rows']} rows x {profile['columns_count']} columns")
    elif node == "planner" and update.get("plan"):
        lines.append("plan:")
        lines += [f"  {i}. {step}" for i, step in enumerate(update["plan"], 1)]
    elif node == "supervisor":
        messages = update.get("messages") or []
        if messages:
            lines.append(str(messages[-1].content))
    elif node in {"analyst", "charter"}:
        findings = update.get("findings") or []
        charts = update.get("charts") or []
        messages = update.get("messages") or []
        lines += [f"finding: {f['text']}" for f in findings]
        lines += [f"chart:   {c}" for c in charts]
        called = False
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                called = True
                lines.append(f"tool -> {call['name']}({str(call['args'])[:120]})")
        # A worker turn that calls no tool and produces no finding used to
        # print NOTHING, and the transcript looked like the supervisor talking
        # to itself. Never let a node run silently: show what it said.
        if not findings and not charts and not called:
            for message in messages:
                text = str(getattr(message, "content", "") or "").strip()
                if text:
                    lines.append(f"no result. said: {' '.join(text.split())[:160]!r}")
    elif node == "reporter" and update.get("report_path"):
        lines.append(f"report -> {update['report_path']}")

    lines += [f"! {error}" for error in update.get("errors") or []]
    return lines

"""CSV Insight Agent -- the web app.

Start it with `python run_app.py` (or double-click run_app.bat on Windows),
not with `streamlit run` directly: the launcher decides whether colleagues
may connect, and that depends on APP_PASSWORD being set.

The page never runs an analysis itself. It starts one in src/jobs.py and
then only looks at it, so a click, a reload or a closed tab cannot abort a
run half-way. Everything shown here comes from src/runner.py, the same code
the command line uses.
"""

from __future__ import annotations

import hmac
import io
import os
import re
import shutil
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CSV Insight Agent", page_icon=":bar_chart:", layout="wide")

# Same as main.py: langchain-openai's structured output makes pydantic warn
# on every planner/supervisor call. Harmless, and it floods the console.
warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings", category=UserWarning)

try:
    from src.config import DATA_DIR, settings
except ValueError as exc:  # src.config.ConfigError
    st.error(f"There is an error in the settings (.env):\n\n{exc}")
    st.stop()

import pandas as pd  # noqa: E402
from langgraph.types import Command  # noqa: E402

import main  # noqa: E402  -- credential_problem, explain_fatal, VERSION
from src import runner  # noqa: E402
from src.dataset import read_csv_file  # noqa: E402
from src.errors import FatalModelError  # noqa: E402
from src.graph import build_graph, make_checkpointer  # noqa: E402
from src.jobs import JobManager  # noqa: E402

SAMPLE = DATA_DIR / "sales.csv"
MAX_UPLOAD_MB = 50
CHART_WIDTH = 760  # pixels; full page width makes a bar chart a wall
# A run id comes from the address bar (?run=...) and becomes a folder name:
# anything but letters, digits, - and _ could point outside outputs/runs/.
THREAD_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
IMAGE = re.compile(r"!\[([^\]]*)\]\(<?([^)\s>]+)>?\)")
STATUS_LABEL = {
    "running": ":material/progress_activity: running",
    "waiting for approval": ":material/pending_actions: waiting for approval",
    "done": ":material/check_circle: done",
    "failed": ":material/error: failed",
    "stopped": ":material/pause_circle: stopped",
}


@st.cache_resource
def jobs() -> JobManager:
    """One graph, one checkpoint database and one job list for the whole app.

    cache_resource makes this a single object shared by every browser tab:
    exactly what lets a colleague close the tab and find the run later.
    """
    checkpointer, _connection = make_checkpointer()
    return JobManager(build_graph(checkpointer=checkpointer))


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# --- sign in ------------------------------------------------------------------


def signed_in_user() -> str:
    """The user's name; with APP_PASSWORD set, only after the right password."""
    if st.session_state.get("user"):
        return st.session_state["user"]

    password = os.environ.get("APP_PASSWORD", "")
    st.title("CSV Insight Agent")
    with st.form("sign-in"):
        name = st.text_input("Your name", max_chars=40, help="Your runs are listed under this name.")
        given = st.text_input("Password", type="password") if password else ""
        submitted = st.form_submit_button("Sign in", type="primary")
    if not password:
        st.caption("No APP_PASSWORD is set, so this app only accepts connections from this computer.")

    if submitted:
        name = " ".join(name.split())
        if not name:
            st.error("Please enter your name.")
        elif password and not hmac.compare_digest(given.encode(), password.encode()):
            time.sleep(1.5)  # slows down guessing
            st.error("Wrong password.")
        else:
            st.session_state["user"] = name
            st.rerun()
    st.stop()


# --- sidebar ------------------------------------------------------------------


def open_run(thread_id: str | None) -> None:
    if thread_id and not THREAD_ID.fullmatch(thread_id):
        thread_id = None
    st.session_state["thread"] = thread_id
    if thread_id:
        st.query_params["run"] = thread_id
    else:
        st.query_params.clear()


def sidebar(user: str) -> None:
    with st.sidebar:
        st.markdown(f"**{user}**")
        st.caption(f"Model: {settings.provider} / {settings.resolved_model()}")
        if st.button("New analysis", icon=":material/add:", type="primary", width="stretch"):
            open_run(None)
            st.rerun()

        everyone = st.toggle("Show everyone's runs", value=False)
        st.subheader("Runs")
        runs = runner.list_runs(limit=40, user=None if everyone else user)
        if not runs:
            st.caption("No runs yet.")
        for meta in runs:
            status = meta.get("status", "")
            job = jobs().get(meta["thread_id"])
            if job and job.active:
                status = "running"
            label = f"{(meta.get('question') or '?')[:48]}"
            detail = f"{STATUS_LABEL.get(status, status)} · {meta.get('created', '')[:16].replace('T', ' ')}"
            if everyone:
                detail += f" · {meta.get('user') or '?'}"
            current = st.session_state.get("thread") == meta["thread_id"]
            if st.button(label, key=f"open-{meta['thread_id']}", width="stretch",
                         type="secondary" if not current else "primary"):
                open_run(meta["thread_id"])
                st.rerun()
            st.caption(detail)

        totals = runner.usage_by_user().get(user)
        if totals:
            st.divider()
            st.caption(
                f"Your tokens so far: {totals['input_tokens']:,} in · {totals['output_tokens']:,} out"
            )
        if st.button("Sign out", icon=":material/logout:", type="tertiary"):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()
        st.caption(f"version {main.VERSION}")


# --- starting a run -----------------------------------------------------------


def safe_filename(name: str) -> str:
    stem = re.sub(r"[^\w.-]+", "_", Path(name).name).strip("._") or "data.csv"
    return stem if stem.lower().endswith(".csv") else f"{stem}.csv"


def new_run_page(user: str) -> None:
    st.title("New analysis")
    source = st.radio("Data", ["Upload a CSV file", "Sample data (sales.csv)"], horizontal=True)
    upload = None
    if source.startswith("Upload"):
        upload = st.file_uploader("CSV file", type=["csv"], max_upload_size=MAX_UPLOAD_MB)
        if upload is not None:
            preview = read_preview(upload.getvalue())
            if isinstance(preview, str):
                st.error(preview)
                upload = None
            else:
                st.caption(f"First rows of {upload.name}:")
                st.dataframe(preview, hide_index=True)

    question = st.text_area(
        "Your question",
        placeholder="Which region drives revenue, and is it growing over time?",
        max_chars=1000,
    )
    ready = bool(question.strip()) and (upload is not None or not source.startswith("Upload"))
    if st.button("Start analysis", type="primary", icon=":material/play_arrow:", disabled=not ready):
        thread_id = runner.new_thread_id()
        if upload is not None:
            with tempfile.TemporaryDirectory() as scratch:
                staged = Path(scratch) / safe_filename(upload.name)
                staged.write_bytes(upload.getvalue())
                payload = runner.start_payload(
                    staged, question.strip(), thread_id, user=user, copy_input=True
                )
        else:
            payload = runner.start_payload(SAMPLE, question.strip(), thread_id, user=user)
        if settings.provider == "fake":
            # The fake model is one scripted queue; a new run needs a fresh one.
            from src.llm import get_llm

            get_llm.cache_clear()
        jobs().start(thread_id, payload)
        open_run(thread_id)
        st.rerun()


def read_preview(data: bytes) -> pd.DataFrame | str:
    """The first rows, or a message saying why the file cannot be read.

    Read exactly as the agent will read it (src/dataset.py), so the preview
    never shows columns the analysis would not see -- German Excel files
    with semicolons and decimal commas included.
    """
    try:
        frame = read_csv_file(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 -- any parse error is a user message
        return f"This file cannot be read as a CSV: {exc}"
    if frame.shape[1] < 2:
        return (
            "This file has only one column, so there is nothing to compare. "
            "Is it really a table? Columns must be separated by commas or semicolons."
        )
    if frame.empty:
        return "This file has column names but no rows."
    return frame.head(5)


# --- one run ------------------------------------------------------------------


def run_page(thread_id: str, user: str) -> None:
    folder = runner.run_folder(thread_id)
    meta = runner.read_meta(folder)
    if not meta:
        st.warning("This run no longer exists.")
        if st.button("Back"):
            open_run(None)
            st.rerun()
        return

    st.title(meta.get("question") or "Analysis")
    st.caption(
        f"{meta.get('user') or '?'} · {meta.get('created', '').replace('T', ' ')} · "
        f"data: {meta.get('csv', '?')} · {meta.get('model', '')}"
    )

    job = jobs().get(thread_id)
    if job and job.active:
        progress(thread_id)
        return
    owner = meta.get("user") == user

    snapshot = jobs().graph.get_state(config_for(thread_id))
    waiting = [i.value for task in snapshot.tasks for i in task.interrupts]
    if waiting:
        approval(thread_id, waiting[0], job)
    elif snapshot.values.get("report_path") and not snapshot.next:
        report(thread_id, folder, meta, job, owner)
    else:
        unfinished(thread_id, meta, job, bool(snapshot.next), owner)


@st.fragment(run_every=1.5)
def progress(thread_id: str) -> None:
    """Refreshes itself every 1.5 s while the run works; the rest of the page
    stays still. When the run stops or pauses, the whole page is redrawn."""
    job = jobs().get(thread_id)
    if not job or not job.active:
        st.rerun(scope="app")

    elapsed = int(time.time() - job.started)
    with st.status(f"Working … {elapsed // 60}:{elapsed % 60:02d}", expanded=True):
        st.code("\n".join(job.lines[-40:]) or "starting …", language=None, wrap_lines=True)
    st.caption("You can close this page; the run continues and appears under Runs.")
    if st.button("Stop after the current step", icon=":material/stop:",
                 disabled=job.stop_requested.is_set()):
        jobs().stop(thread_id)


def approval(thread_id: str, request: dict, job) -> None:
    st.subheader("Plan: approve or change it")
    st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(request.get("plan") or [], 1)))
    if st.button("Approve and run", type="primary", icon=":material/check:"):
        resume(thread_id, Command(resume="approve"))
    feedback = st.text_area(
        "Or say what to change",
        placeholder="e.g. also compare the product categories; skip the monthly trend",
    )
    if st.button("Revise the plan", icon=":material/edit:", disabled=not feedback.strip()):
        resume(thread_id, Command(resume=feedback.strip()))
    if job:
        log(job)


def resume(thread_id: str, payload) -> None:
    try:
        jobs().start(thread_id, payload)
    except RuntimeError as exc:
        st.warning(str(exc))
        return
    st.rerun()


def report(thread_id: str, folder: Path, meta: dict, job, owner: bool) -> None:
    path = Path(meta.get("report_path") or "")
    if not path.is_file():
        st.warning("The report file is missing from the run folder.")
        return
    markdown = path.read_text(encoding="utf-8")
    usage = meta.get("usage") or {}

    top = st.columns([1, 1, 3])
    top[0].download_button("Report (.md)", markdown, file_name=path.name,
                           mime="text/markdown", icon=":material/description:")
    top[1].download_button("Everything (.zip)", zip_folder(folder), file_name=f"{thread_id}.zip",
                           mime="application/zip", icon=":material/folder_zip:")
    top[2].caption(
        f"{meta.get('findings', 0)} findings · {len(meta.get('charts') or [])} charts · "
        f"tokens: {usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out"
    )

    st.divider()
    shown = render_report(markdown, folder)
    others = [name for name in meta.get("charts") or [] if name not in shown]
    if others:
        st.subheader("Charts")
        for name in others:
            if (folder / name).is_file():
                st.image(str(folder / name), caption=name, width=CHART_WIDTH)

    st.divider()
    if job:
        log(job)
    if owner:
        delete_button(thread_id, folder)


def render_report(markdown: str, folder: Path) -> set[str]:
    """The report, with its chart references shown as the actual pictures.

    The report says `![title](revenue_by_region.png)`: a path next to the
    report file. A browser cannot open files on this computer, so each image
    reference is replaced by the chart itself -- only ever a file from this
    run's own folder, whatever path the text names.
    """
    shown: set[str] = set()
    position = 0
    for match in IMAGE.finditer(markdown):
        st.markdown(markdown[position:match.start()])
        name = Path(match.group(2)).name
        picture = folder / name
        if picture.is_file():
            st.image(str(picture), caption=match.group(1) or None, width=CHART_WIDTH)
            shown.add(name)
        else:
            st.caption(f"(chart not found: {name})")
        position = match.end()
    st.markdown(markdown[position:])
    return shown


def zip_folder(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(folder.iterdir()):
            if item.is_file() and not item.name.endswith(".tmp"):
                archive.write(item, item.name)
    return buffer.getvalue()


def unfinished(thread_id: str, meta: dict, job, can_continue: bool, owner: bool) -> None:
    status = meta.get("status", "")
    error = job.error if job else None
    if isinstance(error, FatalModelError):
        st.error(main.explain_fatal(error, thread_id, resume_hint="Fix it, then press Continue."))
    elif status == "failed":
        st.error(f"The run failed: {meta.get('error') or error or 'unknown error'}")
    elif status == "stopped":
        st.info("This run was stopped. It can continue from where it stopped.")
    else:
        # meta says "running" but nothing runs it: the app was restarted
        # (or the computer went to sleep) in the middle of the run.
        st.info("This run was interrupted when the app was closed. It can continue from its last finished step.")

    if can_continue and st.button("Continue", type="primary", icon=":material/play_arrow:"):
        resume(thread_id, None)
    if job:
        log(job)
    if owner:
        delete_button(thread_id, runner.run_folder(thread_id))


def log(job) -> None:
    if job.lines:
        with st.expander("What the agent did"):
            st.code("\n".join(job.lines), language=None, wrap_lines=True)


def delete_button(thread_id: str, folder: Path) -> None:
    with st.expander("Delete this run"):
        st.caption("Deletes the report, the charts and the data copy of this run.")
        if st.button("Delete", key=f"delete-{thread_id}", icon=":material/delete:"):
            shutil.rmtree(folder, ignore_errors=True)
            open_run(None)
            st.rerun()


# --- the page -----------------------------------------------------------------


def page() -> None:
    user = signed_in_user()
    problem = main.credential_problem(settings)
    if problem:
        st.error(problem)
        st.stop()

    if "thread" not in st.session_state:
        open_run(st.query_params.get("run"))
    sidebar(user)

    thread_id = st.session_state.get("thread")
    if thread_id:
        run_page(thread_id, user)
    else:
        new_run_page(user)


# Streamlit runs this file as __main__; a test that imports it for one helper
# does not get a whole page drawn outside a browser.
if __name__ == "__main__":
    page()

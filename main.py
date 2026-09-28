#!/usr/bin/env python3
"""CLI entry point.

    python main.py data/sales.csv "Which region drives revenue, and is the trend up?"
    python main.py data/sales.csv "..." --no-approval
    python main.py --resume --thread-id run-42

Streaming: `graph.stream(..., stream_mode="updates")` yields one dict per node
as soon as that node finishes -- `{"analyst": {...partial state...}}`. That is
what lets you show progress instead of a spinner. Use
`stream_mode="messages"` instead if you want token-by-token output.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import threading
import warnings
from pathlib import Path

# Import config *after* any --provider override reaches the environment, since
# Settings reads os.environ once at import time.


VERSION = "0.14.1"


def make_output_safe() -> None:
    """Never let printing a character crash the run.

    Everything this CLI prints about the run is text a model wrote, and
    models write typographic characters: the minus sign in
    "(1 − discount_pct/100)", arrows, curly quotes. On Windows, Python
    prints Unicode fine to the terminal itself -- but the moment output is
    piped or redirected (`| Select-String`, `> run.log`), it encodes with the
    legacy code page, cp1252 has no U+2212, and `print` raises
    UnicodeEncodeError. That happens here in main.py, outside every node's
    safe_node, so it did not become an error in the state: it killed the
    run, half-way, on a real user's machine.

    Keep the stream's encoding (the receiving program expects it) and only
    change what happens to a character it cannot hold: it becomes "?"
    instead of an exception. A wrong glyph in a log is cosmetic; a dead run
    is not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # a stream that cannot be changed
                pass


class ApprovalNeeded(Exception):
    """The plan waits for a human, and this terminal cannot ask one."""


def run_interruptibly(work):
    """Run `work` in a thread and keep the main thread free for Ctrl+C.

    LangGraph runs each node in a worker thread and the main thread waits for
    it. Ctrl+C during a slow model call was therefore handled only when that
    call returned: 17.7 seconds on a 20-second request, and a hung request
    held it for the full timeout. On Windows a blocked wait may not be
    interruptible at all. Joining in short slices means the main thread is
    back in Python code five times a second, where KeyboardInterrupt can land.
    """
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = work()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True, name="graph-run")
    worker.start()
    while worker.is_alive():
        worker.join(0.2)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV analysis agent (LangGraph)")
    parser.add_argument(
        "--version", action="version", version=f"csv-insight-agent {VERSION}"
    )
    parser.add_argument("csv", nargs="?", help="path to the CSV file")
    parser.add_argument("question", nargs="?", help="what you want to know")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="conversation id; reuse it to resume an interrupted run",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "fake"],
        help="override LLM_PROVIDER for this run",
    )
    parser.add_argument("--model", help="override LLM_MODEL for this run")
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="skip the human plan-approval interrupt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a thread that stopped at an interrupt",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        if not args.model:
            # --provider alone means "that provider's default model". Without
            # this, LLM_MODEL from .env still applied: `--provider anthropic`
            # with LLM_MODEL=gpt-5.6-terra sent an OpenAI model name to
            # Anthropic. An empty value also stops .env from setting it later.
            os.environ["LLM_MODEL"] = ""
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.no_approval:
        os.environ["REQUIRE_APPROVAL"] = "false"


# Which environment variable each hosted provider needs.
REQUIRED_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def credential_problem(settings) -> str | None:
    """Fail before the graph runs if the provider cannot possibly work.

    Without this, a missing key shows up as "OpenAIError: Missing
    credentials" *after* the loader has already run, which points at the key
    when the real mistake is often elsewhere -- `--model <something>` with
    LLM_PROVIDER left on a different provider, say: the model name was
    overridden and the provider was not. Name both in the message, and say
    what to do.
    """
    import os

    key = REQUIRED_KEYS.get(settings.provider)
    if not key or os.getenv(key):
        return None

    model = settings.resolved_model(settings.provider)
    other = "anthropic" if settings.provider == "openai" else "openai"
    lines = [
        f"error: LLM_PROVIDER={settings.provider} needs {key}, which is not set.",
        f"       (this run would use: {settings.provider} / {model})",
        "",
        "Set the key in .env, next to main.py:",
        "",
        f"    LLM_PROVIDER={settings.provider}",
        f"    {key}=...",
        "",
        "Or switch provider -- note that `--model` alone only changes the",
        "model name, never the provider:",
        "",
        f"    python main.py <csv> <question> --provider {other}",
        "",
        "Make sure LLM_PROVIDER appears only once in .env, and run",
        "`python check_llm.py` to confirm it works before a full run.",
    ]
    return "\n".join(lines)


def _api_message(cause: Exception) -> tuple[str, str | None]:
    """The provider's own sentence, and the parameter it names (if any).

    `str(exc)` on an SDK error is "Error code: 400 - {'error': {...}}": the
    useful sentence is in there, wrapped in a dict repr. Dig it out.
    """
    from src.llm import rejected_parameter

    body = getattr(cause, "body", None)
    message = None
    if isinstance(body, dict):
        error = body.get("error", body)
        message = error.get("message") if isinstance(error, dict) else error
    elif isinstance(body, str):
        message = body
    # Whatever shape it came in -- a list or a dict crashed the old
    # `message.split()` -- the user gets one readable line.
    if message is not None and not isinstance(message, str):
        import json

        message = json.dumps(message, ensure_ascii=False, default=str)
    return (message or str(cause)), rejected_parameter(cause)


def explain_fatal(exc, thread_id: str, resume_hint: str | None = None) -> str:
    """Turn a FatalModelError into a diagnosis and a way back in.

    `resume_hint` replaces the command-line resume instructions -- the web
    app has a button for that instead.
    """
    from langchain_core.exceptions import (
        ModelAuthenticationError,
        ModelNotFoundError,
        ModelPermissionDeniedError,
    )

    cause = exc.cause
    message, param = _api_message(cause)

    if isinstance(cause, (ModelAuthenticationError, ModelPermissionDeniedError)):
        fix = "the API key: check OPENAI_API_KEY / ANTHROPIC_API_KEY in .env."
    elif isinstance(cause, ModelNotFoundError):
        fix = "the model name: check LLM_MODEL in .env (it may be misspelled or retired)."
    elif param and param.startswith(("messages", "input", "tool")):
        # The provider is objecting to the CONVERSATION this program sent --
        # its own transcript, not a setting. Resuming replays the same
        # transcript from the checkpoint, so it fails the same way.
        fix = (
            f"the conversation sent to the model was malformed (`{param}`). That "
            "is a bug in this program, not in your settings -- please report the "
            "message above. Start a new run rather than resuming this one."
        )
    elif param:
        fix = (
            f"the model refuses the request parameter `{param}`. "
            "Try another LLM_MODEL, and send the message above if it persists."
        )
    else:
        fix = "a request setting the model does not accept -- see the message above."

    lines = [
        f"stopped at [{exc.node}]: the model API refused the request.",
        "Retrying would send the identical request again, so the run stops",
        "here instead of spending its failure budget on it.",
        "",
        f"  {type(cause).__name__}: {' '.join(message.split())}",
    ]
    if param:
        lines.append(f"  parameter: {param}")
    lines += ["", f"Most likely: {fix}", ""]
    from src.config import settings

    if settings.checkpoint_db == ":memory:":
        # No checkpoint file: there is nothing to resume, so do not say there is.
        lines.append("This run kept no checkpoint (CHECKPOINT_DB=:memory:); start it again after fixing.")
    elif resume_hint:
        lines += [f"Nothing is lost: the run is saved just before [{exc.node}].", resume_hint]
    else:
        lines += [
            f"Nothing is lost: the run is saved just before [{exc.node}].",
            "After fixing it, continue with:",
            "",
            f"    python main.py --resume --thread-id {thread_id}",
        ]
    return "\n".join(lines)


def describe(node: str, update: dict) -> None:
    """Print one readable line per node update.

    The wording lives in src/runner.py, so the terminal and the web app
    describe a run in exactly the same words.
    """
    from src.runner import describe as lines_for

    for line in lines_for(node, update):
        # Plan steps and similar sub-lines are indented under their node.
        print(f"  [{node}] {line}" if not line.startswith("  ") else f"      {line}")


def run(graph, payload, config) -> dict | None:
    """Stream a run, pausing whenever the graph interrupts for a human.

    Two stream modes at once, and the second one is not decoration:

      "updates"  -> one dict per node that FINISHED. This is the useful
                    summary, but it means total silence while a node runs.
      "messages" -> one chunk per token as the model generates.

    When a node takes a second or two, silence is fine. When one analyst
    turn takes minutes -- a slow model, a long transcript, a rate limit --
    a terminal that prints nothing for two minutes looks broken. Everyone who
    hit that asked the same question: "why is it stuck?" It was not stuck;
    the progress simply had nowhere to appear.

    Combining the modes yields `(mode, payload)` tuples, so the token stream
    becomes a liveness indicator and "updates" still carries the real report.
    """
    from langgraph.types import Command

    # A \r-based progress line only makes sense on an interactive terminal.
    # Piped or redirected (`> run.log`, `| tail`), carriage returns are not
    # processed and the "erase" turns into visible junk -- so when stdout is
    # not a TTY we skip the indicator entirely and keep the clean transcript.
    live = sys.stdout.isatty()
    # Width of the terminal, so the line never wraps: `\r` returns to the
    # start of the CURRENT screen row only, and a wrapped line left
    # "[reporter] working......" behind in the transcript.
    width = max(20, shutil.get_terminal_size((80, 20)).columns - 1)
    blank = "\r" + " " * width + "\r"

    while True:
        interrupted = None
        active_node: str | None = None
        chunk_count = 0

        for mode, data in graph.stream(
            payload, config=config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                if not live:
                    continue
                _, meta = data
                node = meta.get("langgraph_node") or "?"
                # ToolNode also emits message chunks (the ToolMessage it
                # appends). It does no thinking, so saying so would be a lie.
                if node.endswith("_tools"):
                    continue
                if node != active_node:
                    active_node, chunk_count = node, 0
                    shown = f"  [{node}] working"[:width]
                    print(shown, end="", flush=True)
                    room = width - len(shown)
                chunk_count += 1
                if chunk_count % 8 == 0 and room > 0:
                    print(".", end="", flush=True)
                    room -= 1
                continue

            # mode == "updates": a node finished. Erase the progress line so
            # the transcript reads as one clean line per node, not two.
            if active_node:
                print(blank, end="", flush=True)
                active_node = None

            for node, update in (data or {}).items():
                if node == "__interrupt__":
                    interrupted = update
                    continue
                describe(node, update or {})

        if active_node:
            print(blank, end="", flush=True)

        if not interrupted:
            return graph.get_state(config).values

        request = interrupted[0].value
        print("\n" + "=" * 62)
        print("PLAN APPROVAL NEEDED")
        print(f"question: {request['question']}")
        for i, step in enumerate(request["plan"], 1):
            print(f"  {i}. {step}")
        print("=" * 62)
        # Asking needs a person who can see the question and type an answer.
        # Piped (`| Select-String`), the prompt went into the pipe and the
        # run sat in input() forever, invisibly; with no stdin it died with
        # an EOFError traceback. The plan is checkpointed either way.
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise ApprovalNeeded()
        try:
            answer = input("approve, or describe what to change > ").strip()
        except EOFError:
            raise ApprovalNeeded() from None

        payload = Command(resume=answer or "approve")


def main() -> int:
    make_output_safe()
    args = parse_args()
    apply_overrides(args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.verbose:
        # langchain-openai's structured output makes pydantic warn on every
        # planner and supervisor call ("Expected `none` ... field_name=
        # 'parsed'"). It is the library serialising its own response, the
        # parsed result is correct, and the warning lands in the middle of
        # the progress line. Silence exactly that message -- nothing else --
        # and let --verbose bring it back.
        warnings.filterwarnings(
            "ignore", message=r"Pydantic serializer warnings", category=UserWarning
        )

    try:
        from src.config import settings
    except ValueError as exc:  # src.config.ConfigError
        print(f"error in the settings (.env or environment): {exc}", file=sys.stderr)
        return 2
    from src.errors import FatalModelError
    from src.graph import build_graph, make_checkpointer
    from src.runner import new_thread_id, start_payload

    problem = credential_problem(settings)
    if problem:
        print(problem, file=sys.stderr)
        return 2

    checkpointer, conn = make_checkpointer()
    graph = build_graph(checkpointer=checkpointer)

    thread_id = args.thread_id or new_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        if args.resume:
            payload = None  # resuming: state comes from the checkpoint
            snapshot = graph.get_state(config)
            if not snapshot.values:
                print(f"no saved state for thread {thread_id!r}", file=sys.stderr)
                return 1
            print(f"resuming thread {thread_id}")
        else:
            if not args.csv or not args.question:
                print("csv path and question are required", file=sys.stderr)
                return 2
            csv_path = Path(args.csv)
            if not csv_path.exists():
                print(f"no such file: {csv_path}", file=sys.stderr)
                return 2
            existing = graph.get_state(config).values
            if existing:
                # A new question on an existing thread used to come back with
                # the OLD report: the checkpoint still held its report_path,
                # and the supervisor finished at once.
                print(
                    f"thread {thread_id!r} already holds a run (question: "
                    f"{existing.get('question', '?')!r}).\n"
                    f"  continue it:   python main.py --resume --thread-id {thread_id}\n"
                    "  or start new:  leave out --thread-id, or pick a new one",
                    file=sys.stderr,
                )
                return 2
            # Absolute, so --resume from another folder still finds the file:
            # the tools reload it from this path in the checkpoint.
            # Each run gets its own folder, outputs/runs/<thread>/, for its
            # charts and report -- the web app runs several at once.
            payload = start_payload(csv_path, args.question, thread_id)
            print(f"thread: {thread_id}")
            print(f"provider: {settings.provider} / {settings.resolved_model()}")
            print(f"question: {args.question}\n")

        final = run_interruptibly(lambda: run(graph, payload, config)) or {}

        print("\n" + "-" * 62)
        findings = final.get("findings") or []
        print(f"findings: {len(findings)}   charts: {len(final.get('charts') or [])}")
        if final.get("report_path"):
            print(f"report:   {final['report_path']}")
        else:
            print("report:   not written")
            for error in (final.get("errors") or [])[-3:]:
                print(f"  ! {error}")
        print(f"resume this thread with: python main.py --resume --thread-id {thread_id}")
        # A run that wrote no report did not succeed, and a script calling
        # this must be able to tell: it used to exit 0 either way.
        return 0 if final.get("report_path") else 1

    except ApprovalNeeded:
        print(
            "\nThe plan needs your approval, but this terminal cannot ask: the output is\n"
            "piped (| Select-String, > file) or there is no keyboard input. The run is saved.\n"
            f"  approve it:  python main.py --resume --thread-id {thread_id}   (no pipe)\n"
            "  or skip approval next time: add --no-approval",
            file=sys.stderr,
        )
        return 4

    except KeyboardInterrupt:
        print(f"\ninterrupted. resume with --resume --thread-id {thread_id}", flush=True)
        sys.stderr.flush()
        # Leave NOW. The model call in flight is still running in a worker
        # thread, and a normal exit waits for it -- up to the request
        # timeout. Everything up to the last finished step is already in the
        # checkpoint, so --resume loses nothing but the unfinished call.
        os._exit(130)
    except FatalModelError as exc:
        print("\n" + explain_fatal(exc, thread_id), file=sys.stderr)
        return 3
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

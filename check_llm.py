#!/usr/bin/env python3
"""Check whether the configured model can do what this graph needs.

    python check_llm.py
    python check_llm.py --provider anthropic

The graph asks three different things of a model, and not every model does
all three -- small or older models often chat fine and then fail at tool
calling:

  1. plain chat            -- the reporter node
  2. structured output      -- the planner and supervisor nodes
  3. tool calling           -- the analyst and charter nodes

Run this before your first real run. Four cheap calls tell you in seconds
what would otherwise take a confusing half-finished graph run to discover.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, TypedDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provider capability check")
    parser.add_argument(
        "--provider", choices=["openai", "anthropic", "fake"]
    )
    parser.add_argument("--model")
    return parser.parse_args()


class _CheckState(TypedDict, total=False):
    out: Any


def through_graph(call):
    """Run `call` the way the agent runs every model call: inside a graph node,
    streamed with stream_mode=["updates", "messages"].

    That mode makes every chat model STREAM its request. The first version of
    this script used plain .invoke() -- a different request -- so it could
    print "ready" for a model the graph could not use at all (a provider that
    refuses streaming for a model got three OKs here and a failure at the
    very first graph call).
    """
    from langgraph.graph import END, START, StateGraph

    def node(_state: _CheckState) -> dict:
        return {"out": call()}

    builder = StateGraph(_CheckState)
    builder.add_node("check", node)
    builder.add_edge(START, "check")
    builder.add_edge("check", END)

    result = None
    for mode, data in builder.compile().stream({}, stream_mode=["updates", "messages"]):
        if mode == "updates" and "check" in (data or {}):
            result = data["check"]["out"]
    return result


def line(ok: bool | None, label: str, detail: str = "") -> None:
    mark = {True: "  OK  ", False: " FAIL ", None: " SKIP "}[ok]
    print(f"[{mark}] {label}" + (f"\n         {detail}" if detail else ""))


def short(exc: Exception, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:limit] + (" ..." if len(text) > limit else "")


def _use_self_test_script() -> None:
    """Point `--provider fake` at a script matching this file's checks.

    Without this, the fake provider serves the test-suite's script (which
    starts with a Plan) and the check fails for a reason that has nothing to
    do with your setup. A diagnostic tool that misdiagnoses itself is worse
    than no diagnostic tool.
    """
    from langchain_core.messages import AIMessage

    import src.llm as llm_module
    from src.schemas import Route
    from tests.fake_llm import scripted_model

    model = scripted_model(
        [
            AIMessage(content="pong"),
            Route(next="reporter", instruction="write it", reason="findings exist", step_index=1),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "group_aggregate",
                        "args": {
                            "group_by": ["region"],
                            "value_column": "revenue",
                            "agg": "sum",
                        },
                        "id": "check_1",
                    }
                ],
            ),
        ]
    )
    llm_module.get_llm.cache_clear()
    llm_module.get_llm = lambda *a, **k: model  # type: ignore[assignment]


def main() -> int:
    args = parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    try:
        from src.config import settings
    except ValueError as exc:  # src.config.ConfigError
        print(f"error in the settings (.env or environment): {exc}")
        return 2
    from src.schemas import Route
    from src.tools.analysis import ANALYSIS_TOOLS

    if settings.provider == "fake":
        # Self-test: prove this script's own three checks work, using a
        # scripted model that answers exactly what the checks ask for.
        _use_self_test_script()

    # Where the settings came from. Printing the provider without printing
    # the file it was read from sends people to edit the wrong thing -- most
    # often `.env.example`, which is only a template and is never loaded.
    from src.config import PROJECT_ROOT

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        source = str(env_file)
    else:
        source = f"{env_file} (MISSING -- defaults and shell variables only)"

    print(f"env file : {source}")
    print(f"provider : {settings.provider}")
    print(f"model    : {settings.resolved_model(settings.provider)}")
    print("-" * 62)

    failures: list[str] = []

    # --- 0. can we even build a client? -----------------------------------
    try:
        import src.llm as llm_module

        llm = llm_module.get_llm()
        line(True, "build client")
    except Exception as exc:
        line(False, "build client", short(exc))
        print("\nNothing else can run. Fix this first.")
        return 1

    # --- 1. plain chat (reporter) ----------------------------------------
    try:
        reply = through_graph(lambda: llm.invoke("Reply with exactly the word: pong"))
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        line(True, "plain chat  (reporter)", f"replied: {text.strip()[:60]!r}")
    except Exception as exc:
        line(False, "plain chat  (reporter)", short(exc))
        failures.append("plain chat")
        print("\nNothing else can run without plain chat. Fix this first.")
        return 1

    # --- 2. structured output (planner, supervisor) ------------------------
    try:
        route = through_graph(
            lambda: llm.with_structured_output(Route).invoke(
                "We have findings but no report yet. Choose the next worker."
            )
        )
        assert isinstance(route, Route)
        line(True, "structured output  (planner, supervisor)", f"next={route.next}")
    except Exception as exc:
        line(False, "structured output  (planner, supervisor)", short(exc))
        failures.append("structured output")

    # --- 3. tool calling (analyst, charter) -------------------------------
    try:
        bound = llm.bind_tools(ANALYSIS_TOOLS)
        reply = through_graph(
            lambda: bound.invoke(
                "Use the group_aggregate tool to sum the column 'revenue' grouped "
                "by the column 'region'. Call the tool; do not answer in prose."
            )
        )
        calls = getattr(reply, "tool_calls", None) or []
        if calls:
            line(True, "tool calling  (analyst, charter)", f"called: {calls[0]['name']}")
        else:
            line(
                False,
                "tool calling  (analyst, charter)",
                "the model answered in prose instead of calling a tool",
            )
            failures.append("tool calling")
    except Exception as exc:
        line(False, "tool calling  (analyst, charter)", short(exc))
        failures.append("tool calling")

    # --- verdict -----------------------------------------------------------
    print("-" * 62)
    if not failures:
        print("All three capabilities work. You are ready for a real run.")
        return 0

    print(f"Missing: {', '.join(failures)}\n")
    if "tool calling" in failures:
        print(
            "Without tool calling the analyst and charter nodes cannot work, so\n"
            "the agent has nothing to report. Pick a model that supports tools."
        )
    if "structured output" in failures:
        print(
            "Without structured output the planner and supervisor cannot route.\n"
            "A larger or newer model usually fixes this."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

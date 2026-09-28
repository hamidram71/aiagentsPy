"""Graph-level tests, all with a scripted model. No API key, no network.

These are the tests that catch the bugs that actually hurt: a wrong edge, a
missing reducer, an interrupt that loses state, a guardrail that never fires.
"""

from __future__ import annotations

from typing import Any

import pytest
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph import build_graph
from src.llm import get_llm
from src.schemas import FindingOut, Plan, Route
from src.state import initial_state
from tests.fake_llm import default_script, scripted_model

CSV = "data/sales.csv"
QUESTION = "Which region drives revenue?"


@pytest.fixture(autouse=True)
def clear_llm_cache():
    """get_llm is lru_cached; a stale cached client would leak between tests."""
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


LLM_CONSUMERS = (
    "src.llm",
    "src.nodes.planner",
    "src.nodes.supervisor",
    "src.nodes.analyst",
    "src.nodes.charter",
    "src.nodes.reporter",
)


def _install(script: list[Any], monkeypatch) -> Any:
    """Make get_llm() return one specific scripted model, everywhere."""
    model = scripted_model(script)
    for module in LLM_CONSUMERS:
        monkeypatch.setattr(f"{module}.get_llm", lambda *a, **k: model)
    return model


def _run(
    script,
    monkeypatch,
    override_settings,
    *,
    approval=False,
    thread="t1",
    **settings_overrides,
):
    override_settings(require_approval=approval, **settings_overrides)
    model = _install(script, monkeypatch)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    final = graph.invoke(initial_state(CSV, QUESTION), config=config)
    assert_script_matched(final)
    return graph, model, config, final


def assert_script_matched(final: dict) -> None:
    """Fail when the scripted model and the graph disagreed about the flow.

    A mismatch does not crash the graph: safe_node turns it into an error,
    the run limps on, and a test that only checks a few fields can stay
    green. Two tests here passed that way for a long time -- the charter
    consumed the reply meant for the reporter, the script "ran out" at just
    the right moment, and every assertion held. A test whose script was not
    followed is not testing what it says.
    """
    mismatches = [
        e for e in final.get("errors") or []
        if "script order mismatch" in e or "script exhausted" in e
    ]
    assert not mismatches, f"the graph did not follow the test script: {mismatches}"


# --- happy path ------------------------------------------------------------


def test_full_run_produces_findings_charts_and_a_report(
    monkeypatch, override_settings
) -> None:
    _, model, _, final = _run(default_script(), monkeypatch, override_settings)

    assert final["profile"]["rows"] > 400
    assert final["plan"]
    assert len(final["findings"]) == 1
    assert final["findings"][0]["source"] == "analyst"
    assert final["charts"], "chart path should be harvested from the tool output"
    assert final["report_path"].endswith(".md")
    assert model.script == [], "every scripted answer should have been consumed"


def test_each_worker_binds_its_own_tools(monkeypatch, override_settings) -> None:
    """Both bindings must be asserted, not just whichever ran last."""
    _, model, _, _ = _run(default_script(), monkeypatch, override_settings)

    # One binding per worker *turn*, so both workers appear more than once.
    distinct = {tuple(sorted(b)) for b in model.tool_bindings}
    assert len(distinct) == 2, f"expected two distinct tool sets, got {distinct}"

    analyst_tools = next(b for b in distinct if len(b) > 1)
    charter_tools = next(b for b in distinct if len(b) == 1)

    assert set(analyst_tools) == {
        "run_pandas", "save_cleaned_data", "column_stats", "group_aggregate",
        "correlations", "value_counts",
    }
    assert charter_tools == ("make_chart",)
    # the charter must NOT be able to run pandas, and vice versa
    assert "run_pandas" not in charter_tools
    assert "make_chart" not in analyst_tools


# --- human in the loop -----------------------------------------------------


def test_interrupt_pauses_then_resumes(monkeypatch, override_settings) -> None:
    override_settings(require_approval=True)
    model = _install(default_script(), monkeypatch)

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "hitl"}}

    first = graph.invoke(initial_state(CSV, QUESTION), config=config)
    assert "__interrupt__" in first
    payload = first["__interrupt__"][0].value
    assert payload["action"] == "review_plan"
    assert payload["plan"]
    # nothing has been analysed yet: the graph really is parked
    assert not graph.get_state(config).values.get("findings")

    final = graph.invoke(Command(resume="approve"), config=config)
    assert final["approved"] is True
    assert final["report_path"]
    assert model.script == []


def test_rejected_plan_goes_back_to_the_planner(monkeypatch, override_settings) -> None:
    override_settings(require_approval=True)
    revised = Plan(steps=["Revenue per channel", "Write it up"], rationale="as asked")
    script = [default_script()[0], revised, *default_script()[1:]]
    model = _install(script, monkeypatch)

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "reject"}}

    graph.invoke(initial_state(CSV, QUESTION), config=config)
    second = graph.invoke(Command(resume="split by channel instead"), config=config)

    # replanned, interrupted again, and the feedback reached the planner
    assert second["plan"] == revised.steps
    prompts = "".join(str(p) for p in model.seen_prompts)
    assert "split by channel instead" in prompts

    final = graph.invoke(Command(resume="approve"), config=config)
    assert final["report_path"]


# --- guardrails ------------------------------------------------------------


def test_supervisor_cannot_finish_without_a_report(
    monkeypatch, override_settings
) -> None:
    """A model that says FINISH too early gets overridden in code."""
    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        AIMessage(content="FINDING: something\nEVIDENCE: numbers"),
        Route(next="FINISH", instruction="", reason="I feel done", step_index=1),  # premature
        AIMessage(content="# Report\n\nbody"),  # the reporter runs anyway
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="early-finish", require_chart=False)

    assert final["report_path"], "FINISH was rewritten to reporter"
    assert model.script == []


def test_charter_before_any_finding_is_rerouted_to_the_analyst(
    monkeypatch, override_settings
) -> None:
    script = [
        default_script()[0],
        Route(next="charter", instruction="draw", reason="eager", step_index=1),  # nothing to draw yet
        AIMessage(content="FINDING: first real finding\nEVIDENCE: n=425"),
        Route(next="reporter", instruction="write", reason="done", step_index=1),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="eager-chart", require_chart=False)

    assert final["findings"][0]["text"].startswith("first real finding")
    assert model.script == []


def test_step_budget_forces_a_partial_report(monkeypatch, override_settings) -> None:
    """With max_steps=1 the supervisor must bail out after one hop."""
    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        AIMessage(content="FINDING: only finding\nEVIDENCE: x"),
        AIMessage(content="# Partial report"),  # forced reporter, no Route consumed
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="budget", max_steps=1
    )

    assert final["report_path"]
    assert len(final["findings"]) == 1
    assert model.script == []


# --- error handling --------------------------------------------------------


def test_a_failing_node_becomes_state_not_a_crash(
    monkeypatch, override_settings
) -> None:
    """A bad CSV path must end the run cleanly with the error recorded."""
    override_settings(require_approval=False)
    _install(default_script(), monkeypatch)

    graph = build_graph(checkpointer=InMemorySaver())
    final = graph.invoke(
        initial_state("data/does_not_exist.csv", QUESTION),
        config={"configurable": {"thread_id": "bad-csv"}},
    )

    assert final["failed_node"] == "loader"
    assert any("FileNotFoundError" in e for e in final["errors"])
    assert not final.get("report_path")


def test_tool_errors_come_back_to_the_model(monkeypatch, override_settings) -> None:
    """ToolNode(handle_tool_errors=True) must not kill the graph."""
    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "group_aggregate",
                    "args": {"group_by": ["nope"], "value_column": "revenue", "agg": "sum"},
                    "id": "bad_1",
                }
            ],
        ),
        AIMessage(content="FINDING: recovered after the bad call\nEVIDENCE: retried"),
        Route(next="reporter", instruction="write", reason="done", step_index=1),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="tool-error", require_chart=False)

    assert final["report_path"]
    assert "recovered" in final["findings"][0]["text"]
    assert model.script == []


# --- structure -------------------------------------------------------------


def test_graph_shape_is_what_we_think_it_is(monkeypatch, override_settings) -> None:
    _install(default_script(), monkeypatch)
    graph = build_graph(checkpointer=InMemorySaver())
    nodes = set(graph.get_graph().nodes)

    for expected in (
        "loader", "planner", "approval", "supervisor",
        "analyst", "analyst_tools", "charter", "charter_tools", "reporter",
    ):
        assert expected in nodes


def test_safe_node_does_not_swallow_the_interrupt(monkeypatch, override_settings) -> None:
    """The footgun regression test.

    `interrupt()` raises a GraphBubbleUp subclass, which is itself an
    Exception. A `safe_node` that caught it would silently skip the human
    pause. Here we wrap the approval node too and prove the pause survives.
    """
    from src.errors import safe_node
    import src.nodes.approval as approval_module

    override_settings(require_approval=True)
    _install(default_script(), monkeypatch)
    monkeypatch.setattr(
        approval_module,
        "approval_node",
        safe_node("approval")(approval_module.approval_node),
    )
    monkeypatch.setattr(
        "src.graph.approval_node", approval_module.approval_node, raising=False
    )

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "bubble-up"}}
    first = graph.invoke(initial_state(CSV, QUESTION), config=config)

    assert "__interrupt__" in first, "safe_node must re-raise GraphBubbleUp"
    assert not first.get("errors"), "the interrupt must not be logged as an error"


# --- regressions: the bugs a code review found -----------------------------
#
# Every test below started life as a real defect. They are the reason the
# earlier "happy path" tests are not enough: a scripted model that always
# behaves hides everything that matters about failure handling.


def test_failed_node_is_cleared_on_success(monkeypatch, override_settings) -> None:
    """The sticky-flag bug, at its root.

    `failed_node` is read by every route_* function. If a node does not clear
    it on success, one transient error poisons every later routing decision.
    """
    script = [
        default_script()[0],
        RuntimeError("429 rate limited"),  # supervisor's first call fails
        Route(next="analyst", instruction="go", reason="retry worked", step_index=1),
        AIMessage(content="FINDING: survived the blip\nEVIDENCE: n=425"),
        Route(next="reporter", instruction="write", reason="done", step_index=1),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(script, monkeypatch, override_settings, thread="sticky", require_chart=False)

    assert final["failed_node"] == "", "the flag must not survive a later success"
    assert final["report_path"], "a single 429 must not end the run"
    assert final["findings"][0]["text"] == "survived the blip"
    assert final["failure_count"] == 1
    assert model.script == []


def test_a_transient_analyst_failure_does_not_disable_its_tool_loop(
    monkeypatch, override_settings
) -> None:
    """With a sticky flag, one analyst failure orphaned every later tool call."""
    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        RuntimeError("timeout"),  # analyst turn 1 fails
        Route(next="analyst", instruction="try again", reason="retry", step_index=1),
        AIMessage(  # turn 2 wants a tool -- this must actually execute
            content="",
            tool_calls=[
                {
                    "name": "group_aggregate",
                    "args": {"group_by": ["region"], "value_column": "revenue", "agg": "sum"},
                    "id": "call_after_failure",
                }
            ],
        ),
        AIMessage(content="FINDING: tools still work\nEVIDENCE: North first"),
        Route(next="reporter", instruction="write", reason="done", step_index=1),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(script, monkeypatch, override_settings, thread="recover", require_chart=False)

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "the tool call after the failure was never executed"
    assert "North" in str(tool_messages[0].content)
    assert final["findings"][0]["text"] == "tools still work"
    assert model.script == []


def test_step_budget_bounds_the_analyst_tool_loop(monkeypatch, override_settings) -> None:
    """analyst <-> analyst_tools never passes through the supervisor.

    A supervisor-only budget therefore does not bound it, and a model that
    keeps calling tools runs to LangGraph's recursion limit -- thousands of
    paid calls. The budget has to be checked inside the loop.

    Budget of four: the supervisor (step 1), two tool-calling turns (steps 2
    and 3), and then -- because step 4 is the last one the budget allows --
    a wrap-up instead of a third tool call, so what the two calls computed
    is not thrown away. (This test once passed by coincidence: see the
    history of settings_consumers in conftest.py.)
    """
    from src.schemas import FindingOut

    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        *[AIMessage(content="", tool_calls=[
            {"name": "value_counts", "args": {"column": "region"}, "id": f"loop{i}"}
        ]) for i in range(2)],
        FindingOut(has_finding=True, statement="North has 110 orders",
                   evidence="value_counts(region)"),
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="loop-budget",
        max_steps=4, max_tool_rounds=10, require_chart=False,
    )

    assert final["report_path"], "the run must terminate, not hit the recursion limit"
    assert model.script == [], "two tool turns and a wrap-up fit in a budget of four"
    analyst_turns = [b for b in model.tool_bindings if "value_counts" in b]
    assert len(analyst_turns) == 2, "the loop must stop at the budget"
    assert final["findings"][0]["text"] == "North has 110 orders", "and keep what it found"


def test_reporter_failure_still_produces_a_report_and_ends(
    monkeypatch, override_settings
) -> None:
    """The last exit needs its own exit.

    Both supervisor bail-out branches route to the reporter, so a reporter
    that cannot run at all used to mean an endless supervisor <-> reporter
    loop. Now: a mechanical fallback report, then END.
    """
    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        AIMessage(content="FINDING: revenue is concentrated\nEVIDENCE: North 38%"),
        Route(next="reporter", instruction="write", reason="done", step_index=1),
        RuntimeError("provider down"),  # the reporter's model call fails
    ]
    _, model, _, final = _run(script, monkeypatch, override_settings, thread="rep-fail", require_chart=False)

    assert final["report_path"], "a fallback report must still be written"
    body = Path(final["report_path"]).read_text(encoding="utf-8")
    assert "revenue is concentrated" in body, "findings must survive into the fallback"
    assert "provider down" in body, "the fallback must say why it is a fallback"
    assert model.script == []


def test_an_unlabelled_analyst_reply_is_not_recorded_as_a_finding(
    monkeypatch, override_settings
) -> None:
    """A refusal is not evidence.

    With a `or text.strip()` fallback, "I cannot answer that" became a
    Finding, satisfied the supervisor's findings guard, and was handed to the
    reporter -- whose prompt treats findings as established fact.
    """
    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        AIMessage(content="I'm sorry, I can't answer that with these columns."),
        # No tool ran for this step, so nothing could back the reply: the
        # extractor is not even asked. (It used to be, with the previous
        # step's tool output as "evidence" -- see _extract_finding.)
        Route(next="analyst", instruction="try the region split", reason="reframe", step_index=1),
        AIMessage(content="FINDING: a real one\nEVIDENCE: numbers"),
        Route(next="reporter", instruction="write", reason="done", step_index=1),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(script, monkeypatch, override_settings, thread="refusal", require_chart=False)

    texts = [f["text"] for f in final["findings"]]
    assert texts == ["a real one"], f"a refusal leaked into the findings: {texts}"
    assert model.script == []


def test_two_reports_in_the_same_second_do_not_overwrite(
    monkeypatch, override_settings, tmp_path
) -> None:
    """Second-granular timestamps collide, and write_text is destructive."""
    import src.nodes.reporter as reporter_module

    monkeypatch.setattr(reporter_module, "OUTPUT_DIR", tmp_path)
    first = reporter_module._unique_path(tmp_path)
    first.write_text("one", encoding="utf-8")
    second = reporter_module._unique_path(tmp_path)

    assert second != first
    assert first.read_text(encoding="utf-8") == "one"


def test_zero_findings_never_reaches_the_reporter_model(
    monkeypatch, override_settings
) -> None:
    """The fabrication hole, closed in code rather than in a prompt.

    A real run on a local llama3.1 produced a fully invented report from zero
    findings: it named the wrong top region and assigned it 99% of the file's
    revenue. The reporter prompt already said "never invent a number"; the
    model ignored it. So now zero findings means the model is never asked.
    """
    script = [
        default_script()[0],
        Route(next="analyst", instruction="go", reason="start", step_index=1),
        # The analyst answers without the FINDING: label...
        AIMessage(content="The East region looks biggest to me."),
        # ...and no tool ran, so there is no number behind it: the extractor
        # is not even asked (see _extract_finding).
        # Deliberately NOTHING for the reporter. If the reporter called the
        # model, the script would be exhausted and ScriptEmpty would be raised.
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="no-findings", max_steps=2
    )

    assert final["findings"] == []
    assert final["report_path"], "an honest report must still be written"

    body = Path(final["report_path"]).read_text(encoding="utf-8")
    assert "No answer produced" in body
    assert "FINDING:" in body, "the report should explain what was missing"
    # The whole point: there was no scripted answer left for the reporter, and
    # the run still succeeded -- so the reporter made no model call at all.
    assert model.script == []
    assert "East" not in body, "nothing from the model's prose may leak into the report"


def test_the_supervisor_does_not_reissue_a_completed_plan_step(
    monkeypatch, override_settings
) -> None:
    """The loop this closes, observed on a real qwen2.5 run.

    Step 1 produced a correct finding. The very next supervisor turn asked
    for step 1 again, with the reason "(Not yet done.)" -- because the prompt
    showed the plan and the findings as two unrelated lists and left the
    model to infer the mapping. It inferred wrong, burned the step budget on
    repeats, and the run ended with a partial report.
    """
    script = [
        Plan(steps=["Revenue per region", "Monthly trend", "Report"], rationale="ok"),
        Route(next="analyst", instruction="Revenue per region", reason="step 1",
              step_index=1),
        AIMessage(content="FINDING: North leads with 1,373,177.21\nEVIDENCE: by region"),
        # The model tries step 1 again. Code must redirect it to a pending one.
        Route(next="analyst", instruction="Revenue per region again", reason="oops",
              step_index=1),
        AIMessage(content="FINDING: revenue rises through the year\nEVIDENCE: monthly"),
        Route(next="reporter", instruction="write", reason="done", step_index=0),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="no-repeat", max_steps=12, require_chart=False)

    assert final["addressed_steps"] == [1, 2], (
        f"the repeat was not redirected to a pending step: {final['addressed_steps']}"
    )
    assert len(final["findings"]) == 2
    assert final["report_path"]
    assert model.script == []

    # And the supervisor prompt must actually show the done/pending state.
    prompts = "".join(str(p) for p in model.seen_prompts)
    assert "[done]" in prompts and "[pending]" in prompts


def test_a_finding_with_no_dispatched_step_marks_nothing(
    monkeypatch, override_settings
) -> None:
    """step_index=0 means "no plan step", so nothing may be marked done."""
    script = [
        # No analysis step to finish, so the report may follow the ad-hoc work.
        Plan(steps=["Write the report"], rationale="ok"),
        Route(next="analyst", instruction="ad-hoc", reason="no step", step_index=0),
        AIMessage(content="FINDING: something true\nEVIDENCE: numbers"),
        Route(next="reporter", instruction="write", reason="done", step_index=0),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="no-step", require_chart=False
    )

    assert final["addressed_steps"] == []
    assert len(final["findings"]) == 1
    assert model.script == []


def test_an_unlabelled_reply_backed_by_a_tool_is_rescued_by_the_extractor(
    monkeypatch, override_settings
) -> None:
    """The exact failure a local qwen2.5 produced, now recovered.

    It called the tool, got the right totals, and then answered in prose
    instead of the FINDING:/EVIDENCE: format. Lesson 4 correctly refused to
    record an unlabelled reply, so the step stayed [pending] and the
    supervisor re-dispatched it until the budget ran out -- five repeats and
    an empty report.

    The fix is the project's own principle applied once more: when you need a
    shape, ask for a typed object instead of describing the shape in words.
    """
    script = [
        Plan(steps=["Revenue per region", "Report"], rationale="ok"),
        Route(next="analyst", instruction="Revenue per region", reason="step 1",
              step_index=1),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "group_aggregate",
                    "args": {"group_by": ["region"], "value_column": "revenue",
                             "agg": "sum"},
                    "id": "c1",
                }
            ],
        ),
        # prose, no FINDING: label -- exactly what the local model did
        AIMessage(content="It looks like North is clearly the biggest region here."),
        # the extractor turns it into a typed finding, grounded in the tool output
        FindingOut(
            has_finding=True,
            statement="North leads revenue with 1,373,177.21 of 3,252,176.04",
            evidence="region=North -> 1,373,177.21",
        ),
        Route(next="reporter", instruction="write", reason="done", step_index=0),
        AIMessage(content="# Report"),
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="rescued", require_chart=False)

    assert len(final["findings"]) == 1
    assert "1,373,177.21" in final["findings"][0]["text"]
    # and the plan step is now marked done, so no re-dispatch loop
    assert final["addressed_steps"] == [1]
    assert final["report_path"]
    assert model.script == []


def test_the_extractor_still_refuses_an_ungrounded_claim(
    monkeypatch, override_settings
) -> None:
    """The rescue must not become a second route to fabrication."""
    script = [
        Plan(steps=["Revenue per region", "Report"], rationale="ok"),
        Route(next="analyst", instruction="go", reason="step 1", step_index=1),
        AIMessage(content="I think East is probably the leader, but let me check later."),
        # no tool ran, so there is nothing to extract -- the extractor is not asked
        # Nothing further is scripted. The step budget then forces the
        # reporter, which with zero findings writes its report WITHOUT an LLM
        # call (lesson 6) -- so an empty script here is the proof.
    ]
    _, model, _, final = _run(
        script, monkeypatch, override_settings, thread="ungrounded", max_steps=2
    )

    assert final["findings"] == []
    assert final["addressed_steps"] == [], "a refusal must not mark a step done"
    body = Path(final["report_path"]).read_text(encoding="utf-8")
    assert "No answer produced" in body
    assert model.script == [], "no model call may remain unaccounted for"


# --- errors no retry can fix -------------------------------------------------


def _refusal(param: str = "reasoning_effort") -> Exception:
    """A real langchain-openai 400, as gpt-5.6-terra sent it."""
    lco = pytest.importorskip("langchain_openai.chat_models.base")
    import httpx

    return lco.OpenAIInvalidRequestError(
        "Error code: 400",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        ),
        body={
            "message": "Function tools with reasoning_effort are not supported ...",
            "type": "invalid_request_error",
            "param": param,
            "code": None,
        },
    )


def test_a_refused_request_stops_the_run_instead_of_burning_the_budget(
    monkeypatch, override_settings
) -> None:
    """The real run: the analyst got the same 400 four times, the supervisor
    started writing "with reasoning_effort disabled" into its instructions,
    and the failure budget was gone. Revert the UNFIXABLE_BY_RETRY branch in
    src/errors.py and this test sees that loop again instead of a stop."""
    from src.errors import FatalModelError

    script = default_script()
    with pytest.raises(FatalModelError) as caught:
        _run([script[0], script[1], _refusal()], monkeypatch, override_settings)

    assert caught.value.node == "analyst"


def test_after_a_fatal_stop_resume_finishes_without_having_spent_anything(
    monkeypatch, override_settings
) -> None:
    from src.errors import FatalModelError

    override_settings(require_approval=False)
    script = default_script()
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "fatal-then-resume"}}

    _install([script[0], script[1], _refusal()], monkeypatch)
    with pytest.raises(FatalModelError):
        graph.invoke(initial_state(CSV, QUESTION), config=config)

    paused = graph.get_state(config)
    assert paused.next == ("analyst",), "the checkpoint sits just before the node"
    assert (paused.values.get("failure_count") or 0) == 0
    assert not paused.values.get("errors")

    # The user fixes .env; the same thread continues from the analyst.
    _install(script[2:], monkeypatch)
    final = graph.invoke(None, config=config)

    assert final["findings"], "the resumed run completes the analysis"
    assert final.get("report_path")


def test_a_transient_error_is_still_data_for_the_supervisor(
    monkeypatch, override_settings
) -> None:
    """The fail-fast rule must stay narrow: a 429 can succeed next time."""
    from langchain_core.exceptions import ModelRateLimitError

    script = default_script()
    graph, _, _, final = _run(
        [script[0], script[1], ModelRateLimitError("429 slow down"), *script[1:]],
        monkeypatch,
        override_settings,
    )

    assert final["failure_count"] == 1
    assert any("429" in e for e in final["errors"])
    assert final["findings"], "the retry after a 429 still produced the analysis"

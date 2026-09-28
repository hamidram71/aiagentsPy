"""The OpenAI client against a fake OpenAI server.

Its own module because it needs `langchain-openai` at import time, and a
module-level skip must only ever skip what actually needs the package --
never the Anthropic or config tests next door (see 0.9.4 in CHANGELOG).
"""

from __future__ import annotations

import pytest

from src.llm import get_llm


@pytest.fixture(autouse=True)
def fresh_client_class():
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")


# --- parameters a model refuses ---------------------------------------------
#
# Two real runs died on a request parameter, not on the analysis:
#
#   gpt-6-luna     400: 'temperature' does not support 0.0 with this model.
#   gpt-5.6-terra  400: Function tools with reasoning_effort are not
#                  supported ... set reasoning_effort to 'none'.
#
# These tests run the REAL ChatOpenAI and the REAL openai SDK against a fake
# server (tests/fake_openai.py), and assert on the JSON that went over the
# wire. An earlier version of these tests monkeypatched `_generate` instead;
# that proved the retry logic and nothing about the request it produced.

openai_pkg = pytest.importorskip("langchain_openai", reason="optional provider")

from src.llm import _openai_class, rejected_parameter  # noqa: E402
from src.schemas import Route  # noqa: E402
from src.tools.analysis import ANALYSIS_TOOLS  # noqa: E402
from tests.fake_openai import FakeOpenAI  # noqa: E402

ROUTE = {"next": "analyst", "instruction": "x", "reason": "y", "step_index": 1}


def _client(server: FakeOpenAI):
    return _openai_class()(
        model="gpt-test", temperature=0.0, http_client=server.client(), max_retries=0
    )


def _sent(server: FakeOpenAI, key: str) -> list:
    return [body.get(key, "<absent>") for body in server.requests]


# The four doors every node uses to reach the model.
DOORS = {
    "invoke": lambda llm: llm.invoke("ping"),
    "structured": lambda llm: llm.with_structured_output(Route).invoke("route"),
    "tools": lambda llm: llm.bind_tools(ANALYSIS_TOOLS).invoke("sum revenue"),
    "tools-stream": lambda llm: list(llm.bind_tools(ANALYSIS_TOOLS).stream("sum revenue")),
}


@pytest.mark.parametrize("door", DOORS)
def test_a_refused_temperature_is_dropped_on_every_door(door) -> None:
    """Revert `_generate`/`_stream` in src/llm.py to plain calls: this fails."""
    server = FakeOpenAI(reject_temperature=True, structured_reply=ROUTE)
    llm = _client(server)

    DOORS[door](llm)

    assert _sent(server, "temperature") == [0.0, "<absent>"]
    assert llm.temperature is None, "the client must remember, not re-learn"


@pytest.mark.parametrize("door", ["tools", "tools-stream"])
def test_refused_reasoning_is_switched_off_for_tool_calls(door) -> None:
    """The exact gpt-5.6-terra failure. We never send reasoning_effort; the
    model applies its own default and refuses it in combination with tools."""
    server = FakeOpenAI(reject_tools_with_reasoning=True)
    llm = _client(server)

    DOORS[door](llm)

    assert _sent(server, "reasoning_effort") == ["<absent>", "none"]


def test_the_reasoning_repair_is_scoped_to_tool_calls_and_remembered() -> None:
    """Planner and supervisor worked with reasoning on. Keep it on for them."""
    server = FakeOpenAI(reject_tools_with_reasoning=True, structured_reply=ROUTE)
    llm = _client(server)

    DOORS["tools"](llm)          # learns: rejected, then repaired
    server.requests.clear()

    DOORS["tools"](llm)          # remembered: right the first time
    DOORS["structured"](llm)     # not a tool call: untouched

    assert _sent(server, "reasoning_effort") == ["none", "<absent>"]


def test_two_quirks_at_once_are_both_repaired_in_one_call() -> None:
    """A model with both quirks must not fail its first tool call."""
    server = FakeOpenAI(reject_temperature=True, reject_tools_with_reasoning=True)
    llm = _client(server)

    reply = DOORS["tools"](llm)

    assert reply.tool_calls, "the call should succeed after two repairs"
    assert len(server.requests) == 3


def test_a_request_is_never_resent_unchanged() -> None:
    """If a repair does not help, the error must surface -- not loop."""

    class Stubborn(FakeOpenAI):
        def handler(self, request):
            import httpx, json
            self.requests.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {
                "message": "Unsupported value: 'temperature'",
                "type": "invalid_request_error", "param": "temperature", "code": None}})

    server = Stubborn()
    llm = _client(server)

    with pytest.raises(openai_pkg.chat_models.base.OpenAIInvalidRequestError):
        DOORS["invoke"](llm)
    assert len(server.requests) == 2, "one original, one repaired -- then give up"


def test_rejected_parameter_reads_the_error_not_the_model_name() -> None:
    class Shaped(Exception):
        def __init__(self, param):
            super().__init__("Error code: 400")
            self.body = {"error": {"param": param, "type": "invalid_request_error"}}

    assert rejected_parameter(Shaped("temperature")) == "temperature"
    assert rejected_parameter(Shaped("reasoning_effort")) == "reasoning_effort"
    # A parameter we cannot repair is still reported, so it can be explained.
    assert rejected_parameter(Shaped("top_p")) == "top_p"
    # Prose fallback only for known parameters in an "unsupported" message.
    assert rejected_parameter(Exception("'temperature' is not supported")) == "temperature"
    assert rejected_parameter(RuntimeError("connection reset")) is None
    assert rejected_parameter(RuntimeError("401 incorrect api key")) is None


def test_a_stream_that_already_yielded_is_not_replayed(monkeypatch) -> None:
    """Retrying after partial output would print the first tokens twice."""
    from langchain_openai import ChatOpenAI

    class Refusal(Exception):
        body = {"error": {"param": "temperature"}}

    def half_then_fail(self, *args, **kwargs):
        yield "chunk-1"
        raise Refusal()

    monkeypatch.setattr(ChatOpenAI, "_stream", half_then_fail, raising=False)
    llm = _openai_class()(model="gpt-test", temperature=0.0)

    seen = []
    with pytest.raises(Refusal):
        for chunk in llm._stream([]):
            seen.append(chunk)
    assert seen == ["chunk-1"]

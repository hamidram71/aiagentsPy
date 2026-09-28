"""A scripted chat model, so the graph can be tested without an API key.

This is the most under-used trick in agent development. An agent is mostly
*plumbing*: reducers, routing, tool wiring, message pairing. All of that can
be tested deterministically if you replace the model with a queue of
pre-written answers. Then the only thing you need a real model for is judging
answer quality -- which is a separate, much smaller test suite.

The script is a single ordered queue. Every consumer pops from the front:

    Plan(...)                         <- planner's with_structured_output call
    Route(next="analyst", ...)        <- supervisor's with_structured_output call
    AIMessage(tool_calls=[...])       <- analyst's bind_tools call
    AIMessage("FINDING: ...")         <- analyst again
    ...

If a consumer pops something of the wrong type, that is a bug in your test
script and the model raises immediately instead of failing mysteriously later.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field


class ScriptEmpty(RuntimeError):
    """The graph asked for more model calls than the script provides."""


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written answers in order. Records what it was asked."""

    script: list[Any] = Field(default_factory=list)
    seen_prompts: list[list[BaseMessage]] = Field(default_factory=list)
    # One entry per bind_tools() call, in order. A single overwritten list
    # would only ever show the *last* binding, so a test claiming to check
    # "the analyst bound its five tools" would silently be checking the
    # charter's one tool instead.
    tool_bindings: list[list[str]] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # -- queue ------------------------------------------------------------
    def _pop(self, expected: type) -> Any:
        if not self.script:
            raise ScriptEmpty(
                f"script exhausted; graph wanted another {expected.__name__}"
            )
        item = self.script.pop(0)

        # An exception in the script means "this call fails" -- that is how
        # the tests simulate a 429, a timeout, or a provider outage.
        if isinstance(item, BaseException):
            raise item

        if not isinstance(item, expected):
            raise AssertionError(
                f"script order mismatch: graph wanted {expected.__name__}, "
                f"script had {type(item).__name__}"
            )
        return item

    # -- BaseChatModel ----------------------------------------------------
    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_prompts.append(list(messages))
        message = self._pop(AIMessage)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        self.tool_bindings.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any):
        return _ScriptedStructured(self, schema)


class _ScriptedStructured:
    """What `with_structured_output()` hands back: something with .invoke()."""

    def __init__(self, model: ScriptedChatModel, schema: type[BaseModel]) -> None:
        self._model = model
        self._schema = schema

    def invoke(self, messages: Any, **kwargs: Any) -> BaseModel:
        self._model.seen_prompts.append(
            messages if isinstance(messages, list) else [messages]
        )
        return self._model._pop(self._schema)


# --------------------------------------------------------------------------
# a default end-to-end script, used by `--provider fake`
# --------------------------------------------------------------------------


def default_script() -> list[Any]:
    """A complete happy-path run: plan -> analyse -> chart -> report."""
    from src.schemas import Plan, Route

    return [
        # Only steps this script actually analyses. The supervisor now refuses
        # to write the report while a plan step is still pending, so a plan
        # promising a trend step the script never analyses is a script bug.
        Plan(
            steps=[
                "Total revenue per region",
                "Write the report",
            ],
            rationale="The question is about regions and about a trend over time.",
        ),
        Route(next="analyst", instruction="Revenue per region", reason="start with the split", step_index=1),
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
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(
            content=(
                "FINDING: North is the largest region by revenue.\n"
                "EVIDENCE: sum(revenue) by region, North ranked first."
            )
        ),
        Route(next="charter", instruction="Bar chart of revenue per region", reason="show it", step_index=1),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "make_chart",
                    "args": {
                        "kind": "bar",
                        "x": "region",
                        "y": "revenue",
                        "agg": "sum",
                        "title": "North leads revenue",
                        "filename": "revenue_by_region.png",
                    },
                    "id": "call_2",
                }
            ],
        ),
        AIMessage(content="CHART: revenue_by_region.png - North's bar is the tallest."),
        Route(next="reporter", instruction="Write it up", reason="analysis done", step_index=1),
        AIMessage(content="# North leads revenue\n\nScripted report body."),
    ]


def scripted_model(script: list[Any] | None = None) -> ScriptedChatModel:
    return ScriptedChatModel(script=list(script) if script is not None else default_script())

"""Pydantic models used for structured LLM output.

`llm.with_structured_output(Model)` forces the model to answer with valid JSON
matching the schema. For routing decisions this is far more reliable than
parsing free text, because the set of allowed values is enforced by the API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """The supervisor's opening move: how it intends to answer the question."""

    steps: list[str] = Field(
        description=(
            "3 to 7 short, concrete analysis steps, in order. One line each, "
            "no more than 120 characters per step."
        ),
        min_length=1,
        max_length=8,
    )
    rationale: str = Field(
        max_length=200,
        description="One sentence on why this plan fits the data, max 200 characters.",
    )


class Route(BaseModel):
    """The supervisor's per-turn decision about who works next.

    Note the `max_length` on the free-text fields, which is not cosmetic.
    Structured output constrains the *shape* of the answer, not its *length*:
    an unbounded `str` lets a model write an essay inside a routing decision.
    Observed on a local qwen2.5: one routing turn generated ~650 tokens of
    prose in `instruction`, taking well over a minute on CPU, when the
    decision itself needs about 50. With a hosted model you pay for that in
    money; with a local one you pay in wall-clock.

    Bound every free-text field in a schema a model fills in. The limit also
    lands in the JSON Schema handed to the provider, so on backends that
    compile the schema into a grammar it is enforced, not merely requested.
    """

    next: Literal["analyst", "charter", "reporter", "FINISH"] = Field(
        description=(
            "analyst = compute numbers with pandas; "
            "charter = draw a chart; "
            "reporter = write the final markdown report; "
            "FINISH = everything is done and the report exists."
        )
    )
    instruction: str = Field(
        max_length=200,
        description=(
            "What exactly the chosen worker should do next, in one short "
            "sentence of at most 200 characters. Empty for FINISH."
        ),
    )
    reason: str = Field(
        max_length=120,
        description="Why this worker, in at most 120 characters.",
    )
    step_index: int = Field(
        description=(
            "Which numbered plan step this work addresses (1-based). Use 0 "
            "only for the final report or when no step applies. Never choose "
            "a step already marked [done]."
        ),
        ge=0,
        le=8,
    )


class FindingOut(BaseModel):
    """One analytical result, extracted as a typed object.

    Why this exists: the analyst prompt asks for a `FINDING:` / `EVIDENCE:`
    text format, and hosted models follow it. A local qwen2.5 did not -- it
    called the tool, got the right numbers, then answered in plain prose. No
    label meant no finding (correctly, see lesson 4), the plan step stayed
    [pending], and the supervisor re-dispatched it until the budget ran out.

    The project already knows the answer to this problem: when you need a
    specific shape from a model, do not describe the shape in words and hope
    -- ask for a typed object. That is what routing does. So when the text
    format is missing, a second, cheap structured call extracts the same
    information from what the model actually said plus the tool output it
    actually received.
    """

    has_finding: bool = Field(
        description=(
            "True only if the conversation contains a concrete, numeric "
            "result computed by a tool. False for refusals, restatements of "
            "the question, or plans to do work later."
        )
    )
    statement: str = Field(
        default="",
        max_length=300,
        description=(
            "One sentence stating what is true, including the number. Empty "
            "when has_finding is false."
        ),
    )
    evidence: str = Field(
        default="",
        max_length=600,
        description="The tool figures supporting it. Empty when has_finding is false.",
    )

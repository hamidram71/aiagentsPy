"""Planner node: one structured LLM call that turns the question into steps."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from src.dataset import profile_to_text
from src.errors import safe_node
from src.llm import get_llm
from src.prompts import PLANNER_SYSTEM
from src.schemas import Plan
from src.state import AgentState


@safe_node("planner")
def planner_node(state: AgentState) -> dict:
    """Produce (or revise) the analysis plan.

    Runs a second time when the human rejected the first plan; the feedback is
    appended to the prompt so the model actually changes something.
    """
    llm = get_llm().with_structured_output(Plan)

    prompt = (
        f"Question: {state['question']}\n\n"
        f"Data profile:\n{profile_to_text(state['profile'])}"
    )
    feedback = state.get("human_feedback")
    if feedback:
        prompt += (
            f"\n\nYour previous plan was rejected by the analyst with this "
            f"feedback:\n{feedback}\n\nProduce a different plan that addresses it."
        )

    plan: Plan = llm.invoke([SystemMessage(content=PLANNER_SYSTEM), ("human", prompt)])
    steps = with_report_step(plan.steps)

    rendered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
    header = f"Plan ({plan.rationale})"
    if feedback:
        header += f", revised after feedback: {feedback}"
    return {
        "plan": steps,
        "human_feedback": "",
        # A fixed id: add_messages REPLACES a message with the same id. So a
        # revised plan takes the rejected plan's place in the transcript
        # instead of sitting next to it. Before, the workers saw both plans,
        # with nothing saying which one had been rejected.
        "messages": [AIMessage(id=PLAN_MESSAGE_ID, content=f"{header}:\n{rendered}")],
    }


PLAN_MESSAGE_ID = "plan"

_REPORT_WORDS = ("report", "summar", "write", "conclu")


def with_report_step(steps: list[str]) -> list[str]:
    """Make sure the plan ends with the report step.

    The supervisor treats the last step as the report, and the prompt tells
    the planner to end with it -- but a model returned the two halves of the
    question and nothing else, and the second half, "monthly revenue
    trend", was then never analysed: as "the last step" it counted as the
    report. The code knows what it relies on, so it checks.
    """
    steps = [s for s in steps if s.strip()]
    if steps and any(word in steps[-1].lower() for word in _REPORT_WORDS):
        return steps
    return [*steps, "Write the report"]

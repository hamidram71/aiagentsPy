"""Human-in-the-loop gate.

`interrupt()` is the LangGraph primitive for "stop here and ask a human". It
raises a special exception that the runtime catches: the checkpointer saves
the exact position in the graph, `.stream()` returns with an `__interrupt__`
payload, and the process can exit. Later -- seconds or days later, in a
different process -- you resume with

    graph.invoke(Command(resume=<value>), config)

and the very same `interrupt()` call *returns* `<value>` as if nothing had
happened. That is why a durable checkpointer (SQLite here, Postgres in
production) is not optional for this pattern: the checkpoint *is* the pause.

Note the consequence: everything before `interrupt()` in a node runs again on
resume. Keep side effects (writes, emails, payments) strictly *after* it.
"""

from __future__ import annotations

from langgraph.types import interrupt

from src.state import AgentState


def approval_node(state: AgentState) -> dict:
    """Show the plan to a human and wait for approve / feedback.

    Accepted resume values:
        True, "approve", "yes", "y", "ok"  -> proceed
        any other string                   -> treated as revision feedback
    """
    payload = {
        "action": "review_plan",
        "question": state["question"],
        "plan": state.get("plan", []),
        "hint": "Reply 'approve' to run it, or describe what to change.",
    }

    answer = interrupt(payload)

    if isinstance(answer, bool):
        return {"approved": answer, "human_feedback": "" if answer else "rejected"}

    text = str(answer or "").strip()
    if text.lower() in {"approve", "approved", "yes", "y", "ok", "go", ""}:
        return {"approved": True, "human_feedback": ""}

    return {"approved": False, "human_feedback": text}

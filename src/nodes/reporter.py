"""Reporter node: one LLM call, then a real file on disk.

Deliberately the only node that writes a deliverable. Keeping side effects in
a single, clearly named node is what makes a graph auditable: when a file
appears, you know exactly which node produced it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage

from src.config import OUTPUT_DIR
from src.dataset import output_folder
from src.errors import safe_node
from src.llm import get_llm
from src.prompts import REPORTER_SYSTEM
from src.state import AgentState


@safe_node("reporter")
def reporter_node(state: AgentState) -> dict:
    """Write outputs/report_<timestamp>.md and record its path in the state."""
    findings = state.get("findings") or []
    charts = state.get("charts") or []
    errors = state.get("errors") or []

    # --- the hole this closes ---------------------------------------------
    # With zero findings, the old code still called the model with
    # "(no findings were produced -- say so plainly)" and trusted it to obey.
    # A local llama3.1 obeyed by inventing an entire report: it named the
    # wrong top region and gave it 99% of the file's revenue, with numbers
    # that all ended in the same digits because it had generated a
    # plausible-looking descending series rather than computing anything.
    #
    # The lesson is the one this project keeps repeating: a prompt *asks*,
    # code *guarantees*. There is nothing for a model to write here, so do
    # not ask it. No findings -> no model call, and a report that says so.
    if not findings:
        path = _unique_path(output_folder(state, OUTPUT_DIR))
        path.write_text(_no_findings_report(state), encoding="utf-8")
        return {
            "report_path": str(path),
            "messages": [
                AIMessage(content=f"Reporter: no findings; wrote {path} without an LLM call.")
            ],
        }

    findings_block = (
        "\n\n".join(
            f"[{i}] {f['text']}\n    evidence: {f['evidence']}"
            for i, f in enumerate(findings, 1)
        )
        or "(no findings were produced -- say so plainly in the report)"
    )
    charts_block = "\n".join(f"- {Path(c).name} (at {c})" for c in charts) or "(none)"
    errors_note = (
        "\n\nSteps that failed during the run (mention the relevant ones under Caveats):\n"
        + "\n".join(f"- {e}" for e in errors[-5:])
        if errors
        else ""
    )

    # What save_cleaned_data changed, straight from the state -- so the report
    # can say which rows were corrected or dropped without depending on the
    # analyst having happened to repeat it inside a finding.
    cleaning = state.get("cleaning_log") or []
    cleaning_block = (
        "Data cleaning applied before the findings (state it under Caveats). Each "
        "entry has the analyst's own description and then what was 'Verified from "
        "the data'. Take row ids, values and counts ONLY from the verified part; "
        "where the two disagree, the verified part is right:\n"
        + "\n".join(f"- {c}" for c in cleaning)
        + "\n\n"
        if cleaning
        else ""
    )

    prompt = (
        f"Question: {state['question']}\n"
        f"Dataset: {state['csv_path']} "
        f"({state.get('profile', {}).get('rows', '?')} rows)\n\n"
        f"{cleaning_block}"
        f"Findings:\n{findings_block}\n\n"
        f"Charts available (reference them with just the file name, they sit "
        f"next to the report):\n{charts_block}"
        f"{errors_note}\n\nWrite the report."
    )

    # The reporter is the graph's last exit: every bail-out branch in the
    # supervisor routes here. So this node must not be able to fail on the
    # model call -- if it did, the only way out of the graph would be the one
    # thing that is broken. Hence the fallback: a mechanical report built
    # straight from the findings, no LLM involved.
    try:
        reply = get_llm().invoke(
            [SystemMessage(content=REPORTER_SYSTEM), ("human", prompt)]
        )
        markdown = reply.content if isinstance(reply.content, str) else str(reply.content)
        if not charts:
            markdown = drop_section(markdown, "Charts")
        note = ""
    except Exception as exc:  # noqa: BLE001
        markdown = _fallback_report(state, findings, charts, exc)
        note = " (fallback, model call failed)"

    path = _unique_path(output_folder(state, OUTPUT_DIR))
    path.write_text(markdown, encoding="utf-8")

    return {
        "report_path": str(path),
        "messages": [AIMessage(content=f"Reporter: wrote {path}{note}")],
    }


def _unique_path(directory: Path) -> Path:
    """A fresh filename every time.

    Second-granular timestamps collide: two reports in the same second and
    `write_text` silently destroys the first. Cheap to prevent, invisible
    when it happens.
    """
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"report_{stamp}.md"
    counter = 2
    while path.exists():
        path = directory / f"report_{stamp}_{counter}.md"
        counter += 1
    return path


def _no_findings_report(state: AgentState) -> str:
    """An honest empty report. No model involved, so nothing can be invented."""
    errors = state.get("errors") or []
    lines = [
        f"# No answer produced: {state['question']}",
        "",
        "The analysis finished without producing a single verified finding, so "
        "there is nothing to report. **No numbers appear below on purpose** — "
        "an empty analysis must not turn into a confident-looking report.",
        "",
        f"**Dataset.** `{state['csv_path']}` "
        f"({state.get('profile', {}).get('rows', '?')} rows)",
        "",
        "## Why this happened",
        "",
        "Usually one of:",
        "",
        "- the analyst never emitted a line starting with `FINDING:`, so nothing "
        "was recorded (common with weaker models — add an explicit "
        "example to `ANALYST_SYSTEM` in `src/prompts.py`);",
        "- the step budget (`MAX_STEPS`) ran out before the analysis finished;",
        "- the tools kept failing. See below.",
    ]
    if errors:
        lines += ["", "## Failures during the run", ""]
        lines += [f"- {e}" for e in errors[-8:]]
    return "\n".join(lines)


def _fallback_report(
    state: AgentState, findings: list, charts: list, exc: Exception
) -> str:
    """A report anyone can read, assembled without a model."""
    lines = [
        f"# Partial report: {state['question']}",
        "",
        "> The report-writing model call failed "
        f"(`{type(exc).__name__}: {exc}`). Below are the raw findings the "
        "analysis produced, unedited.",
        "",
        f"**Dataset.** `{state['csv_path']}` "
        f"({state.get('profile', {}).get('rows', '?')} rows)",
        "",
        "## Findings",
    ]
    if findings:
        for i, finding in enumerate(findings, 1):
            lines += [f"{i}. {finding['text']}", f"   - evidence: {finding['evidence']}"]
    else:
        lines.append("No findings were produced.")

    if charts:
        lines += ["", "## Charts"]
        lines += [f"- `{Path(c).name}`" for c in charts]

    cleaning = state.get("cleaning_log") or []
    if cleaning:
        lines += ["", "## Data cleaning applied"]
        lines += [f"- {c}" for c in cleaning]

    errors = state.get("errors") or []
    if errors:
        lines += ["", "## Failures during the run"]
        lines += [f"- {e}" for e in errors[-8:]]

    return "\n".join(lines)


def drop_section(markdown: str, heading: str) -> str:
    """Remove a `## heading` section, up to the next `## ` heading or the end.

    REPORTER_SYSTEM tells the model to leave out the Charts section when there
    are no charts. On a real run it wrote "## Charts / No charts available."
    instead -- twice. Whether a section is empty is a fact the code already
    knows, so the code enforces it rather than asking a third time.
    """
    import re

    pattern = rf"(?ms)^##[ \t]+{re.escape(heading)}[ \t]*\n.*?(?=^##[ \t]|\Z)"
    return re.sub(pattern, "", markdown).rstrip() + "\n"

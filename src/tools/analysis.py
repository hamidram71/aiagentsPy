"""Tools the analyst node can call.

Two things worth noticing:

1. **`InjectedState`.** The `state` argument is annotated
   `Annotated[dict, InjectedState]`, which tells `ToolNode` to fill it in from
   the graph state instead of exposing it in the tool schema. The model is
   therefore never *asked* for the CSV path and cannot invent one -- a wrong
   path becomes impossible rather than merely unlikely. Any infrastructure
   detail a tool needs should be injected this way.

   Be precise about what this does and does not buy you: it removes the
   argument from the tool schema, not the value from the prompt. In this
   project the loader deliberately puts the path in the seed message, so the
   model does see the string. If a value must never reach the model at all,
   injecting it is necessary but not sufficient -- keep it out of the
   transcript too.

2. **Narrow tools before the general one.** `run_pandas` can do everything, so
   it is tempting to ship only that. But a narrow tool like `group_aggregate`
   has a typed signature the model can hardly get wrong, while free-form code
   fails in a hundred creative ways. Give the model specific tools for the
   common 80% and keep the code escape hatch for the rest.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.config import settings
from src.dataset import save_working_copy, working_df
from src.formatting import fmt_key, fmt_number
from src.sandbox import run_snippet

AggName = Literal["sum", "mean", "count", "median", "min", "max", "nunique"]


def _df(state: dict):
    """The working dataframe: the cleaned copy if one was saved, else the CSV."""
    return working_df(state)


@tool
def run_pandas(code: str, state: Annotated[dict, InjectedState]) -> str:
    """Run a short pandas snippet against the loaded dataframe.

    The dataframe is available as `df`; `pd` and `np` are imported for you.
    The snippet MUST assign its answer to a variable named `result`.
    Imports are rejected, as is any method that reads or writes files.

    Example:
        result = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    """
    outcome = run_snippet(
        code,
        _df(state),
        timeout=settings.code_timeout,
        max_rows=settings.max_rows_to_model,
    )
    return outcome.to_model_string()


@tool
def column_stats(
    columns: list[str], state: Annotated[dict, InjectedState]
) -> str:
    """Descriptive statistics for one or more columns (count/mean/std/quartiles)."""
    df = _df(state)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return f"ERROR: unknown columns {missing}. Available: {list(df.columns)}"
    return df[columns].describe(include="all").to_string()


@tool
def group_aggregate(
    group_by: list[str],
    value_column: str,
    agg: AggName,
    state: Annotated[dict, InjectedState],
    top_n: int | None = 15,
    ascending: bool = False,
) -> str:
    """Group rows by one or more columns and aggregate a value column.

    Use this instead of `run_pandas` for plain group-by questions such as
    "revenue per region" or "average order value per month".

    top_n limits how many groups are shown; pass null for all of them.
    """
    df = _df(state)
    unknown = [c for c in [*group_by, value_column] if c not in df.columns]
    if unknown:
        return f"ERROR: unknown columns {unknown}. Available: {list(df.columns)}"

    # dropna=False: pandas' default silently drops every row whose group key
    # is empty. On sales.csv that was 9 rows and 40,498.07 of revenue gone
    # from "revenue per channel", while the header still said "3 of 3
    # groups". A missing key is a group of its own, shown as "(blank)".
    grouped = df.groupby(group_by, dropna=False)[value_column].agg(agg)
    total_groups = len(grouped)
    grouped = grouped.sort_values(ascending=ascending)
    # `top_n: int = 15` rejected the null a model naturally sends when it
    # wants every group, and ToolNode handed back a ValidationError. The
    # model recovered on the next turn -- but a wasted turn on a local model
    # is half a minute. Accept the input that the tool's own docstring
    # invites rather than making the model guess the type.
    if top_n is not None:
        grouped = grouped.head(top_n)

    # Every row states its own keys, and the header states how many groups
    # exist. Both matter, and both were learned the hard way.
    #
    # `Series.to_string()` on a MultiIndex *elides a repeated level*:
    #
    #     region  order_date
    #     East    2025-03-24    41573.82
    #             2025-10-25    40579.02     <- blank region
    #
    # A human reads the blank as "same as above". A model filled it in with
    # a different region entirely and reported East's number as West's. The
    # meaning of that output depends on visual alignment, and alignment is
    # exactly what a token stream destroys -- so never hand a model a table
    # whose rows are only interpretable by looking upwards.
    #
    # The group count matters for the same family of reasons: truncated to
    # the top 15 of 366 region-date pairs, a model concluded "only one date
    # is provided" and reported that no trend could be measured. Say what
    # was left out, not just what is shown.
    lines = [
        f"{agg}({value_column}) grouped by {group_by} "
        f"-- showing {'bottom' if ascending else 'top'} {len(grouped)} "
        f"of {total_groups} groups:"
    ]
    for keys, value in grouped.items():
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        label = " | ".join(f"{col}={fmt_key(k)}" for col, k in zip(group_by, key_tuple))
        # Round floats: a model handed 40579.020000000004 will copy that
        # binary-float artefact straight into the report. But not to 0.00:
        # see fmt_number for why small values keep significant figures.
        lines.append(f"  {label} -> {fmt_number(value)}")

    if total_groups > len(grouped):
        lines.append(
            f"  ... [{total_groups - len(grouped)} more groups not shown. "
            f"These are NOT totals per {group_by[0]}; each row is one "
            f"{'/'.join(group_by)} combination.]"
            if len(group_by) > 1
            else f"  ... [{total_groups - len(grouped)} more groups not shown]"
        )
    return "\n".join(lines)


@tool
def correlations(
    state: Annotated[dict, InjectedState],
    columns: list[str] | None = None,
    threshold: float = 0.3,
) -> str:
    """Pairwise Pearson correlations between numeric columns.

    Only pairs with |r| >= threshold are returned, strongest first.
    """
    df = _df(state)
    numeric = df.select_dtypes("number")
    if columns:
        unknown = [c for c in columns if c not in numeric.columns]
        if unknown:
            return f"ERROR: not numeric or unknown: {unknown}"
        numeric = numeric[columns]
    if numeric.shape[1] < 2:
        return "ERROR: need at least two numeric columns"

    matrix = numeric.corr(numeric_only=True)
    pairs = []
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = matrix.loc[a, b]
            if abs(r) >= threshold:
                pairs.append((abs(r), f"{a} ~ {b}: r={r:+.3f}"))
    if not pairs:
        return f"no pair reaches |r| >= {threshold}"
    pairs.sort(reverse=True)
    return "\n".join(line for _, line in pairs)


@tool
def value_counts(
    column: str, state: Annotated[dict, InjectedState], top_n: int = 15
) -> str:
    """Frequency of each distinct value in a column, most common first."""
    df = _df(state)
    if column not in df.columns:
        return f"ERROR: unknown column {column!r}. Available: {list(df.columns)}"
    all_counts = df[column].value_counts(dropna=False)
    counts = all_counts.head(top_n)
    share = counts / len(df) * 100
    # The header and footer say what was left out -- the rule the sandbox
    # already follows ("truncation is always announced"). Without them, 15
    # of 88 customers looked like the full list.
    lines = [f"{column}: {len(all_counts)} distinct values -- showing top {len(counts)}:"]
    lines += [
        f"  {fmt_key(key)}: {count} ({share.iloc[i]:.1f}%)"
        for i, (key, count) in enumerate(counts.items())
    ]
    if len(all_counts) > len(counts):
        rest = int(all_counts.iloc[len(counts):].sum())
        lines.append(
            f"  ... [{len(all_counts) - len(counts)} more values not shown, "
            f"{rest} rows]"
        )
    return "\n".join(lines)


# A cleaning step may drop rows, but a snippet that drops most of them is far
# more likely a bug (a filter written backwards) than a cleaning decision.
MAX_ROWS_DROPPED = 0.5


@tool
def save_cleaned_data(
    code: str,
    description: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Replace the working dataset with a cleaned version, for ALL later tools.

    Use this once you have decided how to clean the data: dropping duplicate
    rows, correcting a wrong value, excluding rows. From then on every tool --
    including the chart tool -- works on the cleaned data, so you do not have
    to repeat the cleaning in each snippet.

    `code` is a pandas snippet like run_pandas: `df` is the current working
    data, and you assign the cleaned DataFrame to `result`. Keep the columns.

    `description` says in one sentence exactly what changed, with row ids and
    values -- it goes into the report.

    Example:
        code = (
            'df.loc[df["order_id"] == "SO-1012", "revenue"] = 4326.98\n'
            'result = df.drop_duplicates("order_id")'
        )
        description = "dropped 5 duplicate order_id rows; set SO-1012 revenue to its formula value 4326.98"
    """

    def reply(text: str) -> Command:
        return Command(update={"messages": [ToolMessage(text, tool_call_id=tool_call_id)]})

    before = _df(state)
    outcome = run_snippet(code, before, timeout=settings.code_timeout)
    if not outcome.ok:
        return reply(outcome.to_model_string())

    after = outcome.value
    if not isinstance(after, pd.DataFrame):
        return reply(
            f"ERROR: `result` must be a DataFrame, got {type(after).__name__}. "
            "Nothing was saved."
        )
    missing = [c for c in before.columns if c not in after.columns]
    if missing:
        return reply(
            f"ERROR: the cleaned data lost columns {missing}. Keep every column; "
            "nothing was saved."
        )
    if len(after) > len(before):
        return reply(
            f"ERROR: this would turn {len(before)} rows into {len(after)}. Cleaning "
            "removes or corrects rows; it never adds them -- a merge that fans "
            "out or a concat that repeats the data doubles every total later. "
            "Nothing was saved."
        )
    if len(after) < len(before) * (1 - MAX_ROWS_DROPPED):
        return reply(
            f"ERROR: this would keep {len(after)} of {len(before)} rows. Cleaning "
            "that removes more than half the data is almost always a bug in the "
            "filter. Nothing was saved."
        )

    changed = [
        f"{c} {before[c].dtype} -> {after[c].dtype}"
        for c in before.columns
        if str(before[c].dtype) != str(after[c].dtype)
    ]
    path = save_working_copy(after, state.get("run_dir"))
    # The description is the MODEL's account of what it did, written from
    # memory. On a real run it listed the five removed duplicates and got one
    # id wrong (SO-1188, an order that occurs once, instead of SO-1260), and
    # the report repeated it. What actually changed is computed here from the
    # data itself, and the reporter is told to trust that over the account.
    verified = describe_changes(before, after)
    note = (
        f"{description.strip()} ({len(before)} -> {len(after)} rows). "
        f"Verified from the data: {verified}"
    )
    if changed:
        # Not refused -- a type change can be the point of the cleaning --
        # but said out loud: a date turned into text silently breaks every
        # later time chart.
        note += "; column types changed: " + ", ".join(changed)
    return Command(
        update={
            "data_path": path,
            "cleaning_log": [note],
            "messages": [
                ToolMessage(
                    f"saved cleaned data: {note}. Every tool now uses it.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def describe_changes(before: pd.DataFrame, after: pd.DataFrame, limit: int = 8) -> str:
    """What a cleaning step actually did: removed rows and changed values.

    Rows are matched by index -- run_snippet works on a copy that keeps it --
    and named by an id column when the data has one (order_id, customer_id
    ...), so the report can say "removed SO-1003" rather than "row 12".

    Matching by index is only valid while the rows still line up. After a
    reset_index the labels 0..419 are still a subset of 0..424, and a first
    version compared row 12 of the cleaned data with row 12 of the original
    -- a different order -- and produced a confident, entirely wrong list.
    A real cleaning step leaves most rows untouched, so if fewer than half
    are unchanged, the rows no longer correspond and nothing is listed.
    """
    unreliable = (
        "the rows no longer line up with the original (index reset or rows "
        "reordered), so which rows changed cannot be listed"
    )
    if after.index.has_duplicates or not after.index.isin(before.index).all():
        return unreliable

    id_column = _id_column(before, after)

    def name(index) -> str:
        return str(before.at[index, id_column]) if id_column else f"row {index}"

    def shorten(items: list[str]) -> str:
        more = len(items) - limit
        return ", ".join(items[:limit]) + (f", ... and {more} more" if more > 0 else "")

    def shown(value) -> str:
        return fmt_number(value) if pd.api.types.is_number(value) else str(value)

    kept = after.index
    row_unchanged = np.ones(len(kept), dtype=bool)
    changes: list[str] = []
    for column in before.columns:
        if column not in after.columns or str(before[column].dtype) != str(after[column].dtype):
            continue  # a type change is reported separately; cell-by-cell it is noise
        old, new = before.loc[kept, column], after.loc[kept, column]
        if pd.api.types.is_numeric_dtype(old):
            same = np.isclose(old.astype(float), new.astype(float), rtol=0, atol=1e-9, equal_nan=True)
        else:
            same = np.asarray((old == new) | (old.isna() & new.isna()), dtype=bool)
        row_unchanged &= same
        changes += [f"{name(i)} {column}: {shown(old[i])} -> {shown(new[i])}" for i in old.index[~same]]

    if len(kept) and row_unchanged.mean() < 0.5:
        return unreliable

    parts: list[str] = []
    removed = [index for index in before.index if index not in after.index]
    if removed:
        noun = "row" if len(removed) == 1 else "rows"
        parts.append(f"removed {len(removed)} {noun} ({shorten(sorted(name(i) for i in removed))})")
    if changes:
        noun = "value" if len(changes) == 1 else "values"
        parts.append(f"changed {len(changes)} {noun} ({shorten(changes)})")

    return "; ".join(parts) if parts else "no rows removed and no values changed"


def _id_column(before: pd.DataFrame, after: pd.DataFrame) -> str | None:
    """The column that names a row: *_id or id, unique once cleaned."""
    candidates = [c for c in before.columns if c.lower() == "id" or c.lower().endswith("_id")]
    for column in candidates:
        if column in after.columns and after[column].is_unique:
            return column
    return candidates[0] if candidates else None


ANALYSIS_TOOLS = [
    run_pandas,
    save_cleaned_data,
    column_stats,
    group_aggregate,
    correlations,
    value_counts,
]

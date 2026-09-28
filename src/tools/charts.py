"""Chart tool. Matplotlib on the Agg backend, so it works headless."""

from __future__ import annotations

import re
import textwrap
import threading
from pathlib import Path
from typing import Annotated, Literal

import matplotlib

matplotlib.use("Agg")  # must happen before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.prebuilt import InjectedState  # noqa: E402

from src.config import OUTPUT_DIR  # noqa: E402
from src.dataset import output_folder, working_df  # noqa: E402
from src.formatting import fmt_key, fmt_number  # noqa: E402

_PLOT_LOCK = threading.Lock()

SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")

# pandas resample aliases, and how each bucket is labelled on the axis.
BUCKETS = {
    "day": ("D", "%Y-%m-%d"),
    "week": ("W", "%Y-%m-%d"),
    "month": ("MS", "%Y-%m"),
    "quarter": ("QS", "%Y-%m"),
}

ADJECTIVE = {"day": "daily", "week": "weekly", "month": "monthly", "quarter": "quarterly"}

# A line with more points than this is almost always a trend drawn at the
# wrong grain: 366 daily totals are noise, 12 monthly totals are a trend.
MAX_LINE_POINTS = 60

# Bars get their value written on them up to this many; beyond, it is clutter.
MAX_LABELLED_BARS = 20

COLOR = "#3b6ea8"


def _safe_path(filename: str, directory: Path | None = None):
    """Keep the model inside the output folder no matter what it puts in `filename`."""
    name = SAFE_NAME.sub("_", filename.strip()) or "chart.png"
    if not name.endswith(".png"):
        name += ".png"
    return (directory or OUTPUT_DIR) / name


@tool
def make_chart(
    kind: Literal["bar", "line", "hist", "scatter", "box"],
    x: str,
    title: str,
    filename: str,
    state: Annotated[dict, InjectedState],
    y: str | None = None,
    agg: Literal["sum", "mean", "count", "median", "none"] = "none",
    top_n: int = 15,
    time_bucket: Literal["day", "week", "month", "quarter"] | None = None,
) -> str:
    """Draw one chart and save it as a PNG under outputs/.

    kind=hist and kind=box need only `x`. Everything else needs `x` and `y`.
    Set `agg` to aggregate `y` per distinct `x` before plotting -- e.g.
    kind="bar", x="region", y="revenue", agg="sum".

    For a trend over time, x must be a date column and you should set
    `time_bucket` -- e.g. kind="line", x="order_date", y="revenue",
    agg="sum", time_bucket="month". Without it every distinct date becomes
    its own point.

    Charts use the cleaned working data if save_cleaned_data was called.

    Returns the saved file path, which you should mention in the report.
    """
    df = working_df(state)
    source = "cleaned working data" if state.get("data_path") else "the raw CSV"
    notes: list[str] = []

    for col in filter(None, [x, y]):
        if col not in df.columns:
            return f"ERROR: unknown column {col!r}. Available: {list(df.columns)}"

    # pyplot keeps its figures in one global registry, and the web app draws
    # charts for several people at once. One chart at a time: drawing takes
    # a fraction of a second, a corrupted figure would take an afternoon.
    with _PLOT_LOCK:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=130)

        try:
            if kind == "hist":
                df[x].plot(kind="hist", bins=30, ax=ax, edgecolor="white", color=COLOR)
                ax.set_xlabel(x)

            elif kind == "box":
                if y:
                    df.boxplot(column=y, by=x, ax=ax, grid=False)
                    fig.suptitle("")
                else:
                    df[[x]].plot(kind="box", ax=ax)

            elif kind == "scatter":
                if not y:
                    return "ERROR: scatter needs both x and y"
                df.plot(kind="scatter", x=x, y=y, ax=ax, alpha=0.6, color=COLOR)

            elif time_bucket:  # bar | line over time buckets
                if not y:
                    return f"ERROR: {kind} needs both x and y"
                if not pd.api.types.is_datetime64_any_dtype(df[x]):
                    return (
                        f"ERROR: time_bucket needs a date column, and {x!r} is "
                        f"{df[x].dtype}. Use a date column, or drop time_bucket."
                    )
                freq, label = BUCKETS[time_bucket]
                values = df.set_index(x)[y]
                if agg in ("sum", "none"):
                    # min_count=1: a bucket with no values is missing, not 0.
                    # Summing an all-empty column used to draw a flat line at
                    # zero and report "peak = 0.00".
                    series = values.resample(freq).sum(min_count=1)
                else:
                    series = values.resample(freq).agg(agg)
                if series.dropna().empty:
                    return (
                        f"ERROR: {y!r} has no values in any {time_bucket} -- nothing "
                        "to plot. Nothing was saved."
                    )
                series.index = series.index.strftime(label)
                if kind == "line":
                    series.plot(kind="line", ax=ax, marker="o", color=COLOR)
                    if len(series) <= 24:
                        # Label every bucket. Matplotlib's default skips every
                        # other month, and a monthly chart that hides half its
                        # months invites the reader to guess.
                        ax.set_xticks(range(len(series)))
                        ax.set_xticklabels(series.index)
                else:
                    # No `marker` here: bar plots reject it, and passing
                    # marker=None made EVERY monthly bar chart fail with
                    # "Rectangle.set() got an unexpected keyword argument".
                    series.plot(kind="bar", ax=ax, color=COLOR)
                ax.set_ylabel(f"{agg if agg != 'none' else 'sum'}({y}) per {time_bucket}")
                ax.set_xlabel(x)
                # Report what was actually plotted, not just what was asked for:
                # the charter needs it to describe the chart truthfully, and a
                # test can only prove the chart used the cleaned data by looking
                # at the numbers it drew.
                valid = series.dropna()
                notes.append(
                    f"{len(series)} {ADJECTIVE[time_bucket]} points; "
                    f"peak {valid.idxmax()} = {fmt_number(valid.max())}; "
                    f"first {fmt_number(valid.iloc[0])}, last {fmt_number(valid.iloc[-1])}"
                )
                if len(valid) < len(series):
                    notes.append(f"{len(series) - len(valid)} {time_bucket}s have no data")
                if kind == "line" and len(series) > MAX_LINE_POINTS:
                    notes.append(
                        f"WARNING: {len(series)} points -- for a trend, a coarser "
                        "time_bucket (week, month) reads better"
                    )

            else:  # bar | line, no time buckets
                if not y:
                    return f"ERROR: {kind} needs both x and y"
                if agg != "none":
                    # dropna=False, as in group_aggregate: rows with an empty key
                    # are a group of their own, not silently left out.
                    series = df.groupby(x, dropna=False)[y].agg(agg)
                    series.index = [fmt_key(k) for k in series.index]
                    total_groups = len(series)
                    if kind == "bar":
                        series = series.sort_values(ascending=False).head(top_n)
                        notes.append(
                            f"{len(series)} of {total_groups} {x} groups shown: "
                            + ", ".join(f"{k} = {fmt_number(v)}" for k, v in series.head(6).items())
                            + (" ..." if len(series) > 6 else "")
                        )
                    else:
                        series = series.sort_index()
                else:
                    # Sorted by x: rows in file order drew a zigzag line as soon
                    # as a cleaning step reordered them.
                    series = df.set_index(x)[y].sort_index()
                series.plot(kind=kind, ax=ax, color=COLOR)
                ax.set_ylabel(f"{agg}({y})" if agg != "none" else y)
                ax.set_xlabel(x)
                if kind == "line" and len(series) > MAX_LINE_POINTS:
                    notes.append(
                        f"WARNING: {len(series)} points -- if this is a trend over "
                        "time, redraw with time_bucket='month'"
                    )

            if kind in {"bar", "line"}:
                _format_value_axis(ax)
            if kind == "bar" and ax.containers and len(ax.containers[0]) <= MAX_LABELLED_BARS:
                # The value on each bar: a reader should not have to estimate
                # 1,355,773 from a gridline.
                ax.bar_label(
                    ax.containers[0],
                    labels=[_short(b.get_height()) for b in ax.containers[0]],
                    padding=2,
                    fontsize=8,
                )
                # Headroom, so the label on the tallest bar is not pressed
                # against the frame.
                low, high = ax.get_ylim()
                ax.set_ylim(low, high + (high - low) * 0.06)
            ax.set_title(textwrap.fill(title, 70))
            ax.grid(axis="y", alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.autofmt_xdate(rotation=30)
            fig.tight_layout()

            path = _safe_path(filename, output_folder(state, OUTPUT_DIR))
            fig.savefig(path)
            # The path alone on the first line: charter._saved_charts reads it
            # from there. Anything for the model goes on the lines below.
            detail = "; ".join([f"drawn from {source}", *notes])
            return f"saved chart: {path}\n{detail}"

        except Exception as exc:  # noqa: BLE001 - hand the failure back to the model
            return f"ERROR while plotting: {type(exc).__name__}: {exc}"
        finally:
            plt.close(fig)


def _short(value: float) -> str:
    """Compact label for a bar: 1.36M, 622.8k, 0.0446."""
    if pd.isna(value):
        return ""
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 10_000:
        return f"{value / 1_000:.1f}k"
    return fmt_number(value)


def _format_value_axis(ax) -> None:
    """Readable tick labels at any scale.

    Whole numbers with separators for amounts (1,400,000 -- not "1.4" under
    a "1e6" the reader has to spot), but significant figures for small
    values: the old `{:,.0f}` everywhere labelled every tick of a rate chart
    (0.02, 0.04, 0.06) as "0".
    """
    low, high = ax.get_ylim()
    if max(abs(low), abs(high)) >= 100:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:.3g}"))


CHART_TOOLS = [make_chart]

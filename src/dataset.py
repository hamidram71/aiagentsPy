"""Loading and profiling the CSV. No LLM involved anywhere in this file.

Rule of thumb for agents: anything that can be computed deterministically
should be computed deterministically. The model's job is to decide *what* to
compute and to interpret the result -- not to do arithmetic.
"""

from __future__ import annotations

import json
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import OUTPUT_DIR


# A column is tried as a date only if its name says so -- in English or in
# German ("Datum", "Lieferzeit", "Monat"): a German file's date column was
# never recognised, because "Datum" contains none of the English words.
DATE_NAME_TOKENS = ("date", "time", "day", "month", "datum", "zeit", "tag", "monat")


@lru_cache(maxsize=8)
def load_df(csv_path: str) -> pd.DataFrame:
    """Load a CSV once and reuse it for the rest of the process.

    Cached on the path, so a tool called ten times parses the file once.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = read_csv_file(path)

    # Best-effort datetime parsing: any text column whose name smells like a
    # date and which parses cleanly becomes a real datetime column.
    #
    # "Text column" is `_is_text`, NOT `dtype == "object"`. That comparison
    # was the original test, and pandas 3 quietly broke it: text now loads as
    # the `str` dtype, `str == "object"` is False, and no column was ever
    # parsed again. Nothing failed. The planner prompt went on telling the
    # model "date columns are already parsed to datetime", which had become a
    # lie, and the chart tool could not draw a time axis. A library upgrade
    # turned a true statement in a prompt into a false one -- and only a test
    # on the *result* (is order_date a datetime?) would have noticed.
    for col in df.columns:
        if _is_text(df[col]) and any(
            token in col.lower() for token in DATE_NAME_TOKENS
        ):
            parsed = _parse_dates(df[col])
            if parsed is not None:
                df[col] = parsed

    return df


def read_csv_file(source: Any) -> pd.DataFrame:
    """Read a CSV the way Excel writes it -- in Germany as well.

    German Excel saves "CSV" with semicolons between columns, commas as the
    decimal mark ("1234,56") and, unless you pick "CSV UTF-8", in Windows'
    cp1252 encoding (ä, ö, ü, ß, €). Read as a plain comma CSV, such a file
    became ONE column named "datum;region;umsatz" -- or a UnicodeDecodeError.
    `source` is a path or an open binary buffer (an upload).
    """
    def attempt(**options: Any) -> pd.DataFrame:
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            return pd.read_csv(source, encoding="utf-8-sig", **options)
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding="cp1252", **options)

    df = attempt()
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = attempt(sep=";", decimal=",")
    return df


# German dates are written day first with dots: 13.02.2025. Without saying
# so, pandas reads 01.02.2025 as January 2nd -- and then cannot make sense of
# 13.02.2025 at all, so the whole column stayed text.
_DOTTED_DATE = r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}"


# A value looks like a calendar date only if it carries a year or a full
# d/m/y-style triple. Without this check "Jan"/"Feb" (a month_name column)
# parsed as year 1, and "09:15" (an order_time column) got TODAY's date
# attached -- both turned into dates that mean nothing.
_DATE_LIKE = r"\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"


def _parse_dates(series: pd.Series) -> pd.Series | None:
    """The column as datetimes, or None if it is not really a date column.

    The share of values that parse is measured over NON-BLANK values. It used
    to be over all rows, so a date column with more than 10% empty cells was
    never recognised -- and after a cleaning step that set bad dates to NaT,
    the saved working copy silently lost its date type.
    """
    present = series.dropna()
    present = present[present.astype(str).str.strip() != ""]
    if present.empty:
        return None
    if present.astype(str).str.contains(_DATE_LIKE, regex=True).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        # "Could not infer format" is noise here: a column that does not
        # parse is simply left as text.
        warnings.simplefilter("ignore", UserWarning)
        day_first = bool(present.astype(str).str.match(_DOTTED_DATE).mean() >= 0.9)
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=day_first)
    if parsed[present.index].notna().mean() <= 0.9:
        return None
    return parsed


def _is_text(series: pd.Series) -> bool:
    """True for text columns under pandas 2 (`object`) and pandas 3 (`str`)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def working_df(state: dict) -> pd.DataFrame:
    """The dataframe every tool should use: the cleaned copy if there is one.

    Before this existed, cleaning lived only inside the analyst's snippets. The
    analyst dropped duplicates and fixed SO-1012 in one `run_pandas` call --
    and every other tool, the chart tool included, went on reading the raw
    file. A chart drawn from that would have shown the fake November spike
    that the report had just explained away. Every tool asks here, so
    cleaning is done once and seen by all of them.
    """
    path = state.get("data_path") or state.get("csv_path")
    if not path:
        raise ValueError("csv_path missing from state")
    return load_df(path)


def output_folder(state: dict, fallback: Path) -> Path:
    """Where this run writes its files: its own run folder, else `fallback`."""
    folder = Path(state["run_dir"]) if state.get("run_dir") else fallback
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_working_copy(df: pd.DataFrame, directory: str | Path | None = None) -> str:
    """Write a cleaned dataframe next to the other outputs; return its path.

    A file rather than an in-memory object, so it survives a checkpoint:
    `--resume` in a new process still finds the cleaned data.
    """
    folder = Path(directory) if directory else OUTPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"working_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(path, index=False)
    return str(path)


def profile_df(df: pd.DataFrame, max_categories: int = 8) -> dict[str, Any]:
    """A compact, model-friendly description of the dataframe.

    Deliberately small: this whole dict ends up inside the prompt, so it must
    describe the data without dumping it.
    """
    numeric = df.select_dtypes("number").columns.tolist()
    datetime_cols = df.select_dtypes("datetime").columns.tolist()
    categorical = [
        c for c in df.columns if c not in numeric and c not in datetime_cols
    ]

    columns: dict[str, Any] = {}
    for col in df.columns:
        info: dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
        }
        if col in numeric:
            desc = df[col].describe()
            info |= {
                "min": _py(desc.get("min")),
                "max": _py(desc.get("max")),
                "mean": _py(desc.get("mean")),
                "std": _py(desc.get("std")),
            }
        elif col in datetime_cols:
            info |= {
                "min": str(df[col].min()),
                "max": str(df[col].max()),
            }
        elif info["unique"] <= max_categories:
            info["values"] = [str(v) for v in df[col].dropna().unique()[:max_categories]]
        columns[col] = info

    return {
        "rows": int(len(df)),
        "columns_count": int(df.shape[1]),
        "numeric_columns": numeric,
        "datetime_columns": datetime_cols,
        "categorical_columns": categorical,
        "columns": columns,
        # JSON-native values only. pandas Timestamps here were stored in the
        # checkpoint, and LangGraph refuses to rebuild them on load: every
        # --resume printed "Blocked deserialization of ... Timestamp" nine
        # times. Dates as ISO strings say the same thing and load cleanly.
        "head": json.loads(df.head(3).to_json(orient="records", date_format="iso")),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _py(value: Any) -> Any:
    """numpy scalar -> plain Python, so the dict is JSON-serialisable."""
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def profile_to_text(profile: dict[str, Any]) -> str:
    """Render the profile as the compact text block we put in prompts."""
    lines = [
        f"rows={profile['rows']}, columns={profile['columns_count']}, "
        f"duplicate_rows={profile['duplicate_rows']}",
        "columns:",
    ]
    for name, info in profile["columns"].items():
        bits = [f"dtype={info['dtype']}", f"nulls={info['nulls']}", f"unique={info['unique']}"]
        for key in ("min", "max", "mean", "values"):
            if info.get(key) is not None:
                bits.append(f"{key}={info[key]}")
        lines.append(f"  - {name}: " + ", ".join(str(b) for b in bits))
    return "\n".join(lines)

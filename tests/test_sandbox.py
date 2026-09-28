"""Sandbox tests. These are the tests you must never delete."""

from __future__ import annotations


import pandas as pd
import pytest

from src.sandbox import UnsafeCodeError, run_snippet, validate_code


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["N", "S", "N", "E"],
            "revenue": [10.0, 5.0, 7.0, 2.0],
            "units": [1, 2, 3, 4],
        }
    )


# --- layer 1: static validation -------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from os import system",
        "result = open('/etc/passwd').read()",
        "result = ().__class__.__bases__",
        "result = eval('1+1')",
        "result = __import__('os')",
        "df.to_csv('/tmp/leak.csv')\nresult = 1",
        "with open('x') as f:\n    result = 1",
        "result = getattr(df, 'to_csv')",
    ],
)
def test_rejects_dangerous_code(code: str) -> None:
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


@pytest.mark.parametrize(
    "code",
    [
        "result = df['revenue'].sum()",
        "result = df.groupby('region')['revenue'].mean()",
        "result = pd.concat([df, df]).shape",
        "total = 0\nfor v in df['units']:\n    total += v\nresult = total",
        "result = np.log1p(df['revenue']).round(2).tolist()",
    ],
)
def test_accepts_normal_pandas(code: str) -> None:
    validate_code(code)  # must not raise


# --- end to end -----------------------------------------------------------


def test_result_is_returned(df: pd.DataFrame) -> None:
    out = run_snippet("result = df['revenue'].sum()", df)
    assert out.ok
    assert "24.0" in out.result_repr


def test_missing_result_is_an_error(df: pd.DataFrame) -> None:
    out = run_snippet("x = df['revenue'].sum()", df)
    assert not out.ok
    assert "result" in out.error


def test_exception_is_reported_not_raised(df: pd.DataFrame) -> None:
    out = run_snippet("result = df['nope'].sum()", df)
    assert not out.ok
    assert "KeyError" in out.error


def test_stdout_is_captured(df: pd.DataFrame) -> None:
    out = run_snippet("print('hello')\nresult = 1", df)
    assert out.ok
    assert "hello" in out.stdout


def test_snippet_cannot_mutate_the_original_df(df: pd.DataFrame) -> None:
    out = run_snippet("df['revenue'] = 0\nresult = df['revenue'].sum()", df)
    assert out.ok
    assert df["revenue"].sum() == 24.0  # the caller's frame is untouched


def _bounded(code: str, df: pd.DataFrame, timeout: int = 1, watchdog: float = 15.0):
    """run_snippet in a helper thread, with a watchdog.

    If the timeout ever breaks, these tests must FAIL, not hang: a broken
    timeout once froze the whole suite until an outer limit killed it.
    """
    import threading

    box: dict = {}
    worker = threading.Thread(
        target=lambda: box.update(out=run_snippet(code, df, timeout=timeout)), daemon=True
    )
    worker.start()
    worker.join(watchdog)
    assert not worker.is_alive(), f"the snippet was not interrupted within {watchdog}s"
    return box["out"]


def test_timeout_stops_an_infinite_loop(df: pd.DataFrame) -> None:
    """On every OS now. The SIGALRM version needed a skipif on Windows, where
    this test would otherwise hang the suite forever."""
    out = _bounded("while True:\n    pass", df)
    assert not out.ok
    assert "exceeded" in out.error


def test_timeout_also_stops_a_loop_inside_a_lambda(df: pd.DataFrame) -> None:
    """Lambdas defined in the snippet are snippet code, so they are traced."""
    out = _bounded("result = df['revenue'].apply(lambda v: [0 for _ in range(10**9)])", df)
    assert not out.ok and "exceeded" in out.error


def test_the_timeout_leaves_no_trace_function_behind(df: pd.DataFrame) -> None:
    """A debugger's or coverage tool's tracer must be restored afterwards."""
    import sys

    before = sys.gettrace()
    run_snippet("result = 1", df, timeout=1)
    assert sys.gettrace() is before
    # and after a snippet that had to be interrupted (in this thread, since
    # the trace function is per thread -- watchdog-guarded against a hang)
    import threading

    seen = {}

    def interrupted():
        mine = sys.gettrace()
        run_snippet("while True:\n    pass", df, timeout=1)
        seen["restored"] = sys.gettrace() is mine

    worker = threading.Thread(target=interrupted, daemon=True)
    worker.start()
    worker.join(15)
    assert not worker.is_alive(), "the snippet was not interrupted"
    assert seen["restored"]


def test_big_frames_are_truncated_for_the_model() -> None:
    big = pd.DataFrame({"a": range(500)})
    out = run_snippet("result = df", big, max_rows=10)
    assert out.ok
    assert "500 rows total" in out.result_repr


# --- the to_* / read_* rule ------------------------------------------------
# A blocklist of "to_csv, to_excel, ..." goes stale every pandas release, so
# the rule is an allow-list of pure in-memory converters instead. These are
# the methods a stale blocklist used to let through.


@pytest.mark.parametrize(
    "code",
    [
        "result = df.head().to_markdown('/tmp/leak.md')",
        "result = df.head().to_html('/tmp/leak.html')",
        "result = df.head().to_xml('/tmp/leak.xml')",
        "result = df.head().to_latex('/tmp/leak.tex')",
        "result = df.to_feather('/tmp/leak.f')",
        "result = df.to_clipboard()",
        "result = pd.read_csv('/etc/hostname').shape",
        "result = pd.read_json('/etc/hostname')",
    ],
)
def test_every_io_method_is_refused_not_just_the_listed_ones(code: str) -> None:
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


@pytest.mark.parametrize(
    "code",
    [
        "result = df.head().to_dict()",
        "result = df['revenue'].to_numpy().mean()",
        "result = df['region'].value_counts().to_frame()",
        "result = df.describe().to_string()",
    ],
)
def test_in_memory_converters_still_work(code: str) -> None:
    validate_code(code)


# --- false rejections ------------------------------------------------------
# Every snippet below is ordinary pandas that an earlier version of the
# validator refused. A model that hits one of these gets an error that reads
# like a security refusal and burns a turn guessing what it did wrong.


@pytest.mark.parametrize(
    "code",
    [
        # list.remove collided with a blocklist entry meant for os.remove
        "cols = list(df.columns)\ncols.remove('region')\nresult = df[cols].shape",
        # no exception class was in scope, so no snippet could catch anything
        "try:\n    result = df['revenue'].sum()\nexcept (ValueError, KeyError):\n    result = 0",
        # next / iter / type were missing from the builtins allow-list
        "result = next(iter(df.groupby('region')))[0]",
        "result = type(df['revenue'].sum())",
        # `with` was banned outright to stop open(), but open() is already gone
        "with pd.option_context('display.max_rows', 5):\n    result = df.head().to_string()",
    ],
)
def test_ordinary_pandas_is_not_wrongly_rejected(code: str, df: pd.DataFrame) -> None:
    out = run_snippet(code, df)
    assert out.ok, f"wrongly rejected: {out.error}"


def test_truncation_is_always_announced(df: pd.DataFrame) -> None:
    """A model told nothing about truncation describes the rows it never saw."""
    out = run_snippet("result = list(range(5000))", df, max_rows=10)
    assert out.ok
    assert "5000 items" in out.result_repr

    out = run_snippet("result = {i: i for i in range(100)}", df, max_rows=5)
    assert out.ok
    assert "100 items" in out.result_repr

    out = run_snippet("result = np.arange(1000)", df, max_rows=10)
    assert out.ok
    assert "shape (1000,)" in out.result_repr


def test_the_timeout_works_in_a_worker_thread_too() -> None:
    """Where the graph actually runs tools: LangGraph's ToolNode uses a worker
    thread. With SIGALRM, a run_pandas call there either crashed ("signal only
    works in main thread") or -- after the first fix -- ran with no timeout at
    all. Now it is interrupted like anywhere else."""
    import threading
    import time

    import pandas as pd

    from src.sandbox import run_snippet

    box = {}

    def work():
        started = time.monotonic()
        box["out"] = run_snippet("while True:\n    pass", pd.DataFrame({"a": [1]}), timeout=1)
        box["took"] = time.monotonic() - started

    worker = threading.Thread(target=work)
    worker.start()
    worker.join(timeout=20)

    assert not worker.is_alive(), "the loop was never interrupted"
    assert not box["out"].ok and "exceeded" in box["out"].error
    assert box["took"] < 5

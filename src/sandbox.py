"""A small, honest sandbox for LLM-generated pandas code.

Letting a model run code it wrote is the single most useful and the single
most dangerous tool you can give an agent. This module is the "advanced"
heart of the project, so read it carefully.

Three layers of defence, in order of usefulness:

1. **Static validation (AST).** Before anything runs we parse the snippet and
   walk the tree. Imports, dunder attributes, dangerous builtin names, every
   `read_*` method and every `to_*` method that is not a pure in-memory
   converter are rejected. This catches `import os; os.system(...)`,
   `().__class__.__bases__[0].__subclasses__()` style escapes, and
   `df.to_parquet('/tmp/x')`.

   Note the shape of the `to_*` rule: an allow-list, not a blocklist. pandas
   adds IO methods every release, so a list of "to_csv, to_excel, ..." goes
   stale and quietly lets `to_markdown(path)` through. Enumerate what is
   *permitted* whenever the permitted set is the one that changes slowly.
2. **A restricted namespace.** `__builtins__` is replaced by a short allow-list,
   so even valid-looking code cannot reach `open` or `__import__`.
3. **A wall-clock timeout.** Stops `while True:` and accidental cartesian
   joins from hanging the graph.

What this is NOT: a real security boundary. A determined attacker will get
out of any in-process Python sandbox. For untrusted input in production you
want a container, gVisor, or a dedicated service -- not this file. It *is*
enough to stop a well-behaved model from wrecking your laptop by accident,
which is the actual failure mode 99% of the time.
"""

from __future__ import annotations

import ast
import contextlib
import io
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Layer 1: static validation
# --------------------------------------------------------------------------

FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "exit", "quit",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "breakpoint", "help", "memoryview",
}

# --- filesystem reach, handled by rule rather than by list ----------------
# pandas grows new IO methods every release, so an explicit blocklist of
# to_csv/to_excel/... goes stale and leaves to_markdown, to_xml, to_hdf and
# friends wide open. Instead: every `to_*` attribute is refused unless it is
# on this allow-list of pure in-memory converters, and every `read_*` is
# refused outright.
ALLOWED_TO_METHODS = {
    "to_dict", "to_list", "to_numpy", "to_frame", "to_string", "to_records",
    "to_series", "to_period", "to_timestamp", "to_datetime", "to_numeric",
    "to_flat_index", "to_pydatetime", "to_bytes",
}

FORBIDDEN_ATTRS = {"system", "popen", "spawn", "fork", "rmtree"}

# Exception classes the snippet may catch. Without these, no snippet can
# contain a typed `except`, which is ordinary defensive pandas code.
_EXCEPTIONS = {
    "Exception": Exception,
    "ArithmeticError": ArithmeticError,
    "AttributeError": AttributeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}

ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    **_EXCEPTIONS,
}


class UnsafeCodeError(ValueError):
    """Raised when static validation rejects a snippet."""


def validate_code(code: str) -> None:
    """Raise UnsafeCodeError if `code` is not allowed to run."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"SyntaxError: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError(
                "imports are not allowed; `pd`, `np` and `df` are already available"
            )

        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__"):
                raise UnsafeCodeError(
                    f"dunder attribute {attr!r} is not allowed "
                    "(the classic sandbox escape route)"
                )
            if attr in FORBIDDEN_ATTRS:
                raise UnsafeCodeError(f"attribute {attr!r} is not allowed")
            if attr.startswith("read_"):
                raise UnsafeCodeError(
                    f"{attr!r} reads from disk; the dataframe is already loaded as `df`"
                )
            if attr.startswith("to_") and attr not in ALLOWED_TO_METHODS:
                raise UnsafeCodeError(
                    f"{attr!r} can write to disk; use to_dict/to_string/to_numpy "
                    "to inspect data instead"
                )

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"name {node.id!r} is not allowed")

        if isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCodeError("async code is not allowed")


# --------------------------------------------------------------------------
# Layer 3: timeout
# --------------------------------------------------------------------------


class CodeTimeout(TimeoutError):
    """Raised when a snippet exceeds its wall-clock budget."""


SNIPPET_FILENAME = "<snippet>"


@contextlib.contextmanager
def time_limit(seconds: int):
    """Interrupt the snippet after `seconds` -- on every OS, in any thread.

    History, because each version failed differently:

    1. SIGALRM. Unix only, main thread only. The code checked only the Unix
       half, and inside the graph LangGraph's ToolNode runs tools in a
       worker thread: "signal only works in main thread", and every
       run_pandas call on Linux/macOS failed.
    2. SIGALRM, skipped off the main thread. No more crash -- and no more
       timeout: in a real run the tools are ALWAYS off the main thread, so
       CODE_TIMEOUT was never enforced on any OS. A snippet with CODE_TIMEOUT
       set to 1 ran for three seconds. Honest about it, but unprotected.
    3. This: a trace function, which works in any thread on any OS. Python
       calls it on every new frame and on every line of the snippet itself,
       and it raises CodeTimeout once the deadline has passed. Lines are
       traced only in the snippet's own code (and lambdas it defines), so
       pandas runs at full speed; a single long C-level operation still
       cannot be interrupted until it returns -- a limit worth knowing.
    """
    deadline = time.monotonic() + seconds

    def check(frame, event, arg):  # noqa: ARG001
        if time.monotonic() > deadline:
            raise CodeTimeout(f"code exceeded {seconds}s")
        # Trace line by line only inside the snippet; elsewhere, only the
        # call boundary above is checked.
        return check if frame.f_code.co_filename == SNIPPET_FILENAME else None

    previous = sys.gettrace()
    sys.settrace(check)
    try:
        yield True
    finally:
        sys.settrace(previous)


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------


@dataclass
class ExecResult:
    ok: bool
    result_repr: str = ""
    stdout: str = ""
    error: str = ""
    warning: str = ""
    # The object the snippet assigned to `result`, for callers that need the
    # value itself (save_cleaned_data) rather than its text rendering.
    value: Any = field(default=None, repr=False)

    def to_model_string(self, max_chars: int = 4000) -> str:
        """Format for feeding straight back to the LLM as a ToolMessage."""
        if not self.ok:
            return f"ERROR: {self.error}"
        parts = []
        if self.warning:
            parts.append(f"warning: {self.warning}")
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.strip()}")
        parts.append(f"result:\n{self.result_repr}")
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
        return text


def run_snippet(
    code: str,
    df: pd.DataFrame,
    timeout: int = 15,
    max_rows: int = 20,
) -> ExecResult:
    """Validate, then execute `code` against a copy of `df`.

    Contract with the model: the snippet must assign its answer to a variable
    named `result`. That single convention removes all the guesswork about
    what the snippet "returned".
    """
    try:
        validate_code(code)
    except UnsafeCodeError as exc:
        return ExecResult(ok=False, error=f"rejected by sandbox: {exc}")

    # A copy, so a snippet that mutates `df` cannot poison later steps.
    namespace: dict[str, Any] = {
        "__builtins__": ALLOWED_BUILTINS,
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }

    buffer = io.StringIO()
    warning = ""
    try:
        with time_limit(timeout), contextlib.redirect_stdout(buffer):
            exec(compile(code, SNIPPET_FILENAME, "exec"), namespace)  # noqa: S102
    except CodeTimeout as exc:
        return ExecResult(ok=False, stdout=buffer.getvalue(), error=str(exc))
    except Exception as exc:  # noqa: BLE001 - report any failure to the model
        return ExecResult(
            ok=False,
            stdout=buffer.getvalue(),
            error=f"{type(exc).__name__}: {exc}",
        )

    if "result" not in namespace:
        return ExecResult(
            ok=False,
            stdout=buffer.getvalue(),
            error="the snippet did not assign to `result`",
        )

    return ExecResult(
        ok=True,
        value=namespace["result"],
        result_repr=_render(namespace["result"], max_rows=max_rows),
        stdout=buffer.getvalue(),
        warning=warning,
    )


MAX_REPR_CHARS = 2000


def _render(value: Any, max_rows: int = 20) -> str:
    """Turn a result into short, readable text.

    Truncation is always *announced*. A model that gets 20 of 425 rows and is
    not told so will confidently describe the other 405.
    """
    if isinstance(value, pd.DataFrame):
        text = value.head(max_rows).to_string()
        if len(value) > max_rows:
            text += f"\n... [{len(value)} rows total, showing {max_rows}]"
        return text

    if isinstance(value, pd.Series):
        text = value.head(max_rows).to_string()
        if len(value) > max_rows:
            text += f"\n... [{len(value)} values total, showing {max_rows}]"
        return text

    if isinstance(value, np.ndarray):
        flat = value.ravel()
        text = repr(flat[:max_rows].tolist())
        if flat.size > max_rows:
            text += f"\n... [ndarray shape {value.shape}, showing {max_rows} values]"
        return text

    if isinstance(value, np.generic):
        return str(value.item())

    if isinstance(value, (list, tuple, set, dict)) and len(value) > max_rows:
        items = list(value.items()) if isinstance(value, dict) else list(value)
        return (
            repr(items[:max_rows])
            + f"\n... [{type(value).__name__} of {len(value)} items, showing {max_rows}]"
        )

    text = repr(value)
    if len(text) > MAX_REPR_CHARS:
        text = text[:MAX_REPR_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text

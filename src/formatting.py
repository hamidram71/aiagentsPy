"""How numbers and missing keys are written for the model. One place, one rule.

Everything a tool returns is read by a model that copies figures verbatim into
the report, so the rendering *is* the number as far as the report is
concerned.
"""

from __future__ import annotations

import math
from typing import Any

BLANK = "(blank)"


def fmt_number(value: Any) -> str:
    """Two decimals for ordinary amounts; significant figures below 1.

    `f"{v:,.2f}"` everywhere was the old rule, and it erased small rates: a
    0.4% defect rate and a 0.1% one both printed as 0.00 -- a 4x difference
    reported as "no defects anywhere". Below 1, four significant figures
    keep the difference visible (0.004 vs 0.001, 0.0446 vs 0.033).
    """
    if value is None:
        return BLANK
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (hasattr(value, "dtype") and "int" in str(value.dtype)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return BLANK
    if number != 0 and abs(number) < 1:
        return f"{number:.4g}"
    return f"{number:,.2f}"


def fmt_key(value: Any) -> str:
    """A group key as text; a missing key is named, not printed as `nan`."""
    try:
        if value is None or (not isinstance(value, str) and math.isnan(value)):
            return BLANK
    except TypeError:
        pass
    return str(value)

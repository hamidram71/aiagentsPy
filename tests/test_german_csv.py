"""CSV files as German Excel writes them.

Semicolons between columns, decimal commas, day-first dates with dots and --
for "CSV (Trennzeichen-getrennt)" -- the Windows cp1252 encoding. Before
0.14 such a file loaded as ONE column called "Datum;Region;Umsatz", and the
agent analysed nothing. Colleagues at a German company will mostly have
exactly these files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dataset import load_df, read_csv_file
from src.tools.analysis import group_aggregate

GERMAN = (
    "Auftrag;Datum;Region;Umsatz\n"
    "A-1;01.02.2025;Süd;1234,56\n"
    "A-2;13.02.2025;Nord;99,5\n"
    "A-3;28.02.2025;Süd;10,00\n"
    "A-4;05.03.2025;Nord;0,25\n"
)


@pytest.fixture(params=["cp1252", "utf-8", "utf-8-sig"])
def german_file(request, tmp_path):
    path = tmp_path / f"umsatz_{request.param}.csv"
    path.write_bytes(GERMAN.encode(request.param))
    load_df.cache_clear()
    yield path
    load_df.cache_clear()


def test_columns_numbers_and_umlauts_come_through(german_file) -> None:
    df = read_csv_file(german_file)
    assert list(df.columns) == ["Auftrag", "Datum", "Region", "Umsatz"]
    assert df["Umsatz"].tolist() == [1234.56, 99.5, 10.0, 0.25]
    assert set(df["Region"]) == {"Süd", "Nord"}


def test_dates_are_read_day_first(german_file) -> None:
    df = load_df(str(german_file))
    assert pd.api.types.is_datetime64_any_dtype(df["Datum"])
    # 01.02.2025 is the first of FEBRUARY, not January 2nd.
    assert df["Datum"].iloc[0] == pd.Timestamp("2025-02-01")
    assert df["Datum"].iloc[1] == pd.Timestamp("2025-02-13")


def test_the_agent_tools_see_the_real_columns(german_file) -> None:
    out = group_aggregate.invoke({
        "group_by": ["Region"], "value_column": "Umsatz", "agg": "sum",
        "state": {"csv_path": str(german_file)},
    })
    assert "Region=Süd -> 1,244.56" in out  # 1234.56 + 10.00


def test_an_ordinary_csv_is_unchanged() -> None:
    df = read_csv_file("data/sales.csv")
    assert df.shape == (425, 12)


def test_us_dates_with_slashes_are_not_turned_day_first(tmp_path) -> None:
    """Only the dotted German form means day first. 02/13/2025 stays US."""
    path = tmp_path / "us.csv"
    path.write_text("order_date,amount\n02/13/2025,1\n03/01/2025,2\n", encoding="utf-8")
    load_df.cache_clear()
    df = load_df(str(path))
    assert df["order_date"].iloc[1] == pd.Timestamp("2025-03-01")
    load_df.cache_clear()

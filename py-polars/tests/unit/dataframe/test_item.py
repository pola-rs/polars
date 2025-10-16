from __future__ import annotations

import pytest

import polars as pl


def test_df_item() -> None:
    df = pl.DataFrame({"a": [1]})
    assert df.item() == 1


def test_df_item_empty() -> None:
    df = pl.DataFrame()
    with pytest.raises(ValueError, match=r".* frame has shape \(0, 0\)"):
        df.item()


def test_df_item_incorrect_shape_rows() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match=r".* frame has shape \(2, 1\)"):
        df.item()


def test_df_item_incorrect_shape_columns() -> None:
    df = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match=r".* frame has shape \(1, 2\)"):
        df.item()


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [
        (0, 0, 1),
        (1, "a", 2),
        (-1, 1, 6),
        (-2, "b", 5),
    ],
)
def test_df_item_with_indices(
    row: int, col: int | str, expected: int, df: pl.DataFrame
) -> None:
    assert df.item(row, col) == expected


def test_df_item_with_single_index(df: pl.DataFrame) -> None:
    with pytest.raises(ValueError):
        df.item(0)
    with pytest.raises(ValueError):
        df.item(column="b")
    with pytest.raises(ValueError):
        df.item(None, 0)


@pytest.mark.parametrize(
    ("row", "col"), [(0, 10), (10, 0), (10, 10), (-10, 0), (-10, 10)]
)
def test_df_item_out_of_bounds(row: int, col: int, df: pl.DataFrame) -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        df.item(row, col)

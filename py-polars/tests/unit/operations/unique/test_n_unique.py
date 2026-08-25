from __future__ import annotations

import numpy as np
import pytest

import polars as pl
import polars.selectors as cs


def test_n_unique() -> None:
    s = pl.Series("s", [11, 11, 11, 22, 22, 33, None, None, None])
    assert s.n_unique() == 4


def test_n_unique_subsets() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 3, 4, 5],
            "b": [0.5, 0.5, 1.0, 2.0, 3.0, 3.0],
            "c": [True, True, True, False, True, True],
        }
    )
    # omitting 'subset' counts unique rows
    assert df.n_unique() == 5

    # providing it counts unique col/expr subsets
    assert df.n_unique(subset=["b", "c"]) == 4
    assert df.n_unique(subset=pl.col("c")) == 2
    assert (
        df.n_unique(subset=[(pl.col("a") // 2), (pl.col("c") | (pl.col("b") >= 2))])
        == 3
    )


def test_n_unique_null() -> None:
    assert pl.Series([]).n_unique() == 0
    assert pl.Series([None]).n_unique() == 1
    assert pl.Series([None, None]).n_unique() == 1


@pytest.mark.parametrize(
    ("input", "output"),
    [
        ([], 0),
        (["a", "b", "b", "c"], 3),
        (["a", "b", "b", None], 3),
    ],
)
def test_n_unique_categorical(input: list[str | None], output: int) -> None:
    assert pl.Series(input, dtype=pl.Categorical).n_unique() == output


def test_n_unique_list_of_struct_20341() -> None:
    df = pl.DataFrame(
        {
            "a": [
                [{"a": 1, "b": 2}, {"a": 10, "b": 20}],
                [{"a": 1, "b": 2}, {"a": 10, "b": 20}],
                [{"a": 3, "b": 4}],
            ]
        }
    )
    assert df.select("a").n_unique() == 2
    assert df["a"].n_unique() == 2


def test_n_unique_array() -> None:
    df = pl.DataFrame(
        {
            "arr": [
                np.array([1, 2]),
                np.array([2, 3]),
                np.array([3, 4]),
                np.array([3, 4]),
            ],
        }
    )
    assert df["arr"].dtype == pl.Array
    assert df.select(pl.col("arr")).n_unique() == 3
    assert df.select(pl.col("arr").n_unique()).item() == 3


def test_n_unique_multi_column_expression_gh28903() -> None:
    # gh-28903: n_unique() was ignoring additional columns when subset is a
    # multi-column expression or selector.
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "A": [1, 2, 3, 4, 1, 2],
            "B": [1, 2, 3, 1, 1, 1],
        }
    )
    # A & B together have 5 unique rows: (1,1),(2,2),(3,3),(4,1),(1,1),(2,1) -> 5
    assert df.n_unique(subset=pl.col("A", "B")) == 5
    assert df.n_unique(subset=cs.by_name("B", "A")) == 5
    assert df.n_unique(subset=pl.exclude("id")) == 5
    assert df.n_unique(subset=[cs.by_name("A", "B")]) == 5
    assert df.n_unique(subset=[pl.col("B", "id")]) == 6
    # single-column and list paths must still work
    assert df.n_unique(subset="A") == 4
    assert df.n_unique(subset=["A", "B"]) == 5

from __future__ import annotations

import pyarrow as pa

import polars as pl


def test_explode_empty_df_3402() -> None:
    df = pl.DataFrame({"a": pa.array([], type=pa.large_list(pa.int32()))})
    assert df.explode("a").dtypes == [pl.Int32]


def test_explode_empty_df_3460() -> None:
    df = pl.DataFrame({"a": pa.array([[]], type=pa.large_list(pa.int32()))})
    assert df.explode("a").dtypes == [pl.Int32]


def test_explode_empty_df_3902() -> None:
    df = pl.DataFrame(
        {
            "first": [1, 2, 3, 4, 5],
            "second": [["a"], [], ["b", "c"], [], ["d", "f", "g"]],
        }
    )
    expected = pl.DataFrame(
        {
            "first": [1, 2, 3, 3, 4, 5, 5, 5],
            "second": ["a", None, "b", "c", None, "d", "f", "g"],
        }
    )
    assert df.explode("second").frame_equal(expected)


def test_explode_empty_list_4003() -> None:
    df = pl.DataFrame(
        [
            {"id": 1, "nested": []},
            {"id": 2, "nested": [1]},
            {"id": 3, "nested": [2]},
        ]
    )
    assert df.explode("nested").to_dict(False) == {
        "id": [1, 2, 3],
        "nested": [None, 1, 2],
    }

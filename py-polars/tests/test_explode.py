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


def test_explode_empty_list_4107() -> None:
    df = pl.DataFrame({"b": [[1], [2], []] * 2}).with_row_count()

    pl.testing.assert_frame_equal(
        df.explode(["b"]), df.explode(["b"]).drop("row_nr").with_row_count()
    )


def explode_correct_for_slice() -> None:
    df = pl.DataFrame({"b": [[1, 1], [2, 2], [3, 3], [4, 4]]})
    assert df.slice(2, 2).explode(["b"])["b"].to_list() == [3, 3, 4, 4]

    df = (
        (
            pl.DataFrame({"group": pl.arange(0, 5, eager=True)}).join(
                pl.DataFrame(
                    {
                        "b": [[1, 2, 3], [2, 3], [4], [1, 2, 3], [0]],
                    }
                ),
                how="cross",
            )
        )
        .sort("group")
        .with_row_count()
    )
    expected = pl.DataFrame(
        {
            "row_nr": [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 8, 8, 8, 9],
            "group": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "b": [1, 2, 3, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 2, 3, 4, 1, 2, 3, 0],
        }
    )
    assert df.slice(0, 10).explode(["b"]).frame_equal(expected)

from __future__ import annotations

import pyarrow as pa

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_explode_string() -> None:
    df = pl.Series("a", ["Hello", "World"])
    result = df.to_frame().select(pl.col("a").str.explode()).to_series()

    expected = pl.Series("a", ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"])
    assert_series_equal(result, expected)


def test_groupby_flatten_list() -> None:
    df = pl.DataFrame({"group": ["a", "b", "b"], "values": [[1, 2], [2, 3], [4]]})
    result = df.groupby("group", maintain_order=True).agg(pl.col("values").flatten())

    expected = pl.DataFrame({"group": ["a", "b"], "values": [[1, 2], [2, 3, 4]]})
    assert_frame_equal(result, expected)


def test_groupby_flatten_string() -> None:
    df = pl.DataFrame({"group": ["a", "b", "b"], "values": ["foo", "bar", "baz"]})
    result = df.groupby("group", maintain_order=True).agg(pl.col("values").flatten())

    expected = pl.DataFrame(
        {
            "group": ["a", "b"],
            "values": [["f", "o", "o"], ["b", "a", "r", "b", "a", "z"]],
        }
    )
    assert_frame_equal(result, expected)


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
    assert_frame_equal(df.explode("second"), expected)


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

    assert_frame_equal(
        df.explode(["b"]), df.explode(["b"]).drop("row_nr").with_row_count()
    )


def test_explode_correct_for_slice() -> None:
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
        },
        schema_overrides={"row_nr": pl.UInt32},
    )
    assert_frame_equal(df.slice(0, 10).explode(["b"]), expected)


def test_sliced_null_explode() -> None:
    s = pl.Series("", [[1], [2], [3], [4], [], [6]])
    assert s.slice(2, 4).arr.explode().to_list() == [3, 4, None, 6]
    assert s.slice(2, 2).arr.explode().to_list() == [3, 4]
    assert pl.Series("", [[1], [2], None, [4], [], [6]]).slice(
        2, 4
    ).arr.explode().to_list() == [None, 4, None, 6]

    s = pl.Series("", [["a"], ["b"], ["c"], ["d"], [], ["e"]])
    assert s.slice(2, 4).arr.explode().to_list() == ["c", "d", None, "e"]
    assert s.slice(2, 2).arr.explode().to_list() == ["c", "d"]
    assert pl.Series("", [["a"], ["b"], None, ["d"], [], ["e"]]).slice(
        2, 4
    ).arr.explode().to_list() == [None, "d", None, "e"]

    s = pl.Series("", [[False], [False], [True], [False], [], [True]])
    assert s.slice(2, 2).arr.explode().to_list() == [True, False]
    assert s.slice(2, 4).arr.explode().to_list() == [True, False, None, True]


def test_utf8_explode() -> None:
    assert pl.Series(["foobar", None]).str.explode().to_list() == [
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
        None,
    ]
    assert pl.Series([None, "foo", "bar"]).str.explode().to_list() == [
        None,
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
    ]
    assert pl.Series([None, "foo", "bar", None, "ham"]).str.explode().to_list() == [
        None,
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
        None,
        "h",
        "a",
        "m",
    ]
    assert pl.Series(["foo", "bar", "ham"]).str.explode().to_list() == [
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
        "h",
        "a",
        "m",
    ]
    assert pl.Series(["", None, "foo", "bar"]).str.explode().to_list() == [
        "",
        None,
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
    ]
    assert pl.Series(["", "foo", "bar"]).str.explode().to_list() == [
        "",
        "f",
        "o",
        "o",
        "b",
        "a",
        "r",
    ]


def test_explode_in_agg_context() -> None:
    df = pl.DataFrame(
        {"idxs": [[0], [1], [0, 2]], "array": [[0.0, 3.5], [4.6, 0.0], [0.0, 7.8, 0.0]]}
    )

    assert (
        df.with_row_count("row_nr")
        .explode("idxs")
        .groupby("row_nr")
        .agg(pl.col("array").flatten())
    ).to_dict(False) == {
        "row_nr": [0, 1, 2],
        "array": [[0.0, 3.5], [4.6, 0.0], [0.0, 7.8, 0.0, 0.0, 7.8, 0.0]],
    }


def test_explode_inner_lists_3985() -> None:
    df = pl.DataFrame(
        data={"id": [1, 1, 1], "categories": [["a"], ["b"], ["a", "c"]]}
    ).lazy()

    assert (
        df.groupby("id")
        .agg(pl.col("categories"))
        .with_columns(pl.col("categories").arr.eval(pl.element().arr.explode()))
    ).collect().to_dict(False) == {"id": [1], "categories": [["a", "b", "a", "c"]]}

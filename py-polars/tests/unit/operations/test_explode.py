from __future__ import annotations

import pyarrow as pa
import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal, assert_series_equal


def test_explode_string() -> None:
    df = pl.Series("a", ["Hello", "World"])
    result = df.to_frame().select(pl.col("a").str.explode()).to_series()

    expected = pl.Series("a", ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"])
    assert_series_equal(result, expected)


def test_explode_multiple() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]})

    expected = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    assert_frame_equal(df.explode(cs.all()), expected)
    assert_frame_equal(df.explode(["a", "b"]), expected)
    assert_frame_equal(df.explode("a", "b"), expected)


def test_group_by_flatten_list() -> None:
    df = pl.DataFrame({"group": ["a", "b", "b"], "values": [[1, 2], [2, 3], [4]]})
    result = df.group_by("group", maintain_order=True).agg(pl.col("values").flatten())

    expected = pl.DataFrame({"group": ["a", "b"], "values": [[1, 2], [2, 3, 4]]})
    assert_frame_equal(result, expected)


def test_group_by_flatten_string() -> None:
    df = pl.DataFrame({"group": ["a", "b", "b"], "values": ["foo", "bar", "baz"]})
    result = df.group_by("group", maintain_order=True).agg(
        pl.col("values").str.explode()
    )

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
    assert df.explode("nested").to_dict(as_series=False) == {
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
    assert s.slice(2, 4).list.explode().to_list() == [3, 4, None, 6]
    assert s.slice(2, 2).list.explode().to_list() == [3, 4]
    assert pl.Series("", [[1], [2], None, [4], [], [6]]).slice(
        2, 4
    ).list.explode().to_list() == [None, 4, None, 6]

    s = pl.Series("", [["a"], ["b"], ["c"], ["d"], [], ["e"]])
    assert s.slice(2, 4).list.explode().to_list() == ["c", "d", None, "e"]
    assert s.slice(2, 2).list.explode().to_list() == ["c", "d"]
    assert pl.Series("", [["a"], ["b"], None, ["d"], [], ["e"]]).slice(
        2, 4
    ).list.explode().to_list() == [None, "d", None, "e"]

    s = pl.Series("", [[False], [False], [True], [False], [], [True]])
    assert s.slice(2, 2).list.explode().to_list() == [True, False]
    assert s.slice(2, 4).list.explode().to_list() == [True, False, None, True]


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
        .group_by("row_nr")
        .agg(pl.col("array").flatten())
    ).to_dict(as_series=False) == {
        "row_nr": [0, 1, 2],
        "array": [[0.0, 3.5], [4.6, 0.0], [0.0, 7.8, 0.0, 0.0, 7.8, 0.0]],
    }


def test_explode_inner_lists_3985() -> None:
    df = pl.DataFrame(
        data={"id": [1, 1, 1], "categories": [["a"], ["b"], ["a", "c"]]}
    ).lazy()

    assert (
        df.group_by("id")
        .agg(pl.col("categories"))
        .with_columns(pl.col("categories").list.eval(pl.element().list.explode()))
    ).collect().to_dict(as_series=False) == {
        "id": [1],
        "categories": [["a", "b", "a", "c"]],
    }


def test_list_struct_explode_6905() -> None:
    assert pl.DataFrame(
        {
            "group": [
                [],
                [
                    {"params": [1]},
                    {"params": []},
                ],
            ]
        },
        schema={"group": pl.List(pl.Struct([pl.Field("params", pl.List(pl.Int32))]))},
    )["group"].list.explode().to_list() == [
        {"params": None},
        {"params": [1]},
        {"params": []},
    ]


def test_explode_binary() -> None:
    assert pl.Series([[1, 2], [3]]).cast(
        pl.List(pl.Binary)
    ).list.explode().to_list() == [
        b"1",
        b"2",
        b"3",
    ]


def test_explode_null_list() -> None:
    assert pl.Series([["a"], None], dtype=pl.List(pl.Utf8))[
        1:2
    ].list.min().to_list() == [None]


def test_explode_invalid_element_count() -> None:
    df = pl.DataFrame(
        {
            "col1": [["X", "Y", "Z"], ["F", "G"], ["P"]],
            "col2": [["A", "B", "C"], ["C"], ["D", "E"]],
        }
    ).with_row_count()
    with pytest.raises(
        pl.ShapeError, match=r"exploded columns must have matching element counts"
    ):
        df.explode(["col1", "col2"])


def test_logical_explode() -> None:
    out = (
        pl.DataFrame(
            {"cats": ["Value1", "Value2", "Value1"]},
            schema_overrides={"cats": pl.Categorical},
        )
        .group_by(1)
        .agg(pl.struct("cats"))
        .explode("cats")
        .unnest("cats")
    )
    assert out["cats"].dtype == pl.Categorical
    assert out["cats"].to_list() == ["Value1", "Value2", "Value1"]


def test_explode_inner_null() -> None:
    expected = pl.DataFrame({"A": [None, None]}, schema={"A": pl.Null})
    out = pl.DataFrame({"A": [[], []]}, schema={"A": pl.List(pl.Null)}).explode("A")
    assert_frame_equal(out, expected)


def test_explode_array() -> None:
    df = pl.LazyFrame(
        {"a": [[1, 2], [2, 3]], "b": [1, 2]},
        schema_overrides={"a": pl.Array(pl.Int64, 2)},
    )
    expected = pl.DataFrame({"a": [1, 2, 2, 3], "b": [1, 1, 2, 2]})
    for ex in ("a", ~cs.integer()):
        out = df.explode(ex).collect()  # type: ignore[arg-type]
        assert_frame_equal(out, expected)


def test_utf8_list_agg_explode() -> None:
    df = pl.DataFrame({"a": [[None], ["b"]]})

    df = df.select(
        pl.col("a").list.eval(pl.element().filter(pl.element().is_not_null()))
    )
    assert not df["a"].flags["FAST_EXPLODE"]

    df2 = pl.DataFrame({"a": [[], ["b"]]})

    assert_frame_equal(df, df2)
    assert_frame_equal(df.explode("a"), df2.explode("a"))


def test_explode_null_struct() -> None:
    df = [
        {"col1": None},
        {
            "col1": [
                {"field1": None, "field2": None, "field3": None},
                {"field1": None, "field2": "some", "field3": "value"},
            ]
        },
    ]

    assert pl.DataFrame(df).explode("col1").to_dict(as_series=False) == {
        "col1": [
            {"field1": None, "field2": None, "field3": None},
            {"field1": None, "field2": None, "field3": None},
            {"field1": None, "field2": "some", "field3": "value"},
        ]
    }

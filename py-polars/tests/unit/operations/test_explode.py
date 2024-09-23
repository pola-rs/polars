from __future__ import annotations

import pyarrow as pa
import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


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
    df = pl.DataFrame({"b": [[1], [2], []] * 2}).with_row_index()

    assert_frame_equal(
        df.explode(["b"]), df.explode(["b"]).drop("index").with_row_index()
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
        .with_row_index()
    )
    expected = pl.DataFrame(
        {
            "index": [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 8, 8, 8, 9],
            "group": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "b": [1, 2, 3, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 2, 3, 4, 1, 2, 3, 0],
        },
        schema_overrides={"index": pl.UInt32},
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


def test_explode_in_agg_context() -> None:
    df = pl.DataFrame(
        {"idxs": [[0], [1], [0, 2]], "array": [[0.0, 3.5], [4.6, 0.0], [0.0, 7.8, 0.0]]}
    )

    assert (
        df.with_row_index()
        .explode("idxs")
        .group_by("index")
        .agg(pl.col("array").flatten())
    ).to_dict(as_series=False) == {
        "index": [0, 1, 2],
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
        None,
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
    assert pl.Series([["a"], None], dtype=pl.List(pl.String))[
        1:2
    ].list.min().to_list() == [None]


def test_explode_invalid_element_count() -> None:
    df = pl.DataFrame(
        {
            "col1": [["X", "Y", "Z"], ["F", "G"], ["P"]],
            "col2": [["A", "B", "C"], ["C"], ["D", "E"]],
        }
    ).with_row_index()
    with pytest.raises(
        ShapeError, match=r"exploded columns must have matching element counts"
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
        out = df.explode(ex).collect()
        assert_frame_equal(out, expected)


def test_string_list_agg_explode() -> None:
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
            None,
            {"field1": None, "field2": None, "field3": None},
            {"field1": None, "field2": "some", "field3": "value"},
        ]
    }


def test_df_explode_with_array() -> None:
    df = pl.DataFrame(
        {
            "arr": [["a", "b"], ["c", None], None, ["d", "e"]],
            "list": [[1, 2], [3], [4, None], None],
            "val": ["x", "y", "z", "q"],
        },
        schema={
            "arr": pl.Array(pl.String, 2),
            "list": pl.List(pl.Int64),
            "val": pl.String,
        },
    )

    expected_by_arr = pl.DataFrame(
        {
            "arr": ["a", "b", "c", None, None, "d", "e"],
            "list": [[1, 2], [1, 2], [3], [3], [4, None], None, None],
            "val": ["x", "x", "y", "y", "z", "q", "q"],
        }
    )
    assert_frame_equal(df.explode(pl.col("arr")), expected_by_arr)

    expected_by_list = pl.DataFrame(
        {
            "arr": [["a", "b"], ["a", "b"], ["c", None], None, None, ["d", "e"]],
            "list": [1, 2, 3, 4, None, None],
            "val": ["x", "x", "y", "z", "z", "q"],
        },
        schema={
            "arr": pl.Array(pl.String, 2),
            "list": pl.Int64,
            "val": pl.String,
        },
    )
    assert_frame_equal(df.explode(pl.col("list")), expected_by_list)

    df = pl.DataFrame(
        {
            "arr": [["a", "b"], ["c", None], None, ["d", "e"]],
            "list": [[1, 2], [3, 4], None, [5, None]],
            "val": [None, 1, 2, None],
        },
        schema={
            "arr": pl.Array(pl.String, 2),
            "list": pl.List(pl.Int64),
            "val": pl.Int64,
        },
    )
    expected_by_arr_and_list = pl.DataFrame(
        {
            "arr": ["a", "b", "c", None, None, "d", "e"],
            "list": [1, 2, 3, 4, None, 5, None],
            "val": [None, None, 1, 1, 2, None, None],
        },
        schema={
            "arr": pl.String,
            "list": pl.Int64,
            "val": pl.Int64,
        },
    )
    assert_frame_equal(df.explode("arr", "list"), expected_by_arr_and_list)


def test_explode_nullable_list() -> None:
    df = pl.DataFrame({"layout1": [None, [1, 2]], "b": [False, True]}).with_columns(
        layout2=pl.when(pl.col("b")).then([1, 2]),
    )

    explode_df = df.explode("layout1", "layout2")
    expected_df = pl.DataFrame(
        {
            "layout1": [None, 1, 2],
            "b": [False, True, True],
            "layout2": [None, 1, 2],
        }
    )
    assert_frame_equal(explode_df, expected_df)

    explode_expr = df.select(
        pl.col("layout1").explode(),
        pl.col("layout2").explode(),
    )
    expected_df = pl.DataFrame(
        {
            "layout1": [None, 1, 2],
            "layout2": [None, 1, 2],
        }
    )
    assert_frame_equal(explode_expr, expected_df)


def test_group_by_flatten_string() -> None:
    df = pl.DataFrame({"group": ["a", "b", "b"], "values": ["foo", "bar", "baz"]})

    result = df.group_by("group", maintain_order=True).agg(
        pl.col("values").str.split("").explode()
    )

    expected = pl.DataFrame(
        {
            "group": ["a", "b"],
            "values": [["f", "o", "o"], ["b", "a", "r", "b", "a", "z"]],
        }
    )
    assert_frame_equal(result, expected)


def test_fast_explode_merge_right_16923() -> None:
    df = pl.concat(
        [
            pl.DataFrame({"foo": [["a", "b"], ["c"]]}),
            pl.DataFrame({"foo": [None]}, schema={"foo": pl.List(pl.Utf8)}),
        ],
        how="diagonal",
        rechunk=True,
    ).explode("foo")

    assert len(df) == 4


def test_fast_explode_merge_left_16923() -> None:
    df = pl.concat(
        [
            pl.DataFrame({"foo": [None]}, schema={"foo": pl.List(pl.Utf8)}),
            pl.DataFrame({"foo": [["a", "b"], ["c"]]}),
        ],
        how="diagonal",
        rechunk=True,
    ).explode("foo")

    assert len(df) == 4


@pytest.mark.parametrize(
    ("values", "exploded"),
    [
        (["foobar", None], ["f", "o", "o", "b", "a", "r", None]),
        ([None, "foo", "bar"], [None, "f", "o", "o", "b", "a", "r"]),
        (
            [None, "foo", "bar", None, "ham"],
            [None, "f", "o", "o", "b", "a", "r", None, "h", "a", "m"],
        ),
        (["foo", "bar", "ham"], ["f", "o", "o", "b", "a", "r", "h", "a", "m"]),
        (["", None, "foo", "bar"], ["", None, "f", "o", "o", "b", "a", "r"]),
        (["", "foo", "bar"], ["", "f", "o", "o", "b", "a", "r"]),
    ],
)
def test_series_str_explode_deprecated(
    values: list[str | None], exploded: list[str | None]
) -> None:
    with pytest.deprecated_call():
        result = pl.Series(values).str.explode()
    assert result.to_list() == exploded


def test_expr_str_explode_deprecated() -> None:
    df = pl.Series("a", ["Hello", "World"])
    with pytest.deprecated_call():
        result = df.to_frame().select(pl.col("a").str.explode()).to_series()

    expected = pl.Series("a", ["H", "e", "l", "l", "o", "W", "o", "r", "l", "d"])
    assert_series_equal(result, expected)


def test_undefined_col_15852() -> None:
    lf = pl.LazyFrame({"foo": [1]})

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        lf.explode("bar").join(lf, on="foo").collect()


def test_explode_17648() -> None:
    df = pl.DataFrame({"a": [[1, 3], [2, 6, 7], [3, 9, 2], [4], [5, 1, 2, 3, 4]]})
    assert (
        df.slice(1, 2)
        .with_columns(pl.int_ranges(pl.col("a").list.len()).alias("count"))
        .explode("a", "count")
    ).to_dict(as_series=False) == {"a": [2, 6, 7, 3, 9, 2], "count": [0, 1, 2, 0, 1, 2]}


def test_explode_struct_nulls() -> None:
    df = pl.DataFrame({"A": [[{"B": 1}], [None], []]})
    assert df.explode("A").to_dict(as_series=False) == {"A": [{"B": 1}, None, None]}

from __future__ import annotations

import datetime
from typing import Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_arr_min_max() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(pl.Int64, 2))
    assert s.arr.max().to_list() == [2, 4]
    assert s.arr.min().to_list() == [1, 3]

    s_with_null = pl.Series("a", [[None, 2], None, [3, 4]], dtype=pl.Array(pl.Int64, 2))
    assert s_with_null.arr.max().to_list() == [2, None, 4]
    assert s_with_null.arr.min().to_list() == [2, None, 3]


def test_array_min_max_dtype_12123() -> None:
    df = pl.LazyFrame(
        [pl.Series("a", [[1.0, 3.0], [2.0, 5.0]]), pl.Series("b", [1.0, 2.0])],
        schema_overrides={
            "a": pl.Array(pl.Float64, 2),
        },
    )

    df = df.with_columns(
        max=pl.col("a").arr.max().alias("max"),
        min=pl.col("a").arr.min().alias("min"),
    )

    assert df.collect_schema() == {
        "a": pl.Array(pl.Float64, 2),
        "b": pl.Float64,
        "max": pl.Float64,
        "min": pl.Float64,
    }

    out = df.select(pl.col("max") * pl.col("b"), pl.col("min") * pl.col("b")).collect()

    assert_frame_equal(out, pl.DataFrame({"max": [3.0, 10.0], "min": [1.0, 4.0]}))


@pytest.mark.parametrize(
    ("data", "expected_sum", "dtype"),
    [
        ([[1, 2], [4, 3]], [3, 7], pl.Int64),
        ([[1, None], [None, 3], [None, None]], [1, 3, 0], pl.Int64),
        ([[1.0, 2.0], [4.0, 3.0]], [3.0, 7.0], pl.Float32),
        ([[1.0, None], [None, 3.0], [None, None]], [1.0, 3.0, 0], pl.Float32),
        ([[True, False], [True, True], [False, False]], [1, 2, 0], pl.Boolean),
        ([[True, None], [None, False], [None, None]], [1, 0, 0], pl.Boolean),
    ],
)
def test_arr_sum(
    data: list[list[Any]], expected_sum: list[Any], dtype: pl.DataType
) -> None:
    s = pl.Series("a", data, dtype=pl.Array(dtype, 2))
    assert s.arr.sum().to_list() == expected_sum


def test_arr_unique() -> None:
    df = pl.DataFrame(
        {"a": pl.Series("a", [[1, 1], [4, 3]], dtype=pl.Array(pl.Int64, 2))}
    )

    out = df.select(pl.col("a").arr.unique(maintain_order=True))
    expected = pl.DataFrame({"a": [[1], [4, 3]]})
    assert_frame_equal(out, expected)


def test_array_any_all() -> None:
    s = pl.Series(
        [[True, True], [False, True], [False, False], [None, None], None],
        dtype=pl.Array(pl.Boolean, 2),
    )

    expected_any = pl.Series([True, True, False, False, None])
    assert_series_equal(s.arr.any(), expected_any)

    expected_all = pl.Series([True, False, False, True, None])
    assert_series_equal(s.arr.all(), expected_all)

    s = pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(pl.Int64, 2))
    with pytest.raises(ComputeError, match="expected boolean elements in array"):
        s.arr.any()
    with pytest.raises(ComputeError, match="expected boolean elements in array"):
        s.arr.all()


def test_array_sort() -> None:
    s = pl.Series([[2, None, 1], [1, 3, 2]], dtype=pl.Array(pl.UInt32, 3))

    desc = s.arr.sort(descending=True)
    expected = pl.Series([[None, 2, 1], [3, 2, 1]], dtype=pl.Array(pl.UInt32, 3))
    assert_series_equal(desc, expected)

    asc = s.arr.sort(descending=False)
    expected = pl.Series([[None, 1, 2], [1, 2, 3]], dtype=pl.Array(pl.UInt32, 3))
    assert_series_equal(asc, expected)

    # test nulls_last
    s = pl.Series([[None, 1, 2], [-1, None, 9]], dtype=pl.Array(pl.Int8, 3))
    assert_series_equal(
        s.arr.sort(nulls_last=True),
        pl.Series([[1, 2, None], [-1, 9, None]], dtype=pl.Array(pl.Int8, 3)),
    )
    assert_series_equal(
        s.arr.sort(nulls_last=False),
        pl.Series([[None, 1, 2], [None, -1, 9]], dtype=pl.Array(pl.Int8, 3)),
    )


def test_array_reverse() -> None:
    s = pl.Series([[2, None, 1], [1, None, 2]], dtype=pl.Array(pl.UInt32, 3))

    s = s.arr.reverse()
    expected = pl.Series([[1, None, 2], [2, None, 1]], dtype=pl.Array(pl.UInt32, 3))
    assert_series_equal(s, expected)


def test_array_arg_min_max() -> None:
    s = pl.Series("a", [[1, 2, 4], [3, 2, 1]], dtype=pl.Array(pl.UInt32, 3))
    expected = pl.Series("a", [0, 2], dtype=pl.UInt32)
    assert_series_equal(s.arr.arg_min(), expected)
    expected = pl.Series("a", [2, 0], dtype=pl.UInt32)
    assert_series_equal(s.arr.arg_max(), expected)


def test_array_get() -> None:
    s = pl.Series(
        "a",
        [[1, 2, 3, 4], [5, 6, None, None], [7, 8, 9, 10]],
        dtype=pl.Array(pl.Int64, 4),
    )

    # Test index literal.
    out = s.arr.get(1, null_on_oob=False)
    expected = pl.Series("a", [2, 6, 8], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # Null index literal.
    out_df = s.to_frame().select(pl.col.a.arr.get(pl.lit(None), null_on_oob=False))
    expected_df = pl.Series("a", [None, None, None], dtype=pl.Int64).to_frame()
    assert_frame_equal(out_df, expected_df)

    # Out-of-bounds index literal.
    with pytest.raises(ComputeError, match="get index is out of bounds"):
        out = s.arr.get(100, null_on_oob=False)

    # Negative index literal.
    out = s.arr.get(-2, null_on_oob=False)
    expected = pl.Series("a", [3, None, 9], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # Test index expr.
    with pytest.raises(ComputeError, match="get index is out of bounds"):
        out = s.arr.get(pl.Series([1, -2, 100]), null_on_oob=False)

    out = s.arr.get(pl.Series([1, -2, 0]), null_on_oob=False)
    expected = pl.Series("a", [2, None, 7], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # Test logical type.
    s = pl.Series(
        "a",
        [
            [datetime.date(1999, 1, 1), datetime.date(2000, 1, 1)],
            [datetime.date(2001, 10, 1), None],
            [None, None],
        ],
        dtype=pl.Array(pl.Date, 2),
    )
    with pytest.raises(ComputeError, match="get index is out of bounds"):
        out = s.arr.get(pl.Series([1, -2, 4]), null_on_oob=False)


def test_array_get_null_on_oob() -> None:
    s = pl.Series(
        "a",
        [[1, 2, 3, 4], [5, 6, None, None], [7, 8, 9, 10]],
        dtype=pl.Array(pl.Int64, 4),
    )

    # Test index literal.
    out = s.arr.get(1, null_on_oob=True)
    expected = pl.Series("a", [2, 6, 8], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # Null index literal.
    out_df = s.to_frame().select(pl.col.a.arr.get(pl.lit(None), null_on_oob=True))
    expected_df = pl.Series("a", [None, None, None], dtype=pl.Int64).to_frame()
    assert_frame_equal(out_df, expected_df)

    # Out-of-bounds index literal.
    out = s.arr.get(100, null_on_oob=True)
    expected = pl.Series("a", [None, None, None], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # Negative index literal.
    out = s.arr.get(-2, null_on_oob=True)
    expected = pl.Series("a", [3, None, 9], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # Test index expr.
    out = s.arr.get(pl.Series([1, -2, 100]), null_on_oob=True)
    expected = pl.Series("a", [2, None, None], dtype=pl.Int64)
    assert_series_equal(out, expected)

    # Test logical type.
    s = pl.Series(
        "a",
        [
            [datetime.date(1999, 1, 1), datetime.date(2000, 1, 1)],
            [datetime.date(2001, 10, 1), None],
            [None, None],
        ],
        dtype=pl.Array(pl.Date, 2),
    )
    out = s.arr.get(pl.Series([1, -2, 4]), null_on_oob=True)
    expected = pl.Series(
        "a",
        [datetime.date(2000, 1, 1), datetime.date(2001, 10, 1), None],
        dtype=pl.Date,
    )
    assert_series_equal(out, expected)


def test_arr_first_last() -> None:
    s = pl.Series(
        "a",
        [[1, 2, 3], [None, 5, 6], [None, None, None]],
        dtype=pl.Array(pl.Int64, 3),
    )

    first = s.arr.first()
    expected_first = pl.Series(
        "a",
        [1, None, None],
        dtype=pl.Int64,
    )
    assert_series_equal(first, expected_first)

    last = s.arr.last()
    expected_last = pl.Series(
        "a",
        [3, 6, None],
        dtype=pl.Int64,
    )
    assert_series_equal(last, expected_last)


@pytest.mark.parametrize(
    ("data", "set", "dtype"),
    [
        ([1, 2], [[1, 2], [3, 4]], pl.Int64),
        ([True, False], [[True, False], [True, True]], pl.Boolean),
        (["a", "b"], [["a", "b"], ["c", "d"]], pl.String),
        ([b"a", b"b"], [[b"a", b"b"], [b"c", b"d"]], pl.Binary),
        (
            [{"a": 1}, {"a": 2}],
            [[{"a": 1}, {"a": 2}], [{"b": 1}, {"a": 3}]],
            pl.Struct([pl.Field("a", pl.Int64)]),
        ),
    ],
)
def test_is_in_array(data: list[Any], set: list[list[Any]], dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        {"a": data, "arr": set},
        schema={"a": dtype, "arr": pl.Array(dtype, 2)},
    )
    out = df.select(is_in=pl.col("a").is_in(pl.col("arr"))).to_series()
    expected = pl.Series("is_in", [True, False])
    assert_series_equal(out, expected)


def test_array_join() -> None:
    df = pl.DataFrame(
        {
            "a": [["ab", "c", "d"], ["e", "f", "g"], [None, None, None], None],
            "separator": ["&", None, "*", "_"],
        },
        schema={
            "a": pl.Array(pl.String, 3),
            "separator": pl.String,
        },
    )
    out = df.select(pl.col("a").arr.join("-"))
    assert out.to_dict(as_series=False) == {"a": ["ab-c-d", "e-f-g", "", None]}
    out = df.select(pl.col("a").arr.join(pl.col("separator")))
    assert out.to_dict(as_series=False) == {"a": ["ab&c&d", None, "", None]}

    # test ignore_nulls argument
    df = pl.DataFrame(
        {
            "a": [
                ["a", None, "b", None],
                None,
                [None, None, None, None],
                ["c", "d", "e", "f"],
            ],
            "separator": ["-", "&", " ", "@"],
        },
        schema={
            "a": pl.Array(pl.String, 4),
            "separator": pl.String,
        },
    )
    # ignore nulls
    out = df.select(pl.col("a").arr.join("-", ignore_nulls=True))
    assert out.to_dict(as_series=False) == {"a": ["a-b", None, "", "c-d-e-f"]}
    out = df.select(pl.col("a").arr.join(pl.col("separator"), ignore_nulls=True))
    assert out.to_dict(as_series=False) == {"a": ["a-b", None, "", "c@d@e@f"]}
    # propagate nulls
    out = df.select(pl.col("a").arr.join("-", ignore_nulls=False))
    assert out.to_dict(as_series=False) == {"a": [None, None, None, "c-d-e-f"]}
    out = df.select(pl.col("a").arr.join(pl.col("separator"), ignore_nulls=False))
    assert out.to_dict(as_series=False) == {"a": [None, None, None, "c@d@e@f"]}


def test_array_explode() -> None:
    df = pl.DataFrame(
        {
            "str": [["a", "b"], ["c", None], None],
            "nested": [[[1, 2], [3]], [[], [4, None]], None],
            "logical": [
                [datetime.date(1998, 1, 1), datetime.date(2000, 10, 1)],
                [datetime.date(2024, 1, 1), None],
                None,
            ],
        },
        schema={
            "str": pl.Array(pl.String, 2),
            "nested": pl.Array(pl.List(pl.Int64), 2),
            "logical": pl.Array(pl.Date, 2),
        },
    )
    out = df.select(pl.all().arr.explode())
    expected = pl.DataFrame(
        {
            "str": ["a", "b", "c", None, None],
            "nested": [[1, 2], [3], [], [4, None], None],
            "logical": [
                datetime.date(1998, 1, 1),
                datetime.date(2000, 10, 1),
                datetime.date(2024, 1, 1),
                None,
                None,
            ],
        }
    )
    assert_frame_equal(out, expected)

    # test no-null fast path
    s = pl.Series(
        [
            [datetime.date(1998, 1, 1), datetime.date(1999, 1, 3)],
            [datetime.date(2000, 1, 1), datetime.date(2023, 10, 1)],
        ],
        dtype=pl.Array(pl.Date, 2),
    )
    out_s = s.arr.explode()
    expected_s = pl.Series(
        [
            datetime.date(1998, 1, 1),
            datetime.date(1999, 1, 3),
            datetime.date(2000, 1, 1),
            datetime.date(2023, 10, 1),
        ],
        dtype=pl.Date,
    )
    assert_series_equal(out_s, expected_s)


@pytest.mark.parametrize(
    ("arr", "data", "expected", "dtype"),
    [
        ([[1, 2], [3, None], None], 1, [1, 0, None], pl.Int64),
        ([[True, False], [True, None], None], True, [1, 1, None], pl.Boolean),
        ([["a", "b"], ["c", None], None], "a", [1, 0, None], pl.String),
        ([[b"a", b"b"], [b"c", None], None], b"a", [1, 0, None], pl.Binary),
    ],
)
def test_array_count_matches(
    arr: list[list[Any] | None], data: Any, expected: list[Any], dtype: pl.DataType
) -> None:
    df = pl.DataFrame({"arr": arr}, schema={"arr": pl.Array(dtype, 2)})
    out = df.select(count_matches=pl.col("arr").arr.count_matches(data))
    assert out.to_dict(as_series=False) == {"count_matches": expected}


def test_array_to_struct() -> None:
    df = pl.DataFrame(
        {"a": [[1, 2, 3], [4, 5, None]]}, schema={"a": pl.Array(pl.Int8, 3)}
    )
    assert df.select([pl.col("a").arr.to_struct()]).to_series().to_list() == [
        {"field_0": 1, "field_1": 2, "field_2": 3},
        {"field_0": 4, "field_1": 5, "field_2": None},
    ]

    df = pl.DataFrame(
        {"a": [[1, 2, None], [1, 2, 3]]}, schema={"a": pl.Array(pl.Int8, 3)}
    )
    assert df.select(
        pl.col("a").arr.to_struct(fields=lambda idx: f"col_name_{idx}")
    ).to_series().to_list() == [
        {"col_name_0": 1, "col_name_1": 2, "col_name_2": None},
        {"col_name_0": 1, "col_name_1": 2, "col_name_2": 3},
    ]

    assert df.lazy().select(pl.col("a").arr.to_struct()).unnest(
        "a"
    ).sum().collect().columns == ["field_0", "field_1", "field_2"]


def test_array_shift() -> None:
    df = pl.DataFrame(
        {"a": [[1, 2, 3], None, [4, 5, 6], [7, 8, 9]], "n": [None, 1, 1, -2]},
        schema={"a": pl.Array(pl.Int64, 3), "n": pl.Int64},
    )

    out = df.select(
        lit=pl.col("a").arr.shift(1), expr=pl.col("a").arr.shift(pl.col("n"))
    )
    expected = pl.DataFrame(
        {
            "lit": [[None, 1, 2], None, [None, 4, 5], [None, 7, 8]],
            "expr": [None, None, [None, 4, 5], [9, None, None]],
        },
        schema={"lit": pl.Array(pl.Int64, 3), "expr": pl.Array(pl.Int64, 3)},
    )
    assert_frame_equal(out, expected)


def test_array_n_unique() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 1, 2], [3, 3, 3], [None, None, None], None],
        },
        schema={"a": pl.Array(pl.Int64, 3)},
    )

    out = df.select(n_unique=pl.col("a").arr.n_unique())
    expected = pl.DataFrame(
        {"n_unique": [2, 1, 1, None]}, schema={"n_unique": pl.UInt32}
    )
    assert_frame_equal(out, expected)

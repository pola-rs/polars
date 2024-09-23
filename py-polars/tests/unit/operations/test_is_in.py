from __future__ import annotations

from datetime import date
from decimal import Decimal as D
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars import StringCache
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_struct_logical_is_in() -> None:
    df1 = pl.DataFrame(
        {
            "x": pl.date_range(date(2022, 1, 1), date(2022, 1, 7), eager=True),
            "y": [0, 4, 6, 2, 3, 4, 5],
        }
    )
    df2 = pl.DataFrame(
        {
            "x": pl.date_range(date(2022, 1, 3), date(2022, 1, 9), eager=True),
            "y": [6, 2, 3, 4, 5, 0, 1],
        }
    )

    s1 = df1.select(pl.struct(["x", "y"])).to_series()
    s2 = df2.select(pl.struct(["x", "y"])).to_series()

    assert s1.is_in(s2).to_list() == [False, False, True, True, True, True, True]


def test_is_in_bool() -> None:
    vals = [True, None]
    df = pl.DataFrame({"A": [True, False, None]})
    assert df.select(pl.col("A").is_in(vals)).to_dict(as_series=False) == {
        "A": [True, False, None]
    }


def test_is_in_bool_11216() -> None:
    s = pl.Series([False]).is_in([False, None])
    expected = pl.Series([True])
    assert_series_equal(s, expected)


def test_is_in_empty_list_4559() -> None:
    assert pl.Series(["a"]).is_in([]).to_list() == [False]


def test_is_in_empty_list_4639() -> None:
    df = pl.DataFrame({"a": [1, None]})
    empty_list: list[int] = []

    result = df.with_columns([pl.col("a").is_in(empty_list).alias("a_in_list")])
    expected = pl.DataFrame({"a": [1, None], "a_in_list": [False, None]})
    assert_frame_equal(result, expected)


def test_is_in_struct() -> None:
    df = pl.DataFrame(
        {
            "struct_elem": [{"a": 1, "b": 11}, {"a": 1, "b": 90}],
            "struct_list": [
                [{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}],
                [{"a": 3, "b": 3}],
            ],
        }
    )

    assert df.filter(pl.col("struct_elem").is_in("struct_list")).to_dict(
        as_series=False
    ) == {
        "struct_elem": [{"a": 1, "b": 11}],
        "struct_list": [[{"a": 1, "b": 11}, {"a": 2, "b": 12}, {"a": 3, "b": 13}]],
    }


def test_is_in_null_prop() -> None:
    assert pl.Series([None], dtype=pl.Float32).is_in(pl.Series([42])).item() is None
    assert pl.Series([{"a": None}, None], dtype=pl.Struct({"a": pl.Float32})).is_in(
        pl.Series([{"a": 42}])
    ).to_list() == [False, None]

    assert pl.Series([{"a": None}, None], dtype=pl.Struct({"a": pl.Boolean})).is_in(
        pl.Series([{"a": 42}])
    ).to_list() == [False, None]


def test_is_in_9070() -> None:
    assert not pl.Series([1]).is_in(pl.Series([1.99])).item()


def test_is_in_float_list_10764() -> None:
    df = pl.DataFrame(
        {
            "lst": [[1.0, 2.0, 3.0, 4.0, 5.0], [3.14, 5.28]],
            "n": [3.0, 2.0],
        }
    )
    assert df.select(pl.col("n").is_in("lst").alias("is_in")).to_dict(
        as_series=False
    ) == {"is_in": [True, False]}


def test_is_in_df() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.select(pl.col("a").is_in([1, 2]))["a"].to_list() == [True, True, False]


def test_is_in_series() -> None:
    s = pl.Series(["a", "b", "c"])

    out = s.is_in(["a", "b"])
    assert out.to_list() == [True, True, False]

    # Check if empty list is converted to pl.String
    out = s.is_in([])
    assert out.to_list() == [False] * out.len()

    for x_y_z in (["x", "y", "z"], {"x", "y", "z"}):
        out = s.is_in(x_y_z)
        assert out.to_list() == [False, False, False]

    df = pl.DataFrame({"a": [1.0, 2.0], "b": [1, 4], "c": ["e", "d"]})
    assert df.select(pl.col("a").is_in(pl.col("b"))).to_series().to_list() == [
        True,
        False,
    ]
    assert df.select(pl.col("b").is_in([])).to_series().to_list() == [False] * df.height

    with pytest.raises(
        InvalidOperationError,
        match=r"'is_in' cannot check for String values in Int64 data",
    ):
        df.select(pl.col("b").is_in(["x", "x"]))

    # check we don't shallow-copy and accidentally modify 'a' (see: #10072)
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [1, 3]).is_in(a)

    assert a.name == "a"
    assert_series_equal(b, pl.Series("b", [True, False]))


def test_is_in_null() -> None:
    s = pl.Series([None, None], dtype=pl.Null)
    result = s.is_in([1, 2, None])
    expected = pl.Series([None, None], dtype=pl.Boolean)
    assert_series_equal(result, expected)


def test_is_in_invalid_shape() -> None:
    with pytest.raises(ComputeError):
        pl.Series("a", [1, 2, 3]).is_in([[]])


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_is_in_float(dtype: PolarsDataType) -> None:
    s = pl.Series([float("nan"), 0.0], dtype=dtype)
    result = s.is_in([-0.0, -float("nan")])
    expected = pl.Series([True, True], dtype=pl.Boolean)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("df", "matches", "expected_error"),
    [
        (
            pl.DataFrame({"a": [1, 2], "b": [[1.0, 2.5], [3.0, 4.0]]}),
            [True, False],
            None,
        ),
        (
            pl.DataFrame({"a": [2.5, 3.0], "b": [[1, 2], [3, 4]]}),
            [False, True],
            None,
        ),
        (
            pl.DataFrame(
                {"a": [None, None], "b": [[1, 2], [3, 4]]},
                schema_overrides={"a": pl.Null},
            ),
            [None, None],
            None,
        ),
        (
            pl.DataFrame({"a": ["1", "2"], "b": [[1, 2], [3, 4]]}),
            None,
            r"'is_in' cannot check for String values in List\(Int64\) data",
        ),
        (
            pl.DataFrame({"a": [date.today(), None], "b": [[1, 2], [3, 4]]}),
            None,
            r"'is_in' cannot check for Date values in List\(Int64\) data",
        ),
    ],
)
def test_is_in_expr_list_series(
    df: pl.DataFrame, matches: list[bool] | None, expected_error: str | None
) -> None:
    expr_is_in = pl.col("a").is_in(pl.col("b"))
    if matches:
        assert df.select(expr_is_in).to_series().to_list() == matches
    else:
        with pytest.raises(InvalidOperationError, match=expected_error):
            df.select(expr_is_in)


def test_is_in_null_series() -> None:
    df = pl.DataFrame({"a": ["a", "b", None]})
    result = df.select(pl.col("a").is_in([None]))
    expected = pl.DataFrame({"a": [False, False, None]})
    assert_frame_equal(result, expected)


def test_is_in_int_range() -> None:
    r = pl.int_range(0, 3, eager=False)
    out = pl.select(r.is_in([1, 2])).to_series()
    assert out.to_list() == [False, True, True]

    r = pl.int_range(0, 3, eager=True)  # type: ignore[assignment]
    out = r.is_in([1, 2])  # type: ignore[assignment]
    assert out.to_list() == [False, True, True]


def test_is_in_date_range() -> None:
    r = pl.date_range(date(2023, 1, 1), date(2023, 1, 3), eager=False)
    out = pl.select(r.is_in([date(2023, 1, 2), date(2023, 1, 3)])).to_series()
    assert out.to_list() == [False, True, True]

    r = pl.date_range(date(2023, 1, 1), date(2023, 1, 3), eager=True)  # type: ignore[assignment]
    out = r.is_in([date(2023, 1, 2), date(2023, 1, 3)])  # type: ignore[assignment]
    assert out.to_list() == [False, True, True]


@StringCache()
@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
def test_cat_is_in_series(dtype: pl.DataType) -> None:
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    s2 = pl.Series(["b", "c"], dtype=dtype)
    expected = pl.Series([False, True, True, None])
    assert_series_equal(s.is_in(s2), expected)

    s2_str = s2.cast(pl.String)
    assert_series_equal(s.is_in(s2_str), expected)


@StringCache()
def test_cat_is_in_series_non_existent() -> None:
    dtype = pl.Categorical
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    s2 = pl.Series(["a", "d", "e"], dtype=dtype)
    expected = pl.Series([True, False, False, None])
    assert_series_equal(s.is_in(s2), expected)

    s2_str = s2.cast(pl.String)
    assert_series_equal(s.is_in(s2_str), expected)


@StringCache()
def test_enum_is_in_series_non_existent() -> None:
    dtype = pl.Enum(["a", "b", "c"])
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    s2_str = pl.Series(["a", "d", "e"])
    expected = pl.Series([True, False, False, None])
    assert_series_equal(s.is_in(s2_str), expected)


@StringCache()
@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
def test_cat_is_in_with_lit_str(dtype: pl.DataType) -> None:
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    lit = ["b"]
    expected = pl.Series([False, True, False, None])

    assert_series_equal(s.is_in(lit), expected)


@StringCache()
@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
def test_cat_is_in_with_lit_str_non_existent(dtype: pl.DataType) -> None:
    s = pl.Series(["a", "b", "c", None], dtype=dtype)
    lit = ["d"]
    expected = pl.Series([False, False, False, None])

    assert_series_equal(s.is_in(lit), expected)


@StringCache()
@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c"])])
def test_cat_is_in_with_lit_str_cache_setup(dtype: pl.DataType) -> None:
    # init the global cache
    _ = pl.Series(["c", "b", "a"], dtype=dtype)

    assert_series_equal(pl.Series(["a"], dtype=dtype).is_in(["a"]), pl.Series([True]))
    assert_series_equal(pl.Series(["b"], dtype=dtype).is_in(["b"]), pl.Series([True]))
    assert_series_equal(pl.Series(["c"], dtype=dtype).is_in(["c"]), pl.Series([True]))


def test_is_in_with_wildcard_13809() -> None:
    out = pl.DataFrame({"A": ["B"]}).select(pl.all().is_in(["C"]))
    expected = pl.DataFrame({"A": [False]})
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c", "d"])])
def test_cat_is_in_from_str(dtype: pl.DataType) -> None:
    s = pl.Series(["c", "c", "b"], dtype=dtype)

    # test local
    assert_series_equal(
        pl.Series(["a", "d", "e", "b"]).is_in(s),
        pl.Series([False, False, False, True]),
    )


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b", "c", "d"])])
def test_cat_list_is_in_from_cat(dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        [
            (["a", "b"], "c"),
            (["a", "b"], "a"),
            (["a", None], None),
            (["a", "c"], None),
            (["a"], "d"),
        ],
        schema={"li": pl.List(dtype), "x": dtype},
        orient="row",
    )
    res = df.select(pl.col("li").list.contains(pl.col("x")))
    expected_df = pl.DataFrame({"li": [False, True, True, False, False]})
    assert_frame_equal(res, expected_df)


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("b", [True, False, False, None, True]),
        (None, [False, False, True, None, False]),
        ("e", [False, False, False, None, False]),
    ],
)
def test_cat_list_is_in_from_cat_single(val: str | None, expected: list[bool]) -> None:
    df = pl.Series(
        "li",
        [["a", "b"], ["a", "c"], ["a", None], None, ["b"]],
        dtype=pl.List(pl.Categorical),
    ).to_frame()
    res = df.select(pl.col("li").list.contains(pl.lit(val, dtype=pl.Categorical)))
    expected_df = pl.DataFrame({"li": expected})
    assert_frame_equal(res, expected_df)


def test_cat_list_is_in_from_str() -> None:
    df = pl.DataFrame(
        [
            (["a", "b"], "c"),
            (["a", "b"], "a"),
            (["a", None], None),
            (["a", "c"], None),
            (["a"], "d"),
        ],
        schema={"li": pl.List(pl.Categorical), "x": pl.String},
        orient="row",
    )
    res = df.select(pl.col("li").list.contains(pl.col("x")))
    expected_df = pl.DataFrame({"li": [False, True, True, False, False]})
    assert_frame_equal(res, expected_df)


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("b", [True, False, False, None, True]),
        (None, [False, False, True, None, False]),
        ("e", [False, False, False, None, False]),
    ],
)
def test_cat_list_is_in_from_single_str(val: str | None, expected: list[bool]) -> None:
    df = pl.Series(
        "li",
        [["a", "b"], ["a", "c"], ["a", None], None, ["b"]],
        dtype=pl.List(pl.Categorical),
    ).to_frame()
    res = df.select(pl.col("li").list.contains(pl.lit(val, dtype=pl.String)))
    expected_df = pl.DataFrame({"li": expected})
    assert_frame_equal(res, expected_df)


def test_is_in_struct_enum_17618() -> None:
    df = pl.DataFrame()
    dtype = pl.Enum(categories=["HBS"])
    df = df.insert_column(0, pl.Series("category", [], dtype=dtype))
    assert df.filter(
        pl.struct("category").is_in(
            pl.Series(
                [{"category": "HBS"}],
                dtype=pl.Struct({"category": df["category"].dtype}),
            )
        )
    ).shape == (0, 1)


def test_is_in_decimal() -> None:
    assert pl.DataFrame({"a": [D("0.0"), D("0.2"), D("0.1")]}).select(
        pl.col("a").is_in([0.0, 0.1])
    )["a"].to_list() == [True, False, True]
    assert pl.DataFrame({"a": [D("0.0"), D("0.2"), D("0.1")]}).select(
        pl.col("a").is_in([D("0.0"), D("0.1")])
    )["a"].to_list() == [True, False, True]
    assert pl.DataFrame({"a": [D("0.0"), D("0.2"), D("0.1")]}).select(
        pl.col("a").is_in([1, 0, 2])
    )["a"].to_list() == [True, False, False]

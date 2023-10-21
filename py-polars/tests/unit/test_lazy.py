from __future__ import annotations

from datetime import date, datetime
from functools import reduce
from inspect import signature
from operator import add
from string import ascii_letters
from typing import TYPE_CHECKING, Any, Callable, NoReturn, cast

import numpy as np
import pytest

import polars as pl
import polars.selectors as cs
from polars import lit, when
from polars.datatypes import FLOAT_DTYPES
from polars.exceptions import ComputeError, PolarsInefficientMapWarning
from polars.testing import assert_frame_equal
from polars.testing.asserts import assert_series_equal

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


def test_init_signature_match() -> None:
    # eager/lazy init signatures are expected to match; if this test fails, it
    # means a parameter was added to one but not the other, and that should be
    # fixed (or an explicit exemption should be made here, with an explanation)
    assert signature(pl.DataFrame.__init__) == signature(pl.LazyFrame.__init__)


def test_lazy() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    _ = ldf.with_columns(pl.lit(1).alias("foo")).select([pl.col("a"), pl.col("foo")])

    # test if it executes
    _ = ldf.with_columns(
        when(pl.col("a") > pl.lit(2)).then(pl.lit(10)).otherwise(pl.lit(1)).alias("new")
    ).collect()

    # test if pl.list is available, this is `to_list` re-exported as list
    eager = ldf.group_by("a").agg(pl.implode("b")).collect()
    assert sorted(eager.rows()) == [(1, [[1.0]]), (2, [[2.0]]), (3, [[3.0]])]


@pytest.mark.parametrize(
    ("data", "repr_"),
    [
        ({}, "0 cols, {}"),
        ({"a": [1]}, '1 col, {"a": Int64}'),
        ({"a": [1], "b": ["B"]}, '2 cols, {"a": Int64, "b": Utf8}'),
        ({"a": [1], "b": ["B"], "c": [0.0]}, '3 cols, {"a": Int64 … "c": Float64}'),
    ],
)
def test_repr(data: dict[str, list[Any]], repr_: str) -> None:
    ldf = pl.LazyFrame(data)
    assert repr(ldf).startswith(f"<LazyFrame [{repr_}] at ")


def test_lazyframe_membership_operator() -> None:
    ldf = pl.LazyFrame({"name": ["Jane", "John"], "age": [20, 30]})
    assert "name" in ldf
    assert "phone" not in ldf

    # note: cannot use lazyframe in boolean context
    with pytest.raises(TypeError, match="ambiguous"):
        not ldf


def test_apply() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    new = ldf.with_columns_seq(
        pl.col("a").map_batches(lambda s: s * 2, return_dtype=pl.Int64).alias("foo")
    )
    expected = ldf.clone().with_columns((pl.col("a") * 2).alias("foo"))
    assert_frame_equal(new, expected)
    assert_frame_equal(new.collect(), expected.collect())

    with pytest.warns(
        PolarsInefficientMapWarning, match="In this case, you can replace"
    ):
        for strategy in ["thread_local", "threading"]:
            ldf = pl.LazyFrame({"a": [1, 2, 3] * 20, "b": [1.0, 2.0, 3.0] * 20})
            new = ldf.with_columns(
                pl.col("a").map_elements(lambda s: s * 2, strategy=strategy).alias("foo")  # type: ignore[arg-type]
            )
            expected = ldf.clone().with_columns((pl.col("a") * 2).alias("foo"))
            assert_frame_equal(new.collect(), expected.collect())


def test_add_eager_column() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert ldf.width == 2

    out = ldf.with_columns(pl.lit(pl.Series("c", [1, 2, 3]))).collect()
    assert out["c"].sum() == 6
    assert out.width == 3


def test_set_null() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = ldf.with_columns(
        when(pl.col("a") > 1).then(lit(None)).otherwise(100).alias("foo")
    ).collect()
    s = out["foo"]
    assert s[0] == 100
    assert s[1] is None
    assert s[2] is None


def test_take_every() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    expected_df = pl.DataFrame({"a": [1, 3], "b": ["w", "y"]})
    assert_frame_equal(expected_df, ldf.take_every(2).collect())


def test_agg() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().min()
    res = ldf.collect()
    assert res.shape == (1, 2)
    assert res.row(0) == (1, 1.0)


def test_count_suffix_10783() -> None:
    df = pl.DataFrame(
        {
            "a": [["a", "c", "b"], ["a", "b", "c"], ["a", "d", "c"], ["c", "a", "b"]],
            "b": [["a", "c", "b"], ["a", "b", "c"], ["a", "d", "c"], ["c", "a", "b"]],
        }
    )
    df_with_cnt = df.with_columns(
        pl.count().over(pl.col("a").list.sort().list.join("").hash()).suffix("_suffix")
    )
    df_expect = df.with_columns(pl.Series("count_suffix", [3, 3, 1, 3]))
    assert_frame_equal(df_with_cnt, df_expect, check_dtype=False)


def test_or() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    out = ldf.filter((pl.col("a") == 1) | (pl.col("b") > 2)).collect()
    assert out.rows() == [(1, 1.0), (3, 3.0)]


def test_filter_str() -> None:
    # use a str instead of a column expr
    ldf = pl.LazyFrame(
        {
            "time": ["11:11:00", "11:12:00", "11:13:00", "11:14:00"],
            "bools": [True, False, True, False],
        }
    )

    # last row based on a filter
    result = ldf.filter(pl.col("bools")).select_seq(pl.last("*")).collect()
    expected = pl.DataFrame({"time": ["11:13:00"], "bools": [True]})
    assert_frame_equal(result, expected)

    # last row based on a filter
    result = ldf.filter("bools").select(pl.last("*")).collect()
    assert_frame_equal(result, expected)


def test_filter_multiple_predicates() -> None:
    ldf = pl.LazyFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [1, 1, 2, 2, 2],
            "c": [1, 1, 2, 3, 4],
        }
    )

    # using multiple predicates
    # multiple predicates
    expected = pl.DataFrame({"a": [1, 1, 1], "b": [1, 1, 2], "c": [1, 1, 2]})
    for out in (
        ldf.filter(pl.col("a") == 1, pl.col("b") <= 2),  # positional/splat
        ldf.filter([pl.col("a") == 1, pl.col("b") <= 2]),  # as list
    ):
        assert_frame_equal(out.collect(), expected)

    # multiple kwargs
    assert_frame_equal(
        ldf.filter(a=1, b=2).collect(),
        pl.DataFrame({"a": [1], "b": [2], "c": [2]}),
    )

    # both positional and keyword args
    assert_frame_equal(
        ldf.filter(pl.col("c") < 4, a=2, b=2).collect(),
        pl.DataFrame({"a": [2], "b": [2], "c": [3]}),
    )

    # check 'predicate' keyword deprecation:
    # note: can disambiguate new/old usage - only expect warning on old-style usage
    with pytest.warns(
        DeprecationWarning,
        match="`filter` no longer takes a 'predicate' parameter",
    ):
        ldf.filter(predicate=pl.col("a").ge(1)).collect()

    ldf = pl.LazyFrame(
        {
            "description": ["eq", "gt", "ge"],
            "predicate": ["==", ">", ">="],
        },
    )
    assert ldf.filter(predicate="==").select("description").collect().item() == "eq"


def test_apply_custom_function() -> None:
    ldf = pl.LazyFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )

    # two ways to determine the length groups.
    df = (
        ldf.group_by("fruits")
        .agg(
            [
                pl.col("cars")
                .map_elements(lambda groups: groups.len(), return_dtype=pl.Int64)
                .alias("custom_1"),
                pl.col("cars")
                .map_elements(lambda groups: groups.len(), return_dtype=pl.Int64)
                .alias("custom_2"),
                pl.count("cars").alias("cars_count"),
            ]
        )
        .sort("custom_1", descending=True)
    ).collect()

    expected = pl.DataFrame(
        {
            "fruits": ["banana", "apple"],
            "custom_1": [3, 2],
            "custom_2": [3, 2],
            "cars_count": [3, 2],
        }
    )
    expected = expected.with_columns(pl.col("cars_count").cast(pl.UInt32))
    assert_frame_equal(df, expected)


def test_group_by() -> None:
    ldf = pl.LazyFrame({"a": [1.0, None, 3.0, 4.0], "groups": ["a", "a", "b", "b"]})

    expected = pl.DataFrame({"groups": ["a", "b"], "a": [1.0, 3.5]})

    out = ldf.group_by("groups").agg(pl.mean("a")).collect()
    assert_frame_equal(out.sort(by="groups"), expected)

    # refer to column via pl.Expr
    out = ldf.group_by(pl.col("groups")).agg(pl.mean("a")).collect()
    assert_frame_equal(out.sort(by="groups"), expected)


def test_shift(fruits_cars: pl.DataFrame) -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    out = df.select(pl.col("a").shift(1))
    assert_series_equal(out["a"], pl.Series("a", [None, 1, 2, 3, 4]))

    res = fruits_cars.lazy().shift(2).collect()

    expected = pl.DataFrame(
        {
            "A": [None, None, 1, 2, 3],
            "fruits": [None, None, "banana", "banana", "apple"],
            "B": [None, None, 5, 4, 3],
            "cars": [None, None, "beetle", "audi", "beetle"],
        }
    )
    assert_frame_equal(res, expected)

    # negative value
    res = fruits_cars.lazy().shift(-2).collect()
    for rows in [3, 4]:
        for cols in range(4):
            assert res[rows, cols] is None


def test_shift_and_fill() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    # use exprs
    out = ldf.with_columns(
        pl.col("a").shift_and_fill(pl.col("b").mean(), n=-2)
    ).collect()
    assert out["a"].null_count() == 0

    # use df method
    out = ldf.shift_and_fill(pl.col("b").std(), n=2).collect()
    assert out["a"].null_count() == 0


def test_arg_unique() -> None:
    ldf = pl.LazyFrame({"a": [4, 1, 4]})
    col_a_unique = ldf.select(pl.col("a").arg_unique()).collect()["a"]
    assert_series_equal(col_a_unique, pl.Series("a", [0, 1]).cast(pl.UInt32))


def test_is_unique() -> None:
    df = pl.DataFrame({"a": [4, 1, 4]})
    result = df.select(pl.col("a").is_unique())["a"]
    assert_series_equal(result, pl.Series("a", [False, True, False]))


def test_is_duplicated() -> None:
    ldf = pl.LazyFrame({"a": [4, 1, 4]}).select(pl.col("a").is_duplicated())
    assert_series_equal(ldf.collect()["a"], pl.Series("a", [True, False, True]))


def test_arg_sort() -> None:
    ldf = pl.LazyFrame({"a": [4, 1, 3]}).select(pl.col("a").arg_sort())
    assert ldf.collect()["a"].to_list() == [1, 2, 0]


def test_window_function() -> None:
    ldf = pl.LazyFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )
    assert ldf.width == 4

    q = ldf.with_columns(
        [
            pl.sum("A").over("fruits").alias("fruit_sum_A"),
            pl.first("B").over("fruits").alias("fruit_first_B"),
            pl.max("B").over("cars").alias("cars_max_B"),
        ]
    )
    assert q.width == 7

    assert q.collect()["cars_max_B"].to_list() == [5, 4, 5, 5, 5]

    out = ldf.select([pl.first("B").over(["fruits", "cars"]).alias("B_first")])
    assert out.collect()["B_first"].to_list() == [5, 4, 3, 3, 5]


def test_when_then_flatten() -> None:
    ldf = pl.LazyFrame({"foo": [1, 2, 3], "bar": [3, 4, 5]})

    assert ldf.select(
        when(pl.col("foo") > 1)
        .then(pl.col("bar"))
        .when(pl.col("bar") < 3)
        .then(10)
        .otherwise(30)
    ).collect()["bar"].to_list() == [30, 4, 5]


def test_describe_plan() -> None:
    assert isinstance(pl.LazyFrame({"a": [1]}).explain(optimized=True), str)
    assert isinstance(pl.LazyFrame({"a": [1]}).explain(optimized=False), str)


def test_inspect(capsys: CaptureFixture[str]) -> None:
    ldf = pl.LazyFrame({"a": [1]})
    ldf.inspect().collect()
    captured = capsys.readouterr()
    assert len(captured.out) > 0

    ldf.select(pl.col("a").cumsum().inspect().alias("bar")).collect()
    res = capsys.readouterr()
    assert len(res.out) > 0


def test_fetch(fruits_cars: pl.DataFrame) -> None:
    res = fruits_cars.lazy().select("*").fetch(2)
    assert_frame_equal(res, res[:2])


def test_fold_filter() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [0, 1, 2]})

    out = ldf.filter(
        pl.fold(
            acc=pl.lit(True),
            function=lambda a, b: a & b,
            exprs=[pl.col(c) > 1 for c in ldf.columns],
        )
    ).collect()

    assert out.shape == (1, 2)
    assert out.rows() == [(3, 2)]

    out = ldf.filter(
        pl.fold(
            acc=pl.lit(True),
            function=lambda a, b: a | b,
            exprs=[pl.col(c) > 1 for c in ldf.columns],
        )
    ).collect()

    assert out.rows() == [(1, 0), (2, 1), (3, 2)]


def test_head_group_by() -> None:
    commodity_prices = {
        "commodity": [
            "Wheat",
            "Wheat",
            "Wheat",
            "Wheat",
            "Corn",
            "Corn",
            "Corn",
            "Corn",
            "Corn",
        ],
        "location": [
            "StPaul",
            "StPaul",
            "StPaul",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
        ],
        "seller": [
            "Bob",
            "Charlie",
            "Susan",
            "Paul",
            "Ed",
            "Mary",
            "Paul",
            "Charlie",
            "Norman",
        ],
        "price": [1.0, 0.7, 0.8, 0.55, 2.0, 3.0, 2.4, 1.8, 2.1],
    }
    ldf = pl.LazyFrame(commodity_prices)

    # this query flexes the wildcard exclusion quite a bit.
    keys = ["commodity", "location"]
    out = (
        ldf.sort(by="price", descending=True)
        .group_by(keys, maintain_order=True)
        .agg([pl.col("*").exclude(keys).head(2).keep_name()])
        .explode(pl.col("*").exclude(keys))
    )

    assert out.collect().rows() == [
        ("Corn", "Chicago", "Mary", 3.0),
        ("Corn", "Chicago", "Paul", 2.4),
        ("Wheat", "StPaul", "Bob", 1.0),
        ("Wheat", "StPaul", "Susan", 0.8),
        ("Wheat", "Chicago", "Paul", 0.55),
    ]

    ldf = pl.LazyFrame(
        {"letters": ["c", "c", "a", "c", "a", "b"], "nrs": [1, 2, 3, 4, 5, 6]}
    )
    out = ldf.group_by("letters").tail(2).sort("letters")
    assert_frame_equal(
        out.collect(),
        pl.DataFrame({"letters": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 2, 4]}),
    )
    out = ldf.group_by("letters").head(2).sort("letters")
    assert_frame_equal(
        out.collect(),
        pl.DataFrame({"letters": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 1, 2]}),
    )


def test_is_null_is_not_null() -> None:
    ldf = pl.LazyFrame({"nrs": [1, 2, None]}).select(
        pl.col("nrs").is_null().alias("is_null"),
        pl.col("nrs").is_not_null().alias("not_null"),
    )
    assert ldf.collect()["is_null"].to_list() == [False, False, True]
    assert ldf.collect()["not_null"].to_list() == [True, True, False]


def test_is_nan_is_not_nan() -> None:
    ldf = pl.LazyFrame({"nrs": np.array([1, 2, np.nan])}).select(
        pl.col("nrs").is_nan().alias("is_nan"),
        pl.col("nrs").is_not_nan().alias("not_nan"),
    )
    assert ldf.collect()["is_nan"].to_list() == [False, False, True]
    assert ldf.collect()["not_nan"].to_list() == [True, True, False]


def test_is_finite_is_infinite() -> None:
    ldf = pl.LazyFrame({"nrs": np.array([1, 2, np.inf])}).select(
        pl.col("nrs").is_infinite().alias("is_inf"),
        pl.col("nrs").is_finite().alias("not_inf"),
    )
    assert ldf.collect()["is_inf"].to_list() == [False, False, True]
    assert ldf.collect()["not_inf"].to_list() == [True, True, False]


def test_len() -> None:
    ldf = pl.LazyFrame({"nrs": [1, 2, 3]})
    assert cast(int, ldf.select(pl.col("nrs").len()).collect().item()) == 3


def test_cum_agg() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 2]})
    assert_series_equal(
        ldf.select(pl.col("a").cumsum()).collect()["a"], pl.Series("a", [1, 3, 6, 8])
    )
    assert_series_equal(
        ldf.select(pl.col("a").cummin()).collect()["a"], pl.Series("a", [1, 1, 1, 1])
    )
    assert_series_equal(
        ldf.select(pl.col("a").cummax()).collect()["a"], pl.Series("a", [1, 2, 3, 3])
    )
    assert_series_equal(
        ldf.select(pl.col("a").cumprod()).collect()["a"], pl.Series("a", [1, 2, 6, 12])
    )


def test_floor() -> None:
    ldf = pl.LazyFrame({"a": [1.8, 1.2, 3.0]}).select(pl.col("a").floor())
    assert_series_equal(ldf.collect()["a"], pl.Series("a", [1, 1, 3]).cast(pl.Float64))


@pytest.mark.parametrize(
    ("n", "ndigits", "expected"),
    [
        (1.005, 2, 1.0),
        (1234.00000254495, 10, 1234.000002545),
        (1835.665, 2, 1835.67),
        (-1835.665, 2, -1835.67),
        (1.27499, 2, 1.27),
        (123.45678, 2, 123.46),
        (1254, 2, 1254.0),
        (1254, 0, 1254.0),
        (123.55, 0, 124.0),
        (123.55, 1, 123.6),
        (-1.23456789, 6, -1.234568),
        (1.0e-5, 5, 0.00001),
        (1.0e-20, 20, 1e-20),
        (1.0e20, 2, 100000000000000000000.0),
    ],
)
def test_round(n: float, ndigits: int, expected: float) -> None:
    for float_dtype in FLOAT_DTYPES:
        ldf = pl.LazyFrame({"value": [n]}, schema_overrides={"value": float_dtype})
        assert_series_equal(
            ldf.select(pl.col("value").round(decimals=ndigits)).collect().to_series(),
            pl.Series("value", [expected], dtype=float_dtype),
        )


def test_dot() -> None:
    ldf = pl.LazyFrame({"a": [1.8, 1.2, 3.0], "b": [3.2, 1, 2]}).select(
        pl.col("a").dot(pl.col("b"))
    )
    assert cast(float, ldf.collect().item()) == 12.96


def test_sort() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 2]}).select(pl.col("a").sort())
    assert_series_equal(ldf.collect()["a"], pl.Series("a", [1, 2, 2, 3]))


def test_custom_group_by() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})
    out = (
        ldf.group_by("b", maintain_order=True)
        .agg([pl.col("a").map_elements(lambda x: x.sum(), return_dtype=pl.Int64)])
        .collect()
    )
    assert out.rows() == [("a", 1), ("b", 2), ("c", 2)]


def test_lazy_columns() -> None:
    ldf = pl.LazyFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
        }
    )
    assert ldf.select(["a", "c"]).columns == ["a", "c"]


def test_cast_frame() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1.0, 2.5, 3.0],
            "b": [4, 5, None],
            "c": [True, False, True],
            "d": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],
        }
    )

    # cast via col:dtype map
    assert lf.cast(
        dtypes={"b": pl.Float32, "c": pl.Utf8, "d": pl.Datetime("ms")}
    ).schema == {
        "a": pl.Float64,
        "b": pl.Float32,
        "c": pl.Utf8,
        "d": pl.Datetime("ms"),
    }

    # cast via selector:dtype map
    lfc = lf.cast(
        {
            cs.float(): pl.UInt8,
            cs.integer(): pl.Int32,
            cs.temporal(): pl.Utf8,
        }
    )
    assert lfc.schema == {"a": pl.UInt8, "b": pl.Int32, "c": pl.Boolean, "d": pl.Utf8}
    assert lfc.collect().rows() == [
        (1, 4, True, "2020-01-02"),
        (2, 5, False, "2021-03-04"),
        (3, None, True, "2022-05-06"),
    ]

    # cast all fields to a single type
    assert lf.cast(pl.Utf8).collect().to_dict(False) == {
        "a": ["1.0", "2.5", "3.0"],
        "b": ["4", "5", None],
        "c": ["true", "false", "true"],
        "d": ["2020-01-02", "2021-03-04", "2022-05-06"],
    }

    # test 'strict' mode
    lf = pl.LazyFrame({"a": [1000, 2000, 3000]})

    with pytest.raises(ComputeError, match="conversion from `i64` to `u8` failed"):
        lf.cast(pl.UInt8).collect()

    assert lf.cast(pl.UInt8, strict=False).collect().rows() == [
        (None,),
        (None,),
        (None,),
    ]


def test_interpolate() -> None:
    df = pl.DataFrame({"a": [1, None, 3]})
    assert df.select(pl.col("a").interpolate())["a"].to_list() == [1, 2, 3]
    assert df["a"].interpolate().to_list() == [1, 2, 3]
    assert df.interpolate()["a"].to_list() == [1, 2, 3]
    assert df.lazy().interpolate().collect()["a"].to_list() == [1, 2, 3]


def test_fill_nan() -> None:
    df = pl.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert_series_equal(df.fill_nan(2.0)["a"], pl.Series("a", [1.0, 2.0, 3.0]))
    assert_series_equal(
        df.lazy().fill_nan(2.0).collect()["a"], pl.Series("a", [1.0, 2.0, 3.0])
    )
    assert_series_equal(
        df.lazy().fill_nan(None).collect()["a"], pl.Series("a", [1.0, None, 3.0])
    )
    assert_series_equal(
        df.select(pl.col("a").fill_nan(2))["a"], pl.Series("a", [1.0, 2.0, 3.0])
    )
    # nearest
    assert pl.Series([None, 1, None, None, None, -8, None, None, 10]).interpolate(
        method="nearest"
    ).to_list() == [None, 1, 1, -8, -8, -8, -8, 10, 10]


def test_fill_null() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0]})

    assert df.select([pl.col("a").fill_null(strategy="min")])["a"][1] == 1.0
    assert df.lazy().fill_null(2).collect()["a"].to_list() == [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="must specify either"):
        df.fill_null()
    with pytest.raises(ValueError, match="cannot specify both"):
        df.fill_null(value=3.0, strategy="max")
    with pytest.raises(ValueError, match="can only specify `limit`"):
        df.fill_null(strategy="max", limit=2)


def test_backward_fill() -> None:
    ldf = pl.LazyFrame({"a": [1.0, None, 3.0]})
    col_a_backward_fill = ldf.select([pl.col("a").backward_fill()]).collect()["a"]
    assert_series_equal(col_a_backward_fill, pl.Series("a", [1, 3, 3]).cast(pl.Float64))


def test_rolling(fruits_cars: pl.DataFrame) -> None:
    ldf = fruits_cars.lazy()
    out = ldf.select(
        [
            pl.col("A").rolling_min(3, min_periods=1).alias("1"),
            pl.col("A").rolling_min(3).alias("1b"),
            pl.col("A").rolling_mean(3, min_periods=1).alias("2"),
            pl.col("A").rolling_mean(3).alias("2b"),
            pl.col("A").rolling_max(3, min_periods=1).alias("3"),
            pl.col("A").rolling_max(3).alias("3b"),
            pl.col("A").rolling_sum(3, min_periods=1).alias("4"),
            pl.col("A").rolling_sum(3).alias("4b"),
            # below we use .round purely for the ability to do assert frame equality
            pl.col("A").rolling_std(3).round(1).alias("std"),
            pl.col("A").rolling_var(3).round(1).alias("var"),
        ]
    )

    assert_frame_equal(
        out.collect(),
        pl.DataFrame(
            {
                "1": [1, 1, 1, 2, 3],
                "1b": [None, None, 1, 2, 3],
                "2": [1.0, 1.5, 2.0, 3.0, 4.0],
                "2b": [None, None, 2.0, 3.0, 4.0],
                "3": [1, 2, 3, 4, 5],
                "3b": [None, None, 3, 4, 5],
                "4": [1, 3, 6, 9, 12],
                "4b": [None, None, 6, 9, 12],
                "std": [None, None, 1.0, 1.0, 1.0],
                "var": [None, None, 1.0, 1.0, 1.0],
            }
        ),
    )

    out_single_val_variance = ldf.select(
        [
            pl.col("A").rolling_std(3, min_periods=1).round(decimals=4).alias("std"),
            pl.col("A").rolling_var(3, min_periods=1).round(decimals=1).alias("var"),
        ]
    ).collect()

    assert cast(float, out_single_val_variance[0, "std"]) == 0.0
    assert cast(float, out_single_val_variance[0, "var"]) == 0.0


def test_arr_namespace(fruits_cars: pl.DataFrame) -> None:
    ldf = fruits_cars.lazy()
    out = ldf.select(
        [
            "fruits",
            pl.col("B")
            .over("fruits", mapping_strategy="join")
            .list.min()
            .alias("B_by_fruits_min1"),
            pl.col("B")
            .min()
            .over("fruits", mapping_strategy="join")
            .alias("B_by_fruits_min2"),
            pl.col("B")
            .over("fruits", mapping_strategy="join")
            .list.max()
            .alias("B_by_fruits_max1"),
            pl.col("B")
            .max()
            .over("fruits", mapping_strategy="join")
            .alias("B_by_fruits_max2"),
            pl.col("B")
            .over("fruits", mapping_strategy="join")
            .list.sum()
            .alias("B_by_fruits_sum1"),
            pl.col("B")
            .sum()
            .over("fruits", mapping_strategy="join")
            .alias("B_by_fruits_sum2"),
            pl.col("B")
            .over("fruits", mapping_strategy="join")
            .list.mean()
            .alias("B_by_fruits_mean1"),
            pl.col("B")
            .mean()
            .over("fruits", mapping_strategy="join")
            .alias("B_by_fruits_mean2"),
        ]
    )
    expected = pl.DataFrame(
        {
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B_by_fruits_min1": [1, 1, 2, 2, 1],
            "B_by_fruits_min2": [1, 1, 2, 2, 1],
            "B_by_fruits_max1": [5, 5, 3, 3, 5],
            "B_by_fruits_max2": [5, 5, 3, 3, 5],
            "B_by_fruits_sum1": [10, 10, 5, 5, 10],
            "B_by_fruits_sum2": [10, 10, 5, 5, 10],
            "B_by_fruits_mean1": [
                3.3333333333333335,
                3.3333333333333335,
                2.5,
                2.5,
                3.3333333333333335,
            ],
            "B_by_fruits_mean2": [
                3.3333333333333335,
                3.3333333333333335,
                2.5,
                2.5,
                3.3333333333333335,
            ],
        }
    )
    assert_frame_equal(out.collect(), expected)


def test_arithmetic() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3]})

    out = ldf.select(
        [
            (pl.col("a") % 2).alias("1"),
            (2 % pl.col("a")).alias("2"),
            (1 // pl.col("a")).alias("3"),
            (1 * pl.col("a")).alias("4"),
            (1 + pl.col("a")).alias("5"),
            (1 - pl.col("a")).alias("6"),
            (pl.col("a") // 2).alias("7"),
            (pl.col("a") * 2).alias("8"),
            (pl.col("a") + 2).alias("9"),
            (pl.col("a") - 2).alias("10"),
            (-pl.col("a")).alias("11"),
        ]
    )
    expected = pl.DataFrame(
        {
            "1": [1, 0, 1],
            "2": [0, 0, 2],
            "3": [1, 0, 0],
            "4": [1, 2, 3],
            "5": [2, 3, 4],
            "6": [0, -1, -2],
            "7": [0, 1, 1],
            "8": [2, 4, 6],
            "9": [3, 4, 5],
            "10": [-1, 0, 1],
            "11": [-1, -2, -3],
        }
    )
    assert_frame_equal(out.collect(), expected)


def test_float_floor_divide() -> None:
    x = 10.4
    step = 0.5
    ldf = pl.LazyFrame({"x": [x]})
    ldf_res = ldf.with_columns(pl.col("x") // step).collect().item()
    assert ldf_res == x // step


def test_lazy_ufunc() -> None:
    ldf = pl.LazyFrame([pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt8)])
    out = ldf.select(
        [
            np.power(cast(Any, pl.col("a")), 2).alias("power_uint8"),
            np.power(cast(Any, pl.col("a")), 2.0).alias("power_float64"),
            np.power(cast(Any, pl.col("a")), 2, dtype=np.uint16).alias("power_uint16"),
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power_uint8", [1, 4, 9, 16], dtype=pl.UInt8),
            pl.Series("power_float64", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
            pl.Series("power_uint16", [1, 4, 9, 16], dtype=pl.UInt16),
        ]
    )
    assert_frame_equal(out.collect(), expected)


def test_lazy_ufunc_expr_not_first() -> None:
    """Check numpy ufunc expressions also work if expression not the first argument."""
    ldf = pl.LazyFrame([pl.Series("a", [1, 2, 3], dtype=pl.Float64)])
    out = ldf.select(
        [
            np.power(2.0, cast(Any, pl.col("a"))).alias("power"),
            (2.0 / cast(Any, pl.col("a"))).alias("divide_scalar"),
            (np.array([2, 2, 2]) / cast(Any, pl.col("a"))).alias("divide_array"),
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power", [2**1, 2**2, 2**3], dtype=pl.Float64),
            pl.Series("divide_scalar", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
            pl.Series("divide_array", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
        ]
    )
    assert_frame_equal(out.collect(), expected)


def test_argminmax() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 2, 2, 2]})
    out = ldf.select(
        [
            pl.col("a").arg_min().alias("min"),
            pl.col("a").arg_max().alias("max"),
        ]
    ).collect()
    assert out["max"][0] == 4
    assert out["min"][0] == 0

    out = (
        ldf.group_by("b", maintain_order=True)
        .agg([pl.col("a").arg_min().alias("min"), pl.col("a").arg_max().alias("max")])
        .collect()
    )
    assert out["max"][0] == 1
    assert out["min"][0] == 0


def test_rename() -> None:
    ldf = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
    out = ldf.rename({"a": "foo", "b": "bar"}).collect()
    assert out.columns == ["foo", "bar", "c"]


def test_with_column_renamed(fruits_cars: pl.DataFrame) -> None:
    res = fruits_cars.lazy().rename({"A": "C"}).collect()
    assert res.columns[0] == "C"


def test_reverse() -> None:
    out = pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).reverse()
    expected = pl.DataFrame({"a": [2, 1], "b": [4, 3]})
    assert_frame_equal(out.collect(), expected)


def test_limit(fruits_cars: pl.DataFrame) -> None:
    assert_frame_equal(fruits_cars.lazy().limit(1).collect(), fruits_cars[0, :])


def test_head(fruits_cars: pl.DataFrame) -> None:
    assert_frame_equal(fruits_cars.lazy().head(2).collect(), fruits_cars[:2, :])


def test_tail(fruits_cars: pl.DataFrame) -> None:
    assert_frame_equal(fruits_cars.lazy().tail(2).collect(), fruits_cars[3:, :])


def test_last(fruits_cars: pl.DataFrame) -> None:
    result = fruits_cars.lazy().last().collect()
    expected = fruits_cars[(len(fruits_cars) - 1) :, :]
    assert_frame_equal(result, expected)


def test_first(fruits_cars: pl.DataFrame) -> None:
    assert_frame_equal(fruits_cars.lazy().first().collect(), fruits_cars[0, :])


def test_join_suffix() -> None:
    df_left = pl.DataFrame(
        {
            "a": ["a", "b", "a", "z"],
            "b": [1, 2, 3, 4],
            "c": [6, 5, 4, 3],
        }
    )
    df_right = pl.DataFrame(
        {
            "a": ["b", "c", "b", "a"],
            "b": [0, 3, 9, 6],
            "c": [1, 0, 2, 1],
        }
    )
    out = df_left.join(df_right, on="a", suffix="_bar")
    assert out.columns == ["a", "b", "c", "b_bar", "c_bar"]
    out = df_left.lazy().join(df_right.lazy(), on="a", suffix="_bar").collect()
    assert out.columns == ["a", "b", "c", "b_bar", "c_bar"]


@pytest.mark.parametrize("no_optimization", [False, True])
def test_collect_all(df: pl.DataFrame, no_optimization: bool) -> None:
    lf1 = df.lazy().select(pl.col("int").sum())
    lf2 = df.lazy().select((pl.col("floats") * 2).sum())
    out = pl.collect_all([lf1, lf2], no_optimization=no_optimization)
    assert cast(int, out[0].item()) == 6
    assert cast(float, out[1].item()) == 12.0


def test_spearman_corr() -> None:
    ldf = pl.LazyFrame(
        {
            "era": [1, 1, 1, 2, 2, 2],
            "prediction": [2, 4, 5, 190, 1, 4],
            "target": [1, 3, 2, 1, 43, 3],
        }
    )

    out = (
        ldf.group_by("era", maintain_order=True).agg(
            pl.corr(pl.col("prediction"), pl.col("target"), method="spearman").alias(
                "c"
            ),
        )
    ).collect()["c"]
    assert np.isclose(out[0], 0.5)
    assert np.isclose(out[1], -1.0)

    # we can also pass in column names directly
    out = (
        ldf.group_by("era", maintain_order=True).agg(
            pl.corr("prediction", "target", method="spearman").alias("c"),
        )
    ).collect()["c"]
    assert np.isclose(out[0], 0.5)
    assert np.isclose(out[1], -1.0)


def test_spearman_corr_ties() -> None:
    """In Spearman correlation, ranks are computed using the average method ."""
    df = pl.DataFrame({"a": [1, 1, 1, 2, 3, 7, 4], "b": [4, 3, 2, 2, 4, 3, 1]})

    result = df.select(
        pl.corr("a", "b", method="spearman").alias("a1"),
        pl.corr(pl.col("a").rank("min"), pl.col("b").rank("min")).alias("a2"),
        pl.corr(pl.col("a").rank(), pl.col("b").rank()).alias("a3"),
    )
    expected = pl.DataFrame(
        [
            pl.Series("a1", [-0.19048482943986483], dtype=pl.Float64),
            pl.Series("a2", [-0.17223653586587362], dtype=pl.Float64),
            pl.Series("a3", [-0.19048482943986483], dtype=pl.Float64),
        ]
    )
    assert_frame_equal(result, expected)


def test_pearson_corr() -> None:
    ldf = pl.LazyFrame(
        {
            "era": [1, 1, 1, 2, 2, 2],
            "prediction": [2, 4, 5, 190, 1, 4],
            "target": [1, 3, 2, 1, 43, 3],
        }
    )

    out = (
        ldf.group_by("era", maintain_order=True).agg(
            pl.corr(pl.col("prediction"), pl.col("target"), method="pearson").alias(
                "c"
            ),
        )
    ).collect()["c"]
    assert out.to_list() == pytest.approx([0.6546536707079772, -5.477514993831792e-1])

    # we can also pass in column names directly
    out = (
        ldf.group_by("era", maintain_order=True).agg(
            pl.corr("prediction", "target", method="pearson").alias("c"),
        )
    ).collect()["c"]
    assert out.to_list() == pytest.approx([0.6546536707079772, -5.477514993831792e-1])


def test_cov(fruits_cars: pl.DataFrame) -> None:
    ldf = fruits_cars.lazy()
    cov_a_b = pl.cov(pl.col("A"), pl.col("B"))
    cov_ab = pl.cov("A", "B")
    assert cast(float, ldf.select(cov_a_b).collect().item()) == -2.5
    assert cast(float, ldf.select(cov_ab).collect().item()) == -2.5


def test_std(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().std().collect()["A"][0] == pytest.approx(
        1.5811388300841898
    )


def test_var(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().var().collect()["A"][0] == pytest.approx(2.5)


def test_max(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().max().collect()["A"][0] == 5
    assert fruits_cars.select(pl.col("A").max())["A"][0] == 5


def test_min(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().min().collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").min())["A"][0] == 1


def test_median(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().median().collect()["A"][0] == 3
    assert fruits_cars.select(pl.col("A").median())["A"][0] == 3


def test_quantile(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().quantile(0.25, "nearest").collect()["A"][0] == 2
    assert fruits_cars.select(pl.col("A").quantile(0.25, "nearest"))["A"][0] == 2

    assert fruits_cars.lazy().quantile(0.24, "lower").collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").quantile(0.24, "lower"))["A"][0] == 1

    assert fruits_cars.lazy().quantile(0.26, "higher").collect()["A"][0] == 3
    assert fruits_cars.select(pl.col("A").quantile(0.26, "higher"))["A"][0] == 3

    assert fruits_cars.lazy().quantile(0.24, "midpoint").collect()["A"][0] == 1.5
    assert fruits_cars.select(pl.col("A").quantile(0.24, "midpoint"))["A"][0] == 1.5

    assert fruits_cars.lazy().quantile(0.24, "linear").collect()["A"][0] == 1.96
    assert fruits_cars.select(pl.col("A").quantile(0.24, "linear"))["A"][0] == 1.96


def test_null_count() -> None:
    lf = pl.LazyFrame({"a": [1, 2, None, 2], "b": [None, 3, None, 3]})
    assert lf.null_count().collect().rows() == [(1, 2)]


def test_unique() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 2], "b": [3, 3, 3]})

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 3]})
    assert_frame_equal(ldf.unique(maintain_order=True).collect(), expected)

    result = ldf.unique(subset="b", maintain_order=True).collect()
    expected = pl.DataFrame({"a": [1], "b": [3]})
    assert_frame_equal(result, expected)

    s0 = pl.Series("a", [1, 2, None, 2])
    # test if the null is included
    assert s0.unique().to_list() == [None, 1, 2]


def test_lazy_concat(df: pl.DataFrame) -> None:
    shape = df.shape
    shape = (shape[0] * 2, shape[1])

    out = pl.concat([df.lazy(), df.lazy()]).collect()
    assert out.shape == shape
    assert_frame_equal(out, df.vstack(df))


def test_self_join() -> None:
    # 2720
    ldf = pl.from_dict(
        data={
            "employee_id": [100, 101, 102],
            "employee_name": ["James", "Alice", "Bob"],
            "manager_id": [None, 100, 101],
        }
    ).lazy()

    out = (
        ldf.join(other=ldf, left_on="manager_id", right_on="employee_id", how="left")
        .select(
            [
                pl.col("employee_id"),
                pl.col("employee_name"),
                pl.col("employee_name_right").alias("manager_name"),
            ]
        )
        .fetch()
    )
    assert set(out.rows()) == {
        (100, "James", None),
        (101, "Alice", "James"),
        (102, "Bob", "Alice"),
    }


def test_preservation_of_subclasses() -> None:
    """Test for LazyFrame inheritance."""

    # We should be able to inherit from polars.LazyFrame
    class SubClassedLazyFrame(pl.LazyFrame):
        pass

    # The constructor creates an object which is an instance of both the
    # superclass and subclass
    ldf = pl.DataFrame({"column_1": [1, 2, 3]}).lazy()
    ldf.__class__ = SubClassedLazyFrame
    extended_ldf = ldf.with_columns(pl.lit(1).alias("column_2"))

    assert isinstance(extended_ldf, pl.LazyFrame)
    assert isinstance(extended_ldf, SubClassedLazyFrame)


def test_group_lengths() -> None:
    ldf = pl.LazyFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "B"],
            "id": ["1", "1", "2", "3", "4", "3", "5"],
        }
    )

    result = ldf.group_by(["group"], maintain_order=True).agg(
        [
            (pl.col("id").unique_counts() / pl.col("id").len())
            .sum()
            .alias("unique_counts_sum"),
            pl.col("id").unique().len().alias("unique_len"),
        ]
    )
    expected = pl.DataFrame(
        {
            "group": ["A", "B"],
            "unique_counts_sum": [1.0, 1.0],
            "unique_len": [2, 3],
        },
        schema_overrides={"unique_len": pl.UInt32},
    )
    assert_frame_equal(result.collect(), expected)


def test_quantile_filtered_agg() -> None:
    assert (
        pl.LazyFrame(
            {
                "group": [0, 0, 0, 0, 1, 1, 1, 1],
                "value": [1, 2, 3, 4, 1, 2, 3, 4],
            }
        )
        .group_by("group")
        .agg(pl.col("value").filter(pl.col("value") < 2).quantile(0.5))
        .collect()["value"]
        .to_list()
    ) == [1.0, 1.0]


def test_lazy_schema() -> None:
    ldf = pl.LazyFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    assert ldf.schema == {"foo": pl.Int64, "bar": pl.Float64, "ham": pl.Utf8}

    ldf = pl.LazyFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    assert ldf.dtypes == [pl.Int64, pl.Float64, pl.Utf8]

    ldfe = ldf.clear()
    assert ldfe.schema == ldf.schema

    ldfe = ldf.clear(2)
    assert ldfe.schema == ldf.schema
    assert ldfe.collect().rows() == [(None, None, None), (None, None, None)]


def test_predicate_count_vstack() -> None:
    l1 = pl.LazyFrame(
        {
            "k": ["x", "y"],
            "v": [3, 2],
        }
    )
    l2 = pl.LazyFrame(
        {
            "k": ["x", "y"],
            "v": [5, 7],
        }
    )
    assert pl.concat([l1, l2]).filter(pl.count().over("k") == 2).collect()[
        "v"
    ].to_list() == [3, 2, 5, 7]


def test_lazy_method() -> None:
    # We want to support `.lazy()` on a Lazy DataFrame to allow more generic user code.
    df = pl.DataFrame({"a": [1, 1, 2, 2, 3, 3], "b": [1, 2, 3, 4, 5, 6]})
    assert_frame_equal(df.lazy(), df.lazy().lazy())


def test_update_schema_after_projection_pd_t4157() -> None:
    ldf = pl.LazyFrame({"c0": [], "c1": [], "c2": []}).rename({"c2": "c2_"})
    assert ldf.drop("c2_").select(pl.col("c0")).collect().columns == ["c0"]


def test_type_coercion_unknown_4190() -> None:
    df = (
        pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).with_columns(
            [pl.col("a") & pl.col("a").fill_null(True)]
        )
    ).collect()
    assert df.shape == (3, 2)
    assert df.rows() == [(1, 1), (2, 2), (3, 3)]


def test_lazy_cache_same_key() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": ["x", "y", "z"]})

    # these have the same schema, but should not be used by cache as they are different
    add_node = ldf.select([(pl.col("a") + pl.col("b")).alias("a"), pl.col("c")]).cache()
    mult_node = ldf.select(
        [(pl.col("a") * pl.col("b")).alias("a"), pl.col("c")]
    ).cache()

    assert mult_node.join(add_node, on="c", suffix="_mult").select(
        [(pl.col("a") - pl.col("a_mult")).alias("a"), pl.col("c")]
    ).collect().to_dict(False) == {"a": [-1, 2, 7], "c": ["x", "y", "z"]}


def test_lazy_cache_hit(monkeypatch: Any, capfd: Any) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    ldf = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": ["x", "y", "z"]})
    add_node = ldf.select([(pl.col("a") + pl.col("b")).alias("a"), pl.col("c")]).cache()
    assert add_node.join(add_node, on="c", suffix="_mult").select(
        [(pl.col("a") - pl.col("a_mult")).alias("a"), pl.col("c")]
    ).collect().to_dict(False) == {"a": [0, 0, 0], "c": ["x", "y", "z"]}

    (out, _) = capfd.readouterr()
    assert "CACHE HIT" in out


def test_lazy_cache_parallel() -> None:
    df_evaluated = 0

    def map_df(df: pl.DataFrame) -> pl.DataFrame:
        nonlocal df_evaluated
        df_evaluated += 1
        return df

    df = pl.LazyFrame({"a": [1]}).map_batches(map_df).cache()

    df = pl.concat(
        [
            df.select(pl.col("a") + 1),
            df.select(pl.col("a") + 2),
            df.select(pl.col("a") + 3),
        ],
        parallel=True,
    )

    assert df_evaluated == 0

    df.collect()
    assert df_evaluated == 1


def test_lazy_cache_nested_parallel() -> None:
    df_inner_evaluated = 0
    df_outer_evaluated = 0

    def map_df_inner(df: pl.DataFrame) -> pl.DataFrame:
        nonlocal df_inner_evaluated
        df_inner_evaluated += 1
        return df

    def map_df_outer(df: pl.DataFrame) -> pl.DataFrame:
        nonlocal df_outer_evaluated
        df_outer_evaluated += 1
        return df

    df_inner = pl.LazyFrame({"a": [1]}).map_batches(map_df_inner).cache()
    df_outer = df_inner.select(pl.col("a") + 1).map_batches(map_df_outer).cache()

    df = pl.concat(
        [
            df_outer.select(pl.col("a") + 2),
            df_outer.select(pl.col("a") + 3),
        ],
        parallel=True,
    )

    assert df_inner_evaluated == 0
    assert df_outer_evaluated == 0

    df.collect()
    assert df_inner_evaluated == 1
    assert df_outer_evaluated == 1


def test_quadratic_behavior_4736() -> None:
    # no assert; if this function does not stall our tests it has passed!
    ldf = pl.LazyFrame(schema=list(ascii_letters))
    ldf.select(reduce(add, (pl.col(fld) for fld in ldf.columns)))


@pytest.mark.parametrize("input_dtype", [pl.Utf8, pl.Int64, pl.Float64])
def test_from_epoch(input_dtype: pl.PolarsDataType) -> None:
    ldf = pl.LazyFrame(
        [
            pl.Series("timestamp_d", [13285]).cast(input_dtype),
            pl.Series("timestamp_s", [1147880044]).cast(input_dtype),
            pl.Series("timestamp_ms", [1147880044 * 1_000]).cast(input_dtype),
            pl.Series("timestamp_us", [1147880044 * 1_000_000]).cast(input_dtype),
            pl.Series("timestamp_ns", [1147880044 * 1_000_000_000]).cast(input_dtype),
        ]
    )

    exp_dt = datetime(2006, 5, 17, 15, 34, 4)
    expected = pl.DataFrame(
        [
            pl.Series("timestamp_d", [date(2006, 5, 17)]),
            pl.Series("timestamp_s", [exp_dt]),  # s is no Polars dtype, defaults to us
            pl.Series("timestamp_ms", [exp_dt]).cast(pl.Datetime("ms")),
            pl.Series("timestamp_us", [exp_dt]),  # us is Polars Datetime default
            pl.Series("timestamp_ns", [exp_dt]).cast(pl.Datetime("ns")),
        ]
    )

    ldf_result = ldf.select(
        [
            pl.from_epoch(pl.col("timestamp_d"), time_unit="d"),
            pl.from_epoch(pl.col("timestamp_s"), time_unit="s"),
            pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms"),
            pl.from_epoch(pl.col("timestamp_us"), time_unit="us"),
            pl.from_epoch(pl.col("timestamp_ns"), time_unit="ns"),
        ]
    ).collect()

    assert_frame_equal(ldf_result, expected)

    ts_col = pl.col("timestamp_s")
    with pytest.raises(ValueError):
        _ = ldf.select(pl.from_epoch(ts_col, time_unit="s2"))  # type: ignore[call-overload]


def test_cumagg_types() -> None:
    ldf = pl.LazyFrame({"a": [1, 2], "b": [True, False], "c": [1.3, 2.4]})
    cumsum_lf = ldf.select(
        [pl.col("a").cumsum(), pl.col("b").cumsum(), pl.col("c").cumsum()]
    )
    assert cumsum_lf.schema["a"] == pl.Int64
    assert cumsum_lf.schema["b"] == pl.UInt32
    assert cumsum_lf.schema["c"] == pl.Float64
    collected_cumsum_lf = cumsum_lf.collect()
    assert collected_cumsum_lf.schema == cumsum_lf.schema

    cumprod_lf = ldf.select(
        [
            pl.col("a").cast(pl.UInt64).cumprod(),
            pl.col("b").cumprod(),
            pl.col("c").cumprod(),
        ]
    )
    assert cumprod_lf.schema["a"] == pl.UInt64
    assert cumprod_lf.schema["b"] == pl.Int64
    assert cumprod_lf.schema["c"] == pl.Float64
    collected_cumprod_lf = cumprod_lf.collect()
    assert collected_cumprod_lf.schema == cumprod_lf.schema


def test_compare_schema_between_lazy_and_eager_6904() -> None:
    float32_df = pl.DataFrame({"x": pl.Series(values=[], dtype=pl.Float32)})
    eager_result = float32_df.select(pl.col("x").sqrt()).select(pl.col(pl.Float32))
    lazy_result = (
        float32_df.lazy()
        .select(pl.col("x").sqrt())
        .select(pl.col(pl.Float32))
        .collect()
    )
    assert eager_result.shape == lazy_result.shape

    eager_result = float32_df.select(pl.col("x").pow(2)).select(pl.col(pl.Float32))
    lazy_result = (
        float32_df.lazy()
        .select(pl.col("x").pow(2))
        .select(pl.col(pl.Float32))
        .collect()
    )
    assert eager_result.shape == lazy_result.shape

    int32_df = pl.DataFrame({"x": pl.Series(values=[], dtype=pl.Int32)})
    eager_result = int32_df.select(pl.col("x").pow(2)).select(pl.col(pl.Float64))
    lazy_result = (
        int32_df.lazy().select(pl.col("x").pow(2)).select(pl.col(pl.Float64)).collect()
    )
    assert eager_result.shape == lazy_result.shape

    int8_df = pl.DataFrame({"x": pl.Series(values=[], dtype=pl.Int8)})
    eager_result = int8_df.select(pl.col("x").diff()).select(pl.col(pl.Int16))
    lazy_result = (
        int8_df.lazy().select(pl.col("x").diff()).select(pl.col(pl.Int16)).collect()
    )
    assert eager_result.shape == lazy_result.shape


@pytest.mark.parametrize(
    "dtype",
    [
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Float32,
        pl.Float64,
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        pl.col("x").arg_max(),
        pl.col("x").arg_min(),
        pl.col("x").max(),
        pl.col("x").mean(),
        pl.col("x").median(),
        pl.col("x").min(),
        pl.col("x").nan_max(),
        pl.col("x").nan_min(),
        pl.col("x").product(),
        pl.col("x").quantile(0.5),
        pl.col("x").std(),
        pl.col("x").sum(),
        pl.col("x").var(),
    ],
)
def test_compare_aggregation_between_lazy_and_eager_6904(
    dtype: pl.PolarsDataType, func: pl.Expr
) -> None:
    df = pl.DataFrame(
        {
            "x": pl.Series(values=[1, 2, 3] * 2, dtype=dtype),
            "y": pl.Series(values=["a"] * 3 + ["b"] * 3),
        }
    )
    result_eager = df.select(func.over("y")).select("x")
    dtype_eager = result_eager["x"].dtype
    result_lazy = df.lazy().select(func.over("y")).select(pl.col(dtype_eager)).collect()
    assert result_eager.frame_equal(result_lazy)


@pytest.mark.parametrize(
    "comparators",
    [
        ("==", pl.LazyFrame.__eq__),
        ("!=", pl.LazyFrame.__ne__),
        (">", pl.LazyFrame.__gt__),
        ("<", pl.LazyFrame.__lt__),
        (">=", pl.LazyFrame.__ge__),
        ("<=", pl.LazyFrame.__le__),
    ],
)
def test_lazy_comparison_operators(
    comparators: tuple[str, Callable[[pl.LazyFrame, Any], NoReturn]]
) -> None:
    # we cannot compare lazy frames, so all should raise a TypeError
    with pytest.raises(
        TypeError,
        match=f'"{comparators[0]!r}" comparison not supported for LazyFrame objects',
    ):
        comparators[1](pl.LazyFrame(), pl.LazyFrame())

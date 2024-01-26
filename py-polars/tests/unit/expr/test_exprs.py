from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta, timezone
from itertools import permutations
from typing import Any, cast

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
else:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo

import numpy as np
import pytest

import polars as pl
from polars.datatypes import (
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERIC_DTYPES,
    TEMPORAL_DTYPES,
)
from polars.testing import assert_frame_equal, assert_series_equal


def test_arg_true() -> None:
    df = pl.DataFrame({"a": [1, 1, 2, 1]})
    res = df.select((pl.col("a") == 1).arg_true())
    expected = pl.DataFrame([pl.Series("a", [0, 1, 3], dtype=pl.UInt32)])
    assert_frame_equal(res, expected)


def test_suffix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().name.suffix("_reverse")])
    assert out.columns == ["A_reverse", "fruits_reverse", "B_reverse", "cars_reverse"]


def test_pipe() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, None, 8]})

    def _multiply(expr: pl.Expr, mul: int) -> pl.Expr:
        return expr * mul

    result = df.select(
        pl.col("foo").pipe(_multiply, mul=2),
        pl.col("bar").pipe(_multiply, mul=3),
    )

    expected = pl.DataFrame({"foo": [2, 4, 6], "bar": [18, None, 24]})
    assert_frame_equal(result, expected)


def test_prefix(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select([pl.all().name.prefix("reverse_")])
    assert out.columns == ["reverse_A", "reverse_fruits", "reverse_B", "reverse_cars"]


def test_filter_where() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 1, 2, 3], "b": [4, 5, 6, 7, 8, 9]})
    result_filter = df.group_by("a", maintain_order=True).agg(
        pl.col("b").filter(pl.col("b") > 4).alias("c")
    )
    expected = pl.DataFrame({"a": [1, 2, 3], "c": [[7], [5, 8], [6, 9]]})
    assert_frame_equal(result_filter, expected)

    with pytest.deprecated_call():
        result_where = df.group_by("a", maintain_order=True).agg(
            pl.col("b").where(pl.col("b") > 4).alias("c")
        )
    assert_frame_equal(result_where, expected)

    # apply filter constraints using kwargs
    df = pl.DataFrame(
        {
            "key": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
            "n": [1, 4, 4, 2, 2, 3, 1, 3, 0, 2, 3, 4],
        },
        schema_overrides={"n": pl.UInt8},
    )
    res = (
        df.group_by("key")
        .agg(
            n_0=pl.col("n").filter(n=0),
            n_1=pl.col("n").filter(n=1),
            n_2=pl.col("n").filter(n=2),
            n_3=pl.col("n").filter(n=3),
            n_4=pl.col("n").filter(n=4),
        )
        .sort(by="key")
    )
    assert res.rows() == [
        ("a", [], [1], [2, 2], [3], [4, 4]),
        ("b", [0], [1], [2], [3, 3], [4]),
    ]


def test_len_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 3, 3], "b": ["a", "a", "b", "a", "a"]})

    out = df.select(pl.len())
    assert out.shape == (1, 1)
    assert cast(int, out.item()) == 5

    out = df.group_by("b", maintain_order=True).agg(pl.len())
    assert out["b"].to_list() == ["a", "b"]
    assert out["len"].to_list() == [4, 1]


def test_map_alias() -> None:
    out = pl.DataFrame({"foo": [1, 2, 3]}).select(
        (pl.col("foo") * 2).name.map(lambda name: f"{name}{name}")
    )
    expected = pl.DataFrame({"foofoo": [2, 4, 6]})
    assert_frame_equal(out, expected)


def test_unique_stable() -> None:
    s = pl.Series("a", [1, 1, 1, 1, 2, 2, 2, 3, 3])
    expected = pl.Series("a", [1, 2, 3])
    assert_series_equal(s.unique(maintain_order=True), expected)


def test_entropy() -> None:
    df = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "B"],
            "id": [1, 2, 1, 4, 5, 4, 6],
        }
    )
    result = df.group_by("group", maintain_order=True).agg(
        pl.col("id").entropy(normalize=True)
    )
    expected = pl.DataFrame(
        {"group": ["A", "B"], "id": [1.0397207708399179, 1.371381017771811]}
    )
    assert_frame_equal(result, expected)


def test_dot_in_group_by() -> None:
    df = pl.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b", "b"],
            "x": [1, 1, 1, 1, 1, 1],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )

    result = df.group_by("group", maintain_order=True).agg(
        pl.col("x").dot("y").alias("dot")
    )
    expected = pl.DataFrame({"group": ["a", "b"], "dot": [6, 15]})
    assert_frame_equal(result, expected)


def test_dtype_col_selection() -> None:
    df = pl.DataFrame(
        data=[],
        schema={
            "a1": pl.Datetime,
            "a2": pl.Datetime("ms"),
            "a3": pl.Datetime("ms"),
            "a4": pl.Datetime("ns"),
            "b": pl.Date,
            "c": pl.Time,
            "d1": pl.Duration,
            "d2": pl.Duration("ms"),
            "d3": pl.Duration("us"),
            "d4": pl.Duration("ns"),
            "e": pl.Int8,
            "f": pl.Int16,
            "g": pl.Int32,
            "h": pl.Int64,
            "i": pl.Float32,
            "j": pl.Float64,
            "k": pl.UInt8,
            "l": pl.UInt16,
            "m": pl.UInt32,
            "n": pl.UInt64,
        },
    )
    assert df.select(pl.col(INTEGER_DTYPES)).columns == [
        "e",
        "f",
        "g",
        "h",
        "k",
        "l",
        "m",
        "n",
    ]
    assert df.select(pl.col(FLOAT_DTYPES)).columns == ["i", "j"]
    assert df.select(pl.col(NUMERIC_DTYPES)).columns == [
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
    ]
    assert df.select(pl.col(TEMPORAL_DTYPES)).columns == [
        "a1",
        "a2",
        "a3",
        "a4",
        "b",
        "c",
        "d1",
        "d2",
        "d3",
        "d4",
    ]
    assert df.select(pl.col(DATETIME_DTYPES)).columns == [
        "a1",
        "a2",
        "a3",
        "a4",
    ]
    assert df.select(pl.col(DURATION_DTYPES)).columns == [
        "d1",
        "d2",
        "d3",
        "d4",
    ]


def test_list_eval_expression() -> None:
    df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})

    for parallel in [True, False]:
        assert df.with_columns(
            pl.concat_list(["a", "b"])
            .list.eval(pl.first().rank(), parallel=parallel)
            .alias("rank")
        ).to_dict(as_series=False) == {
            "a": [1, 8, 3],
            "b": [4, 5, 2],
            "rank": [[1.0, 2.0], [2.0, 1.0], [2.0, 1.0]],
        }

        assert df["a"].reshape((1, -1)).list.eval(
            pl.first(), parallel=parallel
        ).to_list() == [[1, 8, 3]]


def test_null_count_expr() -> None:
    df = pl.DataFrame({"key": ["a", "b", "b", "a"], "val": [1, 2, None, 1]})

    assert df.select([pl.all().null_count()]).to_dict(as_series=False) == {
        "key": [0],
        "val": [1],
    }


def test_pos_neg() -> None:
    df = pl.DataFrame(
        {
            "x": [3, 2, 1],
            "y": [6, 7, 8],
        }
    ).with_columns(-pl.col("x"), +pl.col("y"), -pl.lit(1))

    # #11149: ensure that we preserve the output name (where available)
    assert df.to_dict(as_series=False) == {
        "x": [-3, -2, -1],
        "y": [6, 7, 8],
        "literal": [-1, -1, -1],
    }


def test_power_by_expression() -> None:
    out = pl.DataFrame(
        {"a": [1, None, None, 4, 5, 6], "b": [1, 2, None, 4, None, 6]}
    ).select(
        [
            pl.col("a").pow(pl.col("b")).alias("pow_expr"),
            (pl.col("a") ** pl.col("b")).alias("pow_op"),
            (2 ** pl.col("b")).alias("pow_op_left"),
        ]
    )

    for pow_col in ("pow_expr", "pow_op"):
        assert out[pow_col].to_list() == [1.0, None, None, 256.0, None, 46656.0]
    assert out["pow_op_left"].to_list() == [2.0, 4.0, None, 16.0, None, 64.0]


def test_expression_appends() -> None:
    df = pl.DataFrame({"a": [1, 1, 2]})

    assert df.select(pl.repeat(None, 3).append(pl.col("a"))).n_chunks() == 2
    assert df.select(pl.repeat(None, 3).append(pl.col("a")).rechunk()).n_chunks() == 1

    out = df.select(pl.concat([pl.repeat(None, 3), pl.col("a")]))

    assert out.n_chunks() == 1
    assert out.to_series().to_list() == [None, None, None, 1, 1, 2]


def test_arr_contains() -> None:
    df_groups = pl.DataFrame(
        {
            "animals": [
                ["cat", "mouse", "dog"],
                ["dog", "hedgehog", "mouse", "cat"],
                ["peacock", "mouse", "aardvark"],
            ],
        }
    )
    # string array contains
    assert df_groups.lazy().filter(
        pl.col("animals").list.contains("mouse"),
    ).collect().to_dict(as_series=False) == {
        "animals": [
            ["cat", "mouse", "dog"],
            ["dog", "hedgehog", "mouse", "cat"],
            ["peacock", "mouse", "aardvark"],
        ]
    }
    # string array contains and *not* contains
    assert df_groups.filter(
        pl.col("animals").list.contains("mouse"),
        ~pl.col("animals").list.contains("hedgehog"),
    ).to_dict(as_series=False) == {
        "animals": [
            ["cat", "mouse", "dog"],
            ["peacock", "mouse", "aardvark"],
        ],
    }


def test_rank() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
        }
    )

    s = df.select(pl.col("a").rank(method="average").alias("b")).to_series()
    assert s.to_list() == [1.5, 1.5, 3.5, 3.5, 5.0]
    assert s.dtype == pl.Float64

    s = df.select(pl.col("a").rank(method="max").alias("b")).to_series()
    assert s.to_list() == [2, 2, 4, 4, 5]
    assert s.dtype == pl.get_index_type()


def test_rank_so_4109() -> None:
    # also tests ranks null behavior
    df = pl.from_dict(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "rank": [None, 3, 2, 4, 1, 4, 3, 2, 1, None, 3, 4, 4, 1, None, 3],
        }
    ).sort(by=["id", "rank"])

    assert df.group_by("id").agg(
        [
            pl.col("rank").alias("original"),
            pl.col("rank").rank(method="dense").alias("dense"),
            pl.col("rank").rank(method="average").alias("average"),
        ]
    ).to_dict(as_series=False) == {
        "id": [1, 2, 3, 4],
        "original": [[None, 2, 3, 4], [1, 2, 3, 4], [None, 1, 3, 4], [None, 1, 3, 4]],
        "dense": [[None, 1, 2, 3], [1, 2, 3, 4], [None, 1, 2, 3], [None, 1, 2, 3]],
        "average": [
            [None, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [None, 1.0, 2.0, 3.0],
            [None, 1.0, 2.0, 3.0],
        ],
    }


def test_rank_string_null_11252() -> None:
    rank = pl.Series([None, "", "z", None, "a"]).rank()
    assert rank.to_list() == [None, 1.0, 3.0, None, 2.0]


def test_search_sorted() -> None:
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        arr = np.sort(np.random.randn(10) * 100)
        s = pl.Series(arr)

        for v in range(int(np.min(arr)), int(np.max(arr)), 20):
            assert np.searchsorted(arr, v) == s.search_sorted(v)

    a = pl.Series([1, 2, 3])
    b = pl.Series([1, 2, 2, -1])
    assert a.search_sorted(b).to_list() == [0, 1, 1, 0]
    b = pl.Series([1, 2, 2, None, 3])
    assert a.search_sorted(b).to_list() == [0, 1, 1, 0, 2]

    a = pl.Series(["b", "b", "d", "d"])
    b = pl.Series(["a", "b", "c", "d", "e"])
    assert a.search_sorted(b, side="left").to_list() == [0, 0, 2, 2, 4]
    assert a.search_sorted(b, side="right").to_list() == [0, 2, 2, 4, 4]

    a = pl.Series([1, 1, 4, 4])
    b = pl.Series([0, 1, 2, 4, 5])
    assert a.search_sorted(b, side="left").to_list() == [0, 0, 2, 2, 4]
    assert a.search_sorted(b, side="right").to_list() == [0, 2, 2, 4, 4]


def test_logical_boolean() -> None:
    # note, cannot use expressions in logical
    # boolean context (eg: and/or/not operators)
    with pytest.raises(TypeError, match="ambiguous"):
        pl.col("colx") and pl.col("coly")  # type: ignore[redundant-expr]

    with pytest.raises(TypeError, match="ambiguous"):
        pl.col("colx") or pl.col("coly")  # type: ignore[redundant-expr]

    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

    with pytest.raises(TypeError, match="ambiguous"):
        df.select([(pl.col("a") > pl.col("b")) and (pl.col("b") > pl.col("b"))])

    with pytest.raises(TypeError, match="ambiguous"):
        df.select([(pl.col("a") > pl.col("b")) or (pl.col("b") > pl.col("b"))])


# https://github.com/pola-rs/polars/issues/4951
def test_ewm_with_multiple_chunks() -> None:
    df0 = pl.DataFrame(
        data=[
            ("w", 6.0, 1.0),
            ("x", 5.0, 2.0),
            ("y", 4.0, 3.0),
            ("z", 3.0, 4.0),
        ],
        schema=["a", "b", "c"],
    ).with_columns(
        [
            pl.col(pl.Float64).log().diff().name.prefix("ld_"),
        ]
    )
    assert df0.n_chunks() == 1

    # NOTE: We aren't testing whether `select` creates two chunks;
    # we just need two chunks to properly test `ewm_mean`
    df1 = df0.select(["ld_b", "ld_c"])
    assert df1.n_chunks() == 2

    ewm_std = df1.with_columns(
        pl.all().ewm_std(com=20).name.prefix("ewm_"),
    )
    assert ewm_std.null_count().sum_horizontal()[0] == 4


def test_lit_dtypes() -> None:
    def lit_series(value: Any, dtype: pl.PolarsDataType | None) -> pl.Series:
        return pl.select(pl.lit(value, dtype=dtype)).to_series()

    d = datetime(2049, 10, 5, 1, 2, 3, 987654)
    d_ms = datetime(2049, 10, 5, 1, 2, 3, 987000)
    d_tz = datetime(2049, 10, 5, 1, 2, 3, 987654, tzinfo=ZoneInfo("Asia/Kathmandu"))

    td = timedelta(days=942, hours=6, microseconds=123456)
    td_ms = timedelta(days=942, seconds=21600, microseconds=123000)

    df = pl.DataFrame(
        {
            "dtm_ms": lit_series(d, pl.Datetime("ms")),
            "dtm_us": lit_series(d, pl.Datetime("us")),
            "dtm_ns": lit_series(d, pl.Datetime("ns")),
            "dtm_aware_0": lit_series(d, pl.Datetime("us", "Asia/Kathmandu")),
            "dtm_aware_1": lit_series(d_tz, pl.Datetime("us")),
            "dtm_aware_2": lit_series(d_tz, None),
            "dtm_aware_3": lit_series(d, pl.Datetime(None, "Asia/Kathmandu")),
            "dur_ms": lit_series(td, pl.Duration("ms")),
            "dur_us": lit_series(td, pl.Duration("us")),
            "dur_ns": lit_series(td, pl.Duration("ns")),
            "f32": lit_series(0, pl.Float32),
            "u16": lit_series(0, pl.UInt16),
            "i16": lit_series(0, pl.Int16),
            "i64": lit_series(pl.Series([8]), None),
            "list_i64": lit_series(pl.Series([[1, 2, 3]]), None),
        }
    )
    assert df.dtypes == [
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Datetime("us", "Asia/Kathmandu"),
        pl.Datetime("us", "Asia/Kathmandu"),
        pl.Datetime("us", "Asia/Kathmandu"),
        pl.Datetime("us", "Asia/Kathmandu"),
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
        pl.Float32,
        pl.UInt16,
        pl.Int16,
        pl.Int64,
        pl.List(pl.Int64),
    ]
    assert df.row(0) == (
        d_ms,
        d,
        d,
        d_tz,
        d_tz,
        d_tz,
        d_tz,
        td_ms,
        td,
        td,
        0,
        0,
        0,
        8,
        [1, 2, 3],
    )


def test_lit_empty_tu() -> None:
    td = timedelta(1)
    assert pl.select(pl.lit(td, dtype=pl.Duration)).item() == td
    assert pl.select(pl.lit(td, dtype=pl.Duration)).dtypes[0].time_unit == "us"  # type: ignore[attr-defined]

    t = datetime(2023, 1, 1)
    assert pl.select(pl.lit(t, dtype=pl.Datetime)).item() == t
    assert pl.select(pl.lit(t, dtype=pl.Datetime)).dtypes[0].time_unit == "us"  # type: ignore[attr-defined]


def test_incompatible_lit_dtype() -> None:
    with pytest.raises(
        TypeError,
        match=r"time zone of dtype \('Asia/Kathmandu'\) differs from time zone of value \(datetime.timezone.utc\)",
    ):
        pl.lit(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            dtype=pl.Datetime("us", "Asia/Kathmandu"),
        )


def test_lit_dtype_utc() -> None:
    result = pl.select(
        pl.lit(
            datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu")),
            dtype=pl.Datetime("us", "Asia/Kathmandu"),
        )
    )
    expected = pl.DataFrame(
        {"literal": [datetime(2019, 12, 31, 18, 15, tzinfo=timezone.utc)]}
    ).select(pl.col("literal").dt.convert_time_zone("Asia/Kathmandu"))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (("a",), ["b", "c"]),
        (("a", "b"), ["c"]),
        ((["a", "b"],), ["c"]),
        ((pl.Int64,), ["c"]),
        ((pl.String, pl.Float32), ["a", "b"]),
        (([pl.String, pl.Float32],), ["a", "b"]),
    ],
)
def test_exclude(input: tuple[Any, ...], expected: list[str]) -> None:
    df = pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, "c": pl.String})
    assert df.select(pl.all().exclude(*input)).columns == expected


@pytest.mark.parametrize("input", [(5,), (["a"], date.today()), (pl.Int64, "a")])
def test_exclude_invalid_input(input: tuple[Any, ...]) -> None:
    df = pl.DataFrame(schema=["a", "b", "c"])
    with pytest.raises(TypeError):
        df.select(pl.all().exclude(*input))


def test_operators_vs_expressions() -> None:
    df = pl.DataFrame(
        data={
            "x": [5, 6, 7, 4, 8],
            "y": [1.5, 2.5, 1.0, 4.0, -5.75],
            "z": [-9, 2, -1, 4, 8],
        }
    )
    for c1, c2 in permutations("xyz", r=2):
        df_op = df.select(
            a=pl.col(c1) == pl.col(c2),
            b=pl.col(c1) // pl.col(c2),
            c=pl.col(c1) > pl.col(c2),
            d=pl.col(c1) >= pl.col(c2),
            e=pl.col(c1) < pl.col(c2),
            f=pl.col(c1) <= pl.col(c2),
            g=pl.col(c1) % pl.col(c2),
            h=pl.col(c1) != pl.col(c2),
            i=pl.col(c1) - pl.col(c2),
            j=pl.col(c1) / pl.col(c2),
            k=pl.col(c1) * pl.col(c2),
            l=pl.col(c1) + pl.col(c2),
        )
        df_expr = df.select(
            a=pl.col(c1).eq(pl.col(c2)),
            b=pl.col(c1).floordiv(pl.col(c2)),
            c=pl.col(c1).gt(pl.col(c2)),
            d=pl.col(c1).ge(pl.col(c2)),
            e=pl.col(c1).lt(pl.col(c2)),
            f=pl.col(c1).le(pl.col(c2)),
            g=pl.col(c1).mod(pl.col(c2)),
            h=pl.col(c1).ne(pl.col(c2)),
            i=pl.col(c1).sub(pl.col(c2)),
            j=pl.col(c1).truediv(pl.col(c2)),
            k=pl.col(c1).mul(pl.col(c2)),
            l=pl.col(c1).add(pl.col(c2)),
        )
        assert_frame_equal(df_op, df_expr)

    # xor - only int cols
    assert_frame_equal(
        df.select(pl.col("x") ^ pl.col("z")),
        df.select(pl.col("x").xor(pl.col("z"))),
    )

    # and (&) or (|) chains
    assert_frame_equal(
        df.select(
            all=(pl.col("x") >= pl.col("z")).and_(
                pl.col("y") >= pl.col("z"),
                pl.col("y") == pl.col("y"),
                pl.col("z") <= pl.col("x"),
                pl.col("y") != pl.col("x"),
            )
        ),
        df.select(
            all=(
                (pl.col("x") >= pl.col("z"))
                & (pl.col("y") >= pl.col("z"))
                & (pl.col("y") == pl.col("y"))
                & (pl.col("z") <= pl.col("x"))
                & (pl.col("y") != pl.col("x"))
            )
        ),
    )

    assert_frame_equal(
        df.select(
            any=(pl.col("x") == pl.col("y")).or_(
                pl.col("x") == pl.col("y"),
                pl.col("y") == pl.col("z"),
                pl.col("y").cast(int) == pl.col("z"),
            )
        ),
        df.select(
            any=(pl.col("x") == pl.col("y"))
            | (pl.col("x") == pl.col("y"))
            | (pl.col("y") == pl.col("z"))
            | (pl.col("y").cast(int) == pl.col("z"))
        ),
    )


def test_head() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert df.select(pl.col("a").head(0)).to_dict(as_series=False) == {"a": []}
    assert df.select(pl.col("a").head(3)).to_dict(as_series=False) == {"a": [1, 2, 3]}
    assert df.select(pl.col("a").head(10)).to_dict(as_series=False) == {
        "a": [1, 2, 3, 4, 5]
    }
    assert df.select(pl.col("a").head(pl.len() / 2)).to_dict(as_series=False) == {
        "a": [1, 2]
    }


def test_tail() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert df.select(pl.col("a").tail(0)).to_dict(as_series=False) == {"a": []}
    assert df.select(pl.col("a").tail(3)).to_dict(as_series=False) == {"a": [3, 4, 5]}
    assert df.select(pl.col("a").tail(10)).to_dict(as_series=False) == {
        "a": [1, 2, 3, 4, 5]
    }
    assert df.select(pl.col("a").tail(pl.len() / 2)).to_dict(as_series=False) == {
        "a": [4, 5]
    }


@pytest.mark.parametrize(
    ("const", "dtype"),
    [
        (1, pl.Int8),
        (4, pl.UInt32),
        (4.5, pl.Float32),
        (None, pl.Float64),
        ("白鵬翔", pl.String),
        (date.today(), pl.Date),
        (datetime.now(), pl.Datetime("ns")),
        (time(23, 59, 59), pl.Time),
        (timedelta(hours=7, seconds=123), pl.Duration("ms")),
    ],
)
def test_extend_constant(const: Any, dtype: pl.PolarsDataType) -> None:
    df = pl.DataFrame({"a": pl.Series("s", [None], dtype=dtype)})

    expected = pl.DataFrame(
        {"a": pl.Series("s", [None, const, const, const], dtype=dtype)}
    )

    assert_frame_equal(df.select(pl.col("a").extend_constant(const, 3)), expected)


@pytest.mark.parametrize(
    ("const", "dtype"),
    [
        (1, pl.Int8),
        (4, pl.UInt32),
        (4.5, pl.Float32),
        (None, pl.Float64),
        ("白鵬翔", pl.String),
        (date.today(), pl.Date),
        (datetime.now(), pl.Datetime("ns")),
        (time(23, 59, 59), pl.Time),
        (timedelta(hours=7, seconds=123), pl.Duration("ms")),
    ],
)
def test_extend_constant_arr(const: Any, dtype: pl.PolarsDataType) -> None:
    """
    Test extend_constant in pl.List array.

    NOTE: This function currently fails when the Series is a list with a single [None]
          value. Hence, this function does not begin with [[None]], but [[const]].
    """
    s = pl.Series("s", [[const]], dtype=pl.List(dtype))

    expected = pl.Series("s", [[const, const, const, const]], dtype=pl.List(dtype))

    assert_series_equal(s.list.eval(pl.element().extend_constant(const, 3)), expected)


def test_is_not_deprecated() -> None:
    df = pl.DataFrame({"a": [True, False, True]})

    with pytest.deprecated_call():
        expr = pl.col("a").is_not()
    result = df.select(expr)

    expected = pl.DataFrame({"a": [False, True, False]})
    assert_frame_equal(result, expected)


def test_repr_short_expression() -> None:
    expr = pl.functions.all().len().name.prefix("length:")
    # we cut off the last ten characters because that includes the
    # memory location which will vary between runs
    result = repr(expr).split("0x")[0]

    expected = "<Expr ['.rename_alias(*.count())'] at "
    assert result == expected


def test_repr_long_expression() -> None:
    expr = pl.functions.col(pl.String).str.count_matches("")

    # we cut off the last ten characters because that includes the
    # memory location which will vary between runs
    result = repr(expr).split("0x")[0]

    # note the … denoting that there was truncated text
    expected = "<Expr ['dtype_columns([String]).str.co…'] at "
    assert result == expected
    assert repr(expr).endswith(">")


def test_repr_gather() -> None:
    result = repr(pl.col("a").gather(0))
    assert 'col("a").gather(0)' in result
    result = repr(pl.col("a").get(0))
    assert 'col("a").get(0)' in result


def test_replace_no_cse() -> None:
    plan = (
        pl.LazyFrame({"a": [1], "b": [2]})
        .select([(pl.col("a") * pl.col("a")).sum().replace(1, None)])
        .explain()
    )
    assert "POLARS_CSER" not in plan

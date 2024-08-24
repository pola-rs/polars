from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

import polars as pl
from polars.testing import assert_frame_equal


def test_sort_by_bools() -> None:
    # tests dispatch
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    out = df.with_columns((pl.col("foo") % 2 == 1).alias("foo_odd")).sort(
        by=["foo_odd", "foo"]
    )
    assert out.rows() == [
        (2, 7.0, "b", False),
        (1, 6.0, "a", True),
        (3, 8.0, "c", True),
    ]
    assert out.shape == (3, 4)


def test_repeat_expansion_in_group_by() -> None:
    out = (
        pl.DataFrame({"g": [1, 2, 2, 3, 3, 3]})
        .group_by("g", maintain_order=True)
        .agg(pl.repeat(1, pl.len()).cum_sum())
        .to_dict(as_series=False)
    )
    assert out == {"g": [1, 2, 3], "repeat": [[1], [1, 2], [1, 2, 3]]}


def test_agg_after_head() -> None:
    a = [1, 1, 1, 2, 2, 3, 3, 3, 3]

    df = pl.DataFrame({"a": a, "b": pl.arange(1, len(a) + 1, eager=True)})

    expected = pl.DataFrame({"a": [1, 2, 3], "b": [6, 9, 21]})

    for maintain_order in [True, False]:
        out = df.group_by("a", maintain_order=maintain_order).agg(
            [pl.col("b").head(3).sum()]
        )

        if not maintain_order:
            out = out.sort("a")

        assert_frame_equal(out, expected)


def test_overflow_uint16_agg_mean() -> None:
    assert (
        pl.DataFrame(
            {
                "col1": ["A" for _ in range(1025)],
                "col3": [64 for _ in range(1025)],
            }
        )
        .with_columns(pl.col("col3").cast(pl.UInt16))
        .group_by(["col1"])
        .agg(pl.col("col3").mean())
        .to_dict(as_series=False)
    ) == {"col1": ["A"], "col3": [64.0]}


def test_binary_on_list_agg_3345() -> None:
    df = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "B"],
            "id": [1, 2, 1, 4, 5, 4, 6],
        }
    )

    assert (
        df.group_by(["group"], maintain_order=True)
        .agg(
            [
                (
                    (pl.col("id").unique_counts() / pl.col("id").len()).log()
                    * -1
                    * (pl.col("id").unique_counts() / pl.col("id").len())
                ).sum()
            ]
        )
        .to_dict(as_series=False)
    ) == {"group": ["A", "B"], "id": [0.6365141682948128, 1.0397207708399179]}


def test_maintain_order_after_sampling() -> None:
    # internally samples cardinality
    # check if the maintain_order kwarg is dispatched
    df = pl.DataFrame(
        {
            "type": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "value": [1, 3, 2, 3, 4, 5, 3, 4],
        }
    )

    result = df.group_by("type", maintain_order=True).agg(pl.col("value").sum())
    expected = {"type": ["A", "B", "C", "D"], "value": [5, 8, 5, 7]}
    assert result.to_dict(as_series=False) == expected


def test_sorted_group_by_optimization() -> None:
    df = pl.DataFrame({"a": np.random.randint(0, 5, 20)})

    # the sorted optimization should not randomize the
    # groups, so this is tests that we hit the sorted optimization
    for descending in [True, False]:
        sorted_implicit = (
            df.with_columns(pl.col("a").sort(descending=descending))
            .group_by("a")
            .agg(pl.len())
        )
        sorted_explicit = (
            df.group_by("a").agg(pl.len()).sort("a", descending=descending)
        )
        assert_frame_equal(sorted_explicit, sorted_implicit)


def test_median_on_shifted_col_3522() -> None:
    df = pl.DataFrame(
        {
            "foo": [
                datetime(2022, 5, 5, 12, 31, 34),
                datetime(2022, 5, 5, 12, 47, 1),
                datetime(2022, 5, 6, 8, 59, 11),
            ]
        }
    )
    diffs = df.select(pl.col("foo").diff().dt.total_seconds())
    assert diffs.select(pl.col("foo").median()).to_series()[0] == 36828.5


def test_group_by_agg_equals_zero_3535() -> None:
    # setup test frame
    df = pl.DataFrame(
        data=[
            # note: the 'bb'-keyed values should clearly sum to 0
            ("aa", 10, None),
            ("bb", -10, 0.5),
            ("bb", 10, -0.5),
            ("cc", -99, 10.5),
            ("cc", None, 0.0),
        ],
        schema=[
            ("key", pl.String),
            ("val1", pl.Int16),
            ("val2", pl.Float32),
        ],
        orient="row",
    )
    # group by the key, aggregating the two numeric cols
    assert df.group_by(pl.col("key"), maintain_order=True).agg(
        [pl.col("val1").sum(), pl.col("val2").sum()]
    ).to_dict(as_series=False) == {
        "key": ["aa", "bb", "cc"],
        "val1": [10, 0, -99],
        "val2": [0.0, 0.0, 10.5],
    }


def test_dtype_concat_3735() -> None:
    for dt in [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]:
        d1 = pl.DataFrame([pl.Series("val", [1, 2], dtype=dt)])

    d2 = pl.DataFrame([pl.Series("val", [3, 4], dtype=dt)])
    df = pl.concat([d1, d2])

    assert df.shape == (4, 1)
    assert df.columns == ["val"]
    assert df.to_series().to_list() == [1, 2, 3, 4]


def test_opaque_filter_on_lists_3784() -> None:
    df = pl.DataFrame(
        {"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]}
    ).lazy()
    df = df.with_columns(pl.col("str").cast(pl.Categorical))

    df_groups = df.group_by("group").agg([pl.col("str").alias("str_list")])

    pre = "A"
    succ = "B"

    assert (
        df_groups.filter(
            pl.col("str_list").map_elements(
                lambda variant: pre in variant
                and succ in variant
                and variant.to_list().index(pre) < variant.to_list().index(succ)
            )
        )
    ).collect().to_dict(as_series=False) == {
        "group": [1],
        "str_list": [["A", "B", "B"]],
    }


def test_ternary_none_struct() -> None:
    ignore_nulls = False

    def map_expr(name: str) -> pl.Expr:
        return (
            pl.when(ignore_nulls or pl.col(name).null_count() == 0)
            .then(
                pl.struct(
                    [
                        pl.sum(name).alias("sum"),
                        (pl.len() - pl.col(name).null_count()).alias("count"),
                    ]
                ),
            )
            .otherwise(None)
        ).alias("out")

    assert (
        pl.DataFrame({"groups": [1, 2, 3, 4], "values": [None, None, 1, 2]})
        .group_by("groups", maintain_order=True)
        .agg([map_expr("values")])
    ).to_dict(as_series=False) == {
        "groups": [1, 2, 3, 4],
        "out": [
            None,
            None,
            {"sum": 1, "count": 1},
            {"sum": 2, "count": 1},
        ],
    }


def test_edge_cast_string_duplicates_4259() -> None:
    # carefully constructed data.
    # note that row 2, 3 concatenated are the same string ('5461214484')
    df = pl.DataFrame(
        {
            "a": [99, 54612, 546121],
            "b": [1, 14484, 4484],
        }
    ).with_columns(pl.all().cast(pl.String))

    mask = df.select(["a", "b"]).is_duplicated()
    df_filtered = df.filter(pl.lit(mask))

    assert df_filtered.shape == (0, 2)
    assert df_filtered.rows() == []


def test_query_4438() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1]})

    q = (
        df.lazy()
        .with_columns(pl.col("x").rolling_max(window_size=3).alias("rolling_max"))
        .fill_null(strategy="backward")
        .with_columns(
            pl.col("rolling_max").rolling_max(window_size=3).alias("rolling_max_2")
        )
    )
    assert q.collect()["rolling_max_2"].to_list() == [
        None,
        None,
        3,
        10,
        10,
        10,
        10,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
    ]


def test_query_4538() -> None:
    df = pl.DataFrame(
        [
            pl.Series("value", ["aaa", "bbb"]),
        ]
    )
    assert df.select([pl.col("value").str.to_uppercase().is_in(["AAA"])])[
        "value"
    ].to_list() == [True, False]


def test_none_comparison_4773() -> None:
    df = pl.DataFrame(
        {
            "x": [0, 1, None, 2],
            "y": [1, 2, None, 3],
        }
    ).filter(pl.col("x") != pl.col("y"))
    assert df.shape == (3, 2)
    assert df.rows() == [(0, 1), (1, 2), (2, 3)]


def test_datetime_supertype_5236() -> None:
    df = pd.DataFrame(
        {
            "StartDateTime": [pd.Timestamp.now(tz="UTC"), pd.Timestamp.now(tz="UTC")],
            "EndDateTime": [pd.Timestamp.now(tz="UTC"), pd.Timestamp.now(tz="UTC")],
        }
    )
    out = pl.from_pandas(df).filter(
        pl.col("StartDateTime")
        < (pl.col("EndDateTime").dt.truncate("1d").max() - timedelta(days=1))
    )
    assert out.shape == (0, 2)
    assert out.dtypes == [pl.Datetime("ns", "UTC")] * 2


def test_shift_drop_nulls_10875() -> None:
    assert pl.LazyFrame({"a": [1, 2, 3]}).shift(1).drop_nulls().collect()[
        "a"
    ].to_list() == [1, 2]


def test_temporal_downcasts() -> None:
    s = pl.Series([-1, 0, 1]).cast(pl.Datetime("us"))

    assert s.to_list() == [
        datetime(1969, 12, 31, 23, 59, 59, 999999),
        datetime(1970, 1, 1),
        datetime(1970, 1, 1, 0, 0, 0, 1),
    ]

    # downcast (from us to ms, or from datetime to date) should NOT change the date
    for s_dt in (s.dt.date(), s.cast(pl.Date)):
        assert s_dt.to_list() == [
            date(1969, 12, 31),
            date(1970, 1, 1),
            date(1970, 1, 1),
        ]
    assert s.cast(pl.Datetime("ms")).to_list() == [
        datetime(1969, 12, 31, 23, 59, 59, 999000),
        datetime(1970, 1, 1),
        datetime(1970, 1, 1),
    ]


def test_temporal_time_casts() -> None:
    s = pl.Series([-1, 0, 1]).cast(pl.Datetime("us"))

    for s_dt in (s.dt.time(), s.cast(pl.Time)):
        assert s_dt.to_list() == [
            time(23, 59, 59, 999999),
            time(0, 0, 0, 0),
            time(0, 0, 0, 1),
        ]

from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Any

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars import ComputeError
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
        .agg(pl.repeat(1, pl.count()).cumsum())
        .to_dict(False)
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
                "col3": [64 for i in range(1025)],
            }
        )
        .with_columns(
            [
                pl.col("col3").cast(pl.UInt16),
            ]
        )
        .group_by(["col1"])
        .agg(pl.col("col3").mean())
        .to_dict(False)
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
        .to_dict(False)
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
    assert df.group_by("type", maintain_order=True).agg(pl.col("value").sum()).to_dict(
        False
    ) == {"type": ["A", "B", "C", "D"], "value": [5, 8, 5, 7]}


def test_sorted_group_by_optimization(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_NO_STREAMING_GROUPBY", "1")

    df = pl.DataFrame({"a": np.random.randint(0, 5, 20)})

    # the sorted optimization should not randomize the
    # groups, so this is tests that we hit the sorted optimization
    for descending in [True, False]:
        sorted_implicit = (
            df.with_columns(pl.col("a").sort(descending=descending))
            .group_by("a")
            .agg(pl.count())
        )
        sorted_explicit = (
            df.group_by("a").agg(pl.count()).sort("a", descending=descending)
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
    diffs = df.select(pl.col("foo").diff().dt.seconds())
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
            ("key", pl.Utf8),
            ("val1", pl.Int16),
            ("val2", pl.Float32),
        ],
    )
    # group by the key, aggregating the two numeric cols
    assert df.group_by(pl.col("key"), maintain_order=True).agg(
        [pl.col("val1").sum(), pl.col("val2").sum()]
    ).to_dict(False) == {
        "key": ["aa", "bb", "cc"],
        "val1": [10, 0, -99],
        "val2": [0.0, 0.0, 10.5],
    }


def test_arithmetic_in_aggregation_3739() -> None:
    def demean_dot() -> pl.Expr:
        x = pl.col("x")
        y = pl.col("y")
        x1 = x - x.mean()
        y1 = y - y.mean()
        return (x1 * y1).sum().alias("demean_dot")

    assert (
        pl.DataFrame(
            {
                "key": ["a", "a", "a", "a"],
                "x": [4, 2, 2, 4],
                "y": [2, 0, 2, 0],
            }
        )
        .group_by("key")
        .agg(
            [
                demean_dot(),
            ]
        )
    ).to_dict(False) == {"key": ["a"], "demean_dot": [0.0]}


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
    ).collect().to_dict(False) == {"group": [1], "str_list": [["A", "B", "B"]]}


def test_ternary_none_struct() -> None:
    ignore_nulls = False

    def map_expr(name: str) -> pl.Expr:
        return (
            pl.when(ignore_nulls or pl.col(name).null_count() == 0)
            .then(
                pl.struct(
                    [
                        pl.sum(name).alias("sum"),
                        (pl.count() - pl.col(name).null_count()).alias("count"),
                    ]
                ),
            )
            .otherwise(None)
        ).alias("out")

    assert (
        pl.DataFrame({"groups": [1, 2, 3, 4], "values": [None, None, 1, 2]})
        .group_by("groups", maintain_order=True)
        .agg([map_expr("values")])
    ).to_dict(False) == {
        "groups": [1, 2, 3, 4],
        "out": [
            {"sum": None, "count": None},
            {"sum": None, "count": None},
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
    ).with_columns(pl.all().cast(pl.Utf8))

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
            "StartDateTime": [
                pd.Timestamp(datetime.utcnow(), tz="UTC"),
                pd.Timestamp(datetime.utcnow(), tz="UTC"),
            ],
            "EndDateTime": [
                pd.Timestamp(datetime.utcnow(), tz="UTC"),
                pd.Timestamp(datetime.utcnow(), tz="UTC"),
            ],
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


def test_utf8_date() -> None:
    df = pl.DataFrame({"x1": ["2021-01-01"]}).with_columns(
        **{"x1-date": pl.col("x1").cast(pl.Date)}
    )
    expected = pl.DataFrame({"x1-date":[date(2021,1,1)]})
    out = df.select(pl.col("x1-date"))
    assert out.shape == (1, 1)
    assert out.dtypes == [pl.Date]
    assert_frame_equal(expected, out)


def test_wrong_utf8_date() -> None:
    df = pl.DataFrame({"x1": ["2021-01-aa"]})

    with pytest.raises(ComputeError):
        df.with_columns(**{"x1-date": pl.col("x1").cast(pl.Date)})


def test_utf8_datetime() -> None:
    df = pl.DataFrame(
        {"x1": ["2021-12-19T00:39:57", "2022-12-19T16:39:57"]}
    ).with_columns(
        **{
            "x1-datetime-ns": pl.col("x1").cast(pl.Datetime(time_unit="ns")),
            "x1-datetime-ms": pl.col("x1").cast(pl.Datetime(time_unit="ms")),
            "x1-datetime-us": pl.col("x1").cast(pl.Datetime(time_unit="us")),
        }
    )
    first_row = datetime(year=2021, month=12, day=19, hour=00, minute=39, second=57)
    second_row = datetime(year=2022, month=12, day=19, hour=16, minute=39, second=57)
    expected = pl.DataFrame(
        {
            "x1-datetime-ns": [first_row, second_row],
            "x1-datetime-ms": [first_row, second_row],
            "x1-datetime-us": [first_row, second_row]
        }
    ).select(
        pl.col("x1-datetime-ns").dt.cast_time_unit("ns"),
        pl.col("x1-datetime-ms").dt.cast_time_unit("ms"),
        pl.col("x1-datetime-us").dt.cast_time_unit("us"),
    )

    out = df.select(
        pl.col("x1-datetime-ns"), pl.col("x1-datetime-ms"), pl.col("x1-datetime-us")
    )
    assert out.shape == (2, 3)
    assert out.dtypes == [pl.Datetime, pl.Datetime, pl.Datetime]
    assert_frame_equal(expected, out)


def test_wrong_utf8_datetime() -> None:
    df = pl.DataFrame({"x1": ["2021-12-19 00:39:57", "2022-12-19 16:39:57"]})
    with pytest.raises(ComputeError):
        df.with_columns(
            **{"x1-datetime-ns": pl.col("x1").cast(pl.Datetime(time_unit="ns"))}
        )


def test_utf8_datetime_timezone() -> None:
    ccs_tz = "America/Caracas"
    stg_tz = "America/Santiago"
    utc_tz = "UTC"
    df = pl.DataFrame(
        {"x1": ["1996-12-19T16:39:57 +00:00", "2022-12-19T00:39:57 +00:00"]}
    ).with_columns(
        **{
            "x1-datetime-ns": pl.col("x1").cast(
                pl.Datetime(time_unit="ns", time_zone=ccs_tz)
            ),
            "x1-datetime-ms": pl.col("x1").cast(
                pl.Datetime(time_unit="ms", time_zone=stg_tz)
            ),
            "x1-datetime-us": pl.col("x1").cast(
                pl.Datetime(time_unit="us", time_zone=utc_tz)
            ),
        }
    )

    expected = pl.DataFrame(
        {
            "x1-datetime-ns": [
                datetime(year=1996, month=12, day=19, hour=12, minute=39, second=57),
                datetime(year=2022, month=12, day=18, hour=20, minute=39, second=57),
            ],
            "x1-datetime-ms": [
                datetime(year=1996, month=12, day=19, hour=13, minute=39, second=57),
                datetime(year=2022, month=12, day=18, hour=21, minute=39, second=57),
            ],
            "x1-datetime-us": [
                datetime(year=1996, month=12, day=19, hour=16, minute=39, second=57),
                datetime(year=2022, month=12, day=19, hour=00, minute=39, second=57),
            ],
        }
    ).select(
        pl.col("x1-datetime-ns").dt.cast_time_unit("ns").dt.replace_time_zone(ccs_tz),
        pl.col("x1-datetime-ms").dt.cast_time_unit("ms").dt.replace_time_zone(stg_tz),
        pl.col("x1-datetime-us").dt.cast_time_unit("us").dt.replace_time_zone(utc_tz),
    )

    out = df.select(
        pl.col("x1-datetime-ns"), pl.col("x1-datetime-ms"), pl.col("x1-datetime-us")
    )

    assert out.shape == (2, 3)
    assert out.dtypes == [
        pl.Datetime("ns", "America/Caracas"),
        pl.Datetime("ms", "America/Santiago"),
        pl.Datetime("us", "UTC"),
    ]
    assert_frame_equal(expected, out)

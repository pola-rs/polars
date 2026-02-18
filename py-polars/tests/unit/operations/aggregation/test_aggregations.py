from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric import dataframes

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import numpy.typing as npt

    from polars._typing import PolarsDataType, TimeUnit


def test_quantile_expr_input() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0.0, 0.0, 0.3, 0.2, 0.0]})

    assert_frame_equal(
        df.select([pl.col("a").quantile(pl.col("b").sum() + 0.1)]),
        df.select(pl.col("a").quantile(0.6)),
    )

    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [0.25, 0.3, 0.4, 0.75]})

    assert_frame_equal(
        df.select(
            pl.col.x.quantile(pl.concat_list(pl.col.y.min(), pl.col.y.max().first()))
        ),
        df.select(pl.col.x.quantile([0.25, 0.75])),
    )


def test_boolean_aggs() -> None:
    df = pl.DataFrame({"bool": [True, False, None, True]})

    aggs = [
        pl.mean("bool").alias("mean"),
        pl.std("bool").alias("std"),
        pl.var("bool").alias("var"),
    ]
    assert df.select(aggs).to_dict(as_series=False) == {
        "mean": [0.6666666666666666],
        "std": [0.5773502691896258],
        "var": [0.33333333333333337],
    }

    assert df.group_by(pl.lit(1)).agg(aggs).to_dict(as_series=False) == {
        "literal": [1],
        "mean": [0.6666666666666666],
        "std": [0.5773502691896258],
        "var": [0.33333333333333337],
    }


def test_duration_aggs() -> None:
    df = pl.DataFrame(
        {
            "time1": pl.datetime_range(
                start=datetime(2022, 12, 12),
                end=datetime(2022, 12, 18),
                interval="1d",
                eager=True,
            ),
            "time2": pl.datetime_range(
                start=datetime(2023, 1, 12),
                end=datetime(2023, 1, 18),
                interval="1d",
                eager=True,
            ),
        }
    )

    df = df.with_columns((pl.col("time2") - pl.col("time1")).alias("time_difference"))

    assert df.select("time_difference").mean().to_dict(as_series=False) == {
        "time_difference": [timedelta(days=31)]
    }
    assert df.group_by(pl.lit(1)).agg(pl.mean("time_difference")).to_dict(
        as_series=False
    ) == {
        "literal": [1],
        "time_difference": [timedelta(days=31)],
    }


def test_list_aggregation_that_filters_all_data_6017() -> None:
    out = (
        pl.DataFrame({"col_to_group_by": [2], "flt": [1672740910.967138], "col3": [1]})
        .group_by("col_to_group_by")
        .agg((pl.col("flt").filter(col3=0).diff() * 1000).diff().alias("calc"))
    )

    assert out.schema == {"col_to_group_by": pl.Int64, "calc": pl.List(pl.Float64)}
    assert out.to_dict(as_series=False) == {"col_to_group_by": [2], "calc": [[]]}


def test_median() -> None:
    s = pl.Series([1, 2, 3])
    assert s.median() == 2


def test_single_element_std() -> None:
    s = pl.Series([1])
    assert s.std(ddof=1) is None
    assert s.std(ddof=0) == 0.0


def test_quantile() -> None:
    s = pl.Series([1, 2, 3])
    assert s.quantile(0.5, "nearest") == 2
    assert s.quantile(0.5, "lower") == 2
    assert s.quantile(0.5, "higher") == 2
    assert s.quantile([0.25, 0.75], "linear") == [1.5, 2.5]

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    expected = pl.DataFrame({"a": [[2.0]]})
    assert_frame_equal(
        df.select(pl.col("a").quantile([0.5], interpolation="linear")), expected
    )


def test_quantile_error_checking() -> None:
    s = pl.Series([1, 2, 3])
    with pytest.raises(pl.exceptions.ComputeError):
        s.quantile(-0.1)
    with pytest.raises(pl.exceptions.ComputeError):
        s.quantile(1.1)
    with pytest.raises(pl.exceptions.ComputeError):
        s.quantile([0.0, 1.2])


def test_quantile_date() -> None:
    s = pl.Series(
        "a", [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)]
    )
    assert s.quantile(0.5, "nearest") == datetime(2025, 1, 3)
    assert s.quantile(0.5, "lower") == datetime(2025, 1, 2)
    assert s.quantile(0.5, "higher") == datetime(2025, 1, 3)
    assert s.quantile(0.5, "linear") == datetime(2025, 1, 2, 12)

    df = s.to_frame().lazy()
    result = df.select(
        nearest=pl.col("a").quantile(0.5, "nearest"),
        lower=pl.col("a").quantile(0.5, "lower"),
        higher=pl.col("a").quantile(0.5, "higher"),
        linear=pl.col("a").quantile(0.5, "linear"),
    )
    dt = pl.Datetime("us")
    assert result.collect_schema() == pl.Schema(
        {
            "nearest": dt,
            "lower": dt,
            "higher": dt,
            "linear": dt,
        }
    )
    expected = pl.DataFrame(
        {
            "nearest": pl.Series([datetime(2025, 1, 3)], dtype=dt),
            "lower": pl.Series([datetime(2025, 1, 2)], dtype=dt),
            "higher": pl.Series([datetime(2025, 1, 3)], dtype=dt),
            "linear": pl.Series([datetime(2025, 1, 2, 12)], dtype=dt),
        }
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "UTC"])
def test_quantile_datetime(tu: TimeUnit, tz: str) -> None:
    time_zone = ZoneInfo(tz) if tz else None
    dt = pl.Datetime(tu, time_zone)

    s = pl.Series(
        "a",
        [
            datetime(2025, 1, 1, tzinfo=time_zone),
            datetime(2025, 1, 2, tzinfo=time_zone),
            datetime(2025, 1, 3, tzinfo=time_zone),
            datetime(2025, 1, 4, tzinfo=time_zone),
        ],
        dtype=dt,
    )
    assert s.quantile(0.5, "nearest") == datetime(2025, 1, 3, tzinfo=time_zone)
    assert s.quantile(0.5, "lower") == datetime(2025, 1, 2, tzinfo=time_zone)
    assert s.quantile(0.5, "higher") == datetime(2025, 1, 3, tzinfo=time_zone)
    assert s.quantile(0.5, "linear") == datetime(2025, 1, 2, 12, tzinfo=time_zone)

    df = s.to_frame().lazy()
    result = df.select(
        nearest=pl.col("a").quantile(0.5, "nearest"),
        lower=pl.col("a").quantile(0.5, "lower"),
        higher=pl.col("a").quantile(0.5, "higher"),
        linear=pl.col("a").quantile(0.5, "linear"),
    )
    assert result.collect_schema() == pl.Schema(
        {
            "nearest": dt,
            "lower": dt,
            "higher": dt,
            "linear": dt,
        }
    )
    expected = pl.DataFrame(
        {
            "nearest": pl.Series([datetime(2025, 1, 3, tzinfo=time_zone)], dtype=dt),
            "lower": pl.Series([datetime(2025, 1, 2, tzinfo=time_zone)], dtype=dt),
            "higher": pl.Series([datetime(2025, 1, 3, tzinfo=time_zone)], dtype=dt),
            "linear": pl.Series([datetime(2025, 1, 2, 12, tzinfo=time_zone)], dtype=dt),
        }
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
def test_quantile_duration(tu: TimeUnit) -> None:
    dt = pl.Duration(tu)

    s = pl.Series(
        "a",
        [timedelta(days=1), timedelta(days=2), timedelta(days=3), timedelta(days=4)],
        dtype=dt,
    )
    assert s.quantile(0.5, "nearest") == timedelta(days=3)
    assert s.quantile(0.5, "lower") == timedelta(days=2)
    assert s.quantile(0.5, "higher") == timedelta(days=3)
    assert s.quantile(0.5, "linear") == timedelta(days=2, hours=12)

    df = s.to_frame().lazy()
    result = df.select(
        nearest=pl.col("a").quantile(0.5, "nearest"),
        lower=pl.col("a").quantile(0.5, "lower"),
        higher=pl.col("a").quantile(0.5, "higher"),
        linear=pl.col("a").quantile(0.5, "linear"),
    )
    assert result.collect_schema() == pl.Schema(
        {
            "nearest": dt,
            "lower": dt,
            "higher": dt,
            "linear": dt,
        }
    )
    expected = pl.DataFrame(
        {
            "nearest": pl.Series([timedelta(days=3)], dtype=dt),
            "lower": pl.Series([timedelta(days=2)], dtype=dt),
            "higher": pl.Series([timedelta(days=3)], dtype=dt),
            "linear": pl.Series([timedelta(days=2, hours=12)], dtype=dt),
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_quantile_time() -> None:
    s = pl.Series("a", [time(hour=1), time(hour=2), time(hour=3), time(hour=4)])
    assert s.quantile(0.5, "nearest") == time(hour=3)
    assert s.quantile(0.5, "lower") == time(hour=2)
    assert s.quantile(0.5, "higher") == time(hour=3)
    assert s.quantile(0.5, "linear") == time(hour=2, minute=30)

    df = s.to_frame().lazy()
    result = df.select(
        nearest=pl.col("a").quantile(0.5, "nearest"),
        lower=pl.col("a").quantile(0.5, "lower"),
        higher=pl.col("a").quantile(0.5, "higher"),
        linear=pl.col("a").quantile(0.5, "linear"),
    )
    assert result.collect_schema() == pl.Schema(
        {
            "nearest": pl.Time,
            "lower": pl.Time,
            "higher": pl.Time,
            "linear": pl.Time,
        }
    )
    expected = pl.DataFrame(
        {
            "nearest": pl.Series([time(hour=3)]),
            "lower": pl.Series([time(hour=2)]),
            "higher": pl.Series([time(hour=3)]),
            "linear": pl.Series([time(hour=2, minute=30)]),
        }
    )
    assert_frame_equal(result.collect(), expected)


@pytest.mark.slow
@pytest.mark.parametrize("tp", [int, float])
@pytest.mark.parametrize("n", [1, 2, 10, 100])
def test_quantile_vs_numpy(tp: type, n: int) -> None:
    a: np.ndarray[Any, Any] = np.random.randint(0, 50, n).astype(tp)
    np_result: npt.ArrayLike | None = np.median(a)
    # nan check
    if np_result != np_result:
        np_result = None
    median = pl.Series(a).median()
    if median is not None:
        assert np.isclose(median, np_result)  # type: ignore[arg-type]
    else:
        assert np_result is None

    q = np.random.sample()
    try:
        np_result = np.quantile(a, q)
    except IndexError:
        np_result = None
    if np_result:
        # nan check
        if np_result != np_result:
            np_result = None
        assert np.isclose(
            pl.Series(a).quantile(q, interpolation="linear"),  # type: ignore[arg-type]
            np_result,  # type: ignore[arg-type]
        )

    df = pl.DataFrame({"a": a})

    expected = df.select(
        pl.col.a.quantile(0.25).alias("low"), pl.col.a.quantile(0.75).alias("high")
    ).select(pl.concat_list(["low", "high"]).alias("quantiles"))

    result = df.select(pl.col.a.quantile([0.25, 0.75]).alias("quantiles"))

    assert_frame_equal(expected, result)


def test_mean_overflow() -> None:
    assert np.isclose(
        pl.Series([9_223_372_036_854_775_800, 100]).mean(),  # type: ignore[arg-type]
        4.611686018427388e18,
    )


def test_mean_null_simd() -> None:
    for dtype in [int, float]:
        df = (
            pl.Series(np.random.randint(0, 100, 1000))
            .cast(dtype)
            .to_frame("a")
            .select(pl.when(pl.col("a") > 40).then(pl.col("a")))
        )

        s = df["a"]
        assert s.mean() == s.to_pandas().mean()


def test_literal_group_agg_chunked_7968() -> None:
    df = pl.DataFrame({"A": [1, 1], "B": [1, 3]})
    ser = pl.concat([pl.Series([3]), pl.Series([4, 5])], rechunk=False)

    assert_frame_equal(
        df.group_by("A").agg(pl.col("B").search_sorted(ser)),
        pl.DataFrame(
            [
                pl.Series("A", [1], dtype=pl.Int64),
                pl.Series("B", [[1, 2, 2]], dtype=pl.List(pl.get_index_type())),
            ]
        ),
    )


def test_duration_function_literal() -> None:
    df = pl.DataFrame(
        {
            "A": ["x", "x", "y", "y", "y"],
            "T": pl.datetime_range(
                date(2022, 1, 1), date(2022, 5, 1), interval="1mo", eager=True
            ),
            "S": [1, 2, 4, 8, 16],
        }
    )

    result = df.group_by("A", maintain_order=True).agg(
        (pl.col("T").max() + pl.duration(seconds=1)) - pl.col("T")
    )

    # this checks if the `pl.duration` is flagged as AggState::Literal
    expected = pl.DataFrame(
        {
            "A": ["x", "y"],
            "T": [
                [timedelta(days=31, seconds=1), timedelta(seconds=1)],
                [
                    timedelta(days=61, seconds=1),
                    timedelta(days=30, seconds=1),
                    timedelta(seconds=1),
                ],
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_string_par_materialize_8207() -> None:
    df = pl.LazyFrame(
        {
            "a": ["a", "b", "d", "c", "e"],
            "b": ["P", "L", "R", "T", "a long string"],
        }
    )

    assert df.group_by(["a"]).agg(pl.min("b")).sort("a").collect().to_dict(
        as_series=False
    ) == {
        "a": ["a", "b", "c", "d", "e"],
        "b": ["P", "L", "T", "R", "a long string"],
    }


def test_online_variance() -> None:
    df = pl.DataFrame(
        {
            "id": [1] * 5,
            "no_nulls": [1, 2, 3, 4, 5],
            "nulls": [1, None, 3, None, 5],
        }
    )

    assert_frame_equal(
        df.group_by("id")
        .agg(pl.all().exclude("id").std())
        .select(["no_nulls", "nulls"]),
        df.select(pl.all().exclude("id").std()),
    )


def test_implode_and_agg() -> None:
    df = pl.DataFrame({"type": ["water", "fire", "water", "earth"]})

    assert_frame_equal(
        df.group_by("type").agg(pl.col("type").implode().first().alias("foo")),
        pl.DataFrame(
            {
                "type": ["water", "fire", "earth"],
                "foo": [["water", "water"], ["fire"], ["earth"]],
            }
        ),
        check_row_order=False,
    )

    # implode + function should be allowed in group_by
    assert df.group_by("type", maintain_order=True).agg(
        pl.col("type").implode().list.head().alias("foo")
    ).to_dict(as_series=False) == {
        "type": ["water", "fire", "earth"],
        "foo": [["water", "water"], ["fire"], ["earth"]],
    }
    assert df.select(pl.col("type").implode().list.head(1).over("type")).to_dict(
        as_series=False
    ) == {"type": [["water"], ["fire"], ["water"], ["earth"]]}


def test_mapped_literal_to_literal_9217() -> None:
    df = pl.DataFrame({"unique_id": ["a", "b"]})
    assert df.group_by(True).agg(
        pl.struct(pl.lit("unique_id").alias("unique_id"))
    ).to_dict(as_series=False) == {
        "literal": [True],
        "unique_id": [{"unique_id": "unique_id"}],
    }


def test_sum_empty_and_null_set() -> None:
    series = pl.Series("a", [], dtype=pl.Float32)
    assert series.sum() == 0

    series = pl.Series("a", [None], dtype=pl.Float32)
    assert series.sum() == 0

    df = pl.DataFrame(
        {"a": [None, None, None], "b": [1, 1, 1]},
        schema={"a": pl.Float32, "b": pl.Int64},
    )
    assert df.select(pl.sum("a")).item() == 0.0
    assert df.group_by("b").agg(pl.sum("a"))["a"].item() == 0.0


def test_horizontal_sum_null_to_identity() -> None:
    assert pl.DataFrame({"a": [1, 5], "b": [10, None]}).select(
        pl.sum_horizontal(["a", "b"])
    ).to_series().to_list() == [11, 5]


def test_horizontal_sum_bool_dtype() -> None:
    out = pl.DataFrame({"a": [True, False]}).select(pl.sum_horizontal("a"))
    assert_frame_equal(
        out, pl.DataFrame({"a": pl.Series([1, 0], dtype=pl.get_index_type())})
    )


def test_horizontal_sum_in_group_by_15102() -> None:
    nbr_records = 1000
    out = (
        pl.LazyFrame(
            {
                "x": [None, "two", None] * nbr_records,
                "y": ["one", "two", None] * nbr_records,
                "z": [None, "two", None] * nbr_records,
            }
        )
        .select(pl.sum_horizontal(pl.all().is_null()).alias("num_null"))
        .group_by("num_null")
        .len()
        .sort(by="num_null")
        .collect()
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "num_null": pl.Series([0, 2, 3], dtype=pl.get_index_type()),
                "len": pl.Series([nbr_records] * 3, dtype=pl.get_index_type()),
            }
        ),
    )


def test_first_last_unit_length_12363() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [None, None],
        }
    )

    assert df.select(
        pl.all().drop_nulls().first().name.suffix("_first"),
        pl.all().drop_nulls().last().name.suffix("_last"),
    ).to_dict(as_series=False) == {
        "a_first": [1],
        "b_first": [None],
        "a_last": [2],
        "b_last": [None],
    }


def test_binary_op_agg_context_no_simplify_expr_12423() -> None:
    expect = pl.DataFrame({"x": [1], "y": [1]}, schema={"x": pl.Int64, "y": pl.Int32})

    for simplify_expression in (True, False):
        assert_frame_equal(
            expect,
            pl.LazyFrame({"x": [1]})
            .group_by("x")
            .agg(y=pl.lit(1) * pl.lit(1))
            .collect(
                optimizations=pl.QueryOptFlags(simplify_expression=simplify_expression)
            ),
        )


def test_nan_inf_aggregation() -> None:
    df = pl.DataFrame(
        [
            ("both nan", np.nan),
            ("both nan", np.nan),
            ("nan and 5", np.nan),
            ("nan and 5", 5),
            ("nan and null", np.nan),
            ("nan and null", None),
            ("both none", None),
            ("both none", None),
            ("both inf", np.inf),
            ("both inf", np.inf),
            ("inf and null", np.inf),
            ("inf and null", None),
        ],
        schema=["group", "value"],
        orient="row",
    )

    assert_frame_equal(
        df.group_by("group", maintain_order=True).agg(
            min=pl.col("value").min(),
            max=pl.col("value").max(),
            mean=pl.col("value").mean(),
        ),
        pl.DataFrame(
            [
                ("both nan", np.nan, np.nan, np.nan),
                ("nan and 5", 5, 5, np.nan),
                ("nan and null", np.nan, np.nan, np.nan),
                ("both none", None, None, None),
                ("both inf", np.inf, np.inf, np.inf),
                ("inf and null", np.inf, np.inf, np.inf),
            ],
            schema=["group", "min", "max", "mean"],
            orient="row",
        ),
    )


@pytest.mark.parametrize("dtype", [pl.Int16, pl.UInt16])
def test_int16_max_12904(dtype: PolarsDataType) -> None:
    s = pl.Series([None, 1], dtype=dtype)

    assert s.min() == 1
    assert s.max() == 1


def test_agg_filter_over_empty_df_13610() -> None:
    ldf = pl.LazyFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [True, True, True, True, True],
            "c": [None, None, None, None, None],
        }
    )

    out = (
        ldf.drop_nulls()
        .group_by(["a"], maintain_order=True)
        .agg(pl.col("b").filter(pl.col("b").shift(1)))
        .collect()
    )
    expected = pl.DataFrame(schema={"a": pl.Int64, "b": pl.List(pl.Boolean)})
    assert_frame_equal(out, expected)

    df = pl.DataFrame(schema={"a": pl.Int64, "b": pl.Boolean})
    out = df.group_by("a").agg(pl.col("b").filter(pl.col("b").shift()))
    expected = pl.DataFrame(schema={"a": pl.Int64, "b": pl.List(pl.Boolean)})
    assert_frame_equal(out, expected)


@pytest.mark.may_fail_cloud  # reason: output order is defined for this in cloud
@pytest.mark.may_fail_auto_streaming
@pytest.mark.slow
def test_agg_empty_sum_after_filter_14734() -> None:
    f = (
        pl.DataFrame({"a": [1, 2], "b": [1, 2]})
        .lazy()
        .group_by("a")
        .agg(pl.col("b").filter(pl.lit(False)).sum())
        .collect
    )

    last = f()

    # We need both possible output orders, which should happen within
    # 1000 iterations (during testing it usually happens within 10).
    limit = 1000
    i = 0
    while (curr := f()).equals(last):
        i += 1
        assert i != limit

    expect = pl.Series("b", [0, 0]).to_frame()
    assert_frame_equal(expect, last.select("b"))
    assert_frame_equal(expect, curr.select("b"))


@pytest.mark.slow
def test_grouping_hash_14749() -> None:
    n_groups = 251
    rows_per_group = 4
    assert (
        pl.DataFrame(
            {
                "grp": np.repeat(np.arange(n_groups), rows_per_group),
                "x": np.tile(np.arange(rows_per_group), n_groups),
            }
        )
        .select(pl.col("x").max().over("grp"))["x"]
        .value_counts()
    ).to_dict(as_series=False) == {"x": [3], "count": [1004]}


@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    [
        (pl.Boolean, pl.Float64),
        (pl.UInt8, pl.Float64),
        (pl.UInt16, pl.Float64),
        (pl.UInt32, pl.Float64),
        (pl.UInt64, pl.Float64),
        (pl.Int8, pl.Float64),
        (pl.Int16, pl.Float64),
        (pl.Int32, pl.Float64),
        (pl.Int64, pl.Float64),
        (pl.Float32, pl.Float32),
        (pl.Float64, pl.Float64),
    ],
)
def test_horizontal_mean_single_column(
    in_dtype: PolarsDataType,
    out_dtype: PolarsDataType,
) -> None:
    out = (
        pl.LazyFrame({"a": pl.Series([1, 0]).cast(in_dtype)})
        .select(pl.mean_horizontal(pl.all()))
        .collect()
    )

    assert_frame_equal(out, pl.DataFrame({"a": pl.Series([1.0, 0.0]).cast(out_dtype)}))


def test_horizontal_mean_in_group_by_15115() -> None:
    nbr_records = 1000
    out = (
        pl.LazyFrame(
            {
                "w": [None, "one", "two", "three"] * nbr_records,
                "x": [None, None, "two", "three"] * nbr_records,
                "y": [None, None, None, "three"] * nbr_records,
                "z": [None, None, None, None] * nbr_records,
            }
        )
        .select(pl.mean_horizontal(pl.all().is_null()).alias("mean_null"))
        .group_by("mean_null")
        .len()
        .sort(by="mean_null")
        .collect()
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "mean_null": pl.Series([0.25, 0.5, 0.75, 1.0], dtype=pl.Float64),
                "len": pl.Series([nbr_records] * 4, dtype=pl.get_index_type()),
            }
        ),
    )


def test_group_count_over_null_column_15705() -> None:
    df = pl.DataFrame(
        {"a": [1, 1, 2, 2, 3, 3], "c": [None, None, None, None, None, None]}
    )
    out = df.group_by("a", maintain_order=True).agg(pl.col("c").count())
    assert out["c"].to_list() == [0, 0, 0]


@pytest.mark.release
def test_min_max_2850() -> None:
    # https://github.com/pola-rs/polars/issues/2850
    df = pl.DataFrame(
        {
            "id": [
                130352432,
                130352277,
                130352611,
                130352833,
                130352305,
                130352258,
                130352764,
                130352475,
                130352368,
                130352346,
            ]
        }
    )

    minimum = 130352258
    maximum = 130352833.0

    for _ in range(10):
        permuted = df.sample(fraction=1.0, seed=0)
        computed = permuted.select(
            pl.col("id").min().alias("min"), pl.col("id").max().alias("max")
        )
        assert cast("int", computed[0, "min"]) == minimum
        assert cast("float", computed[0, "max"]) == maximum


def test_multi_arg_structify_15834() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 2, 1, 2],
            "value": [
                0.1973209146402105,
                0.13380719982405365,
                0.6152394463707009,
                0.4558767896005155,
            ],
        }
    )

    assert df.lazy().group_by("group").agg(
        pl.struct(a=1, value=pl.col("value").sum())
    ).collect().sort("group").to_dict(as_series=False) == {
        "group": [1, 2],
        "a": [
            {"a": 1, "value": 0.8125603610109114},
            {"a": 1, "value": 0.5896839894245691},
        ],
    }


def test_filter_aggregation_16642() -> None:
    df = pl.DataFrame(
        {
            "datetime": [
                datetime(2022, 1, 1, 11, 0),
                datetime(2022, 1, 1, 11, 1),
                datetime(2022, 1, 1, 11, 2),
                datetime(2022, 1, 1, 11, 3),
                datetime(2022, 1, 1, 11, 4),
                datetime(2022, 1, 1, 11, 5),
                datetime(2022, 1, 1, 11, 6),
                datetime(2022, 1, 1, 11, 7),
                datetime(2022, 1, 1, 11, 8),
                datetime(2022, 1, 1, 11, 9, 1),
                datetime(2022, 1, 2, 11, 0),
                datetime(2022, 1, 2, 11, 1),
                datetime(2022, 1, 2, 11, 2),
                datetime(2022, 1, 2, 11, 3),
                datetime(2022, 1, 2, 11, 4),
                datetime(2022, 1, 2, 11, 5),
                datetime(2022, 1, 2, 11, 6),
                datetime(2022, 1, 2, 11, 7),
                datetime(2022, 1, 2, 11, 8),
                datetime(2022, 1, 2, 11, 9, 1),
            ],
            "alpha": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ],
            "num": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    grouped = df.group_by(pl.col("datetime").dt.date())

    ts_filter = pl.col("datetime").dt.time() <= pl.time(11, 3)

    report = grouped.agg(pl.col("num").filter(ts_filter).max()).sort("datetime")
    assert report.to_dict(as_series=False) == {
        "datetime": [date(2022, 1, 1), date(2022, 1, 2)],
        "num": [3, 3],
    }


def test_sort_by_over_single_nulls_first() -> None:
    key = [0, 0, 0, 0, 1, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key": key,
            "value": [2, None, 1, 0, 2, None, 1, 0],
        }
    )
    out = df.select(
        pl.all().sort_by("value", nulls_last=False, maintain_order=True).over("key")
    )
    expected = pl.DataFrame(
        {
            "key": key,
            "value": [None, 0, 1, 2, None, 0, 1, 2],
        }
    )
    assert_frame_equal(out, expected)


def test_sort_by_over_single_nulls_last() -> None:
    key = [0, 0, 0, 0, 1, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key": key,
            "value": [2, None, 1, 0, 2, None, 1, 0],
        }
    )
    out = df.select(
        pl.all().sort_by("value", nulls_last=True, maintain_order=True).over("key")
    )
    expected = pl.DataFrame(
        {
            "key": key,
            "value": [0, 1, 2, None, 0, 1, 2, None],
        }
    )
    assert_frame_equal(out, expected)


def test_sort_by_over_multiple_nulls_first() -> None:
    key1 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    key2 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [1, None, 0, 1, None, 0, 1, None, 0, None, 1, 0],
        }
    )
    out = df.select(
        pl.all()
        .sort_by("value", nulls_last=False, maintain_order=True)
        .over("key1", "key2")
    )
    expected = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [None, 0, 1, None, 0, 1, None, 0, 1, None, 0, 1],
        }
    )
    assert_frame_equal(out, expected)


def test_sort_by_over_multiple_nulls_last() -> None:
    key1 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    key2 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    df = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [1, None, 0, 1, None, 0, 1, None, 0, None, 1, 0],
        }
    )
    out = df.select(
        pl.all()
        .sort_by("value", nulls_last=True, maintain_order=True)
        .over("key1", "key2")
    )
    expected = pl.DataFrame(
        {
            "key1": key1,
            "key2": key2,
            "value": [0, 1, None, 0, 1, None, 0, 1, None, 0, 1, None],
        }
    )
    assert_frame_equal(out, expected)


def test_slice_after_agg() -> None:
    assert_frame_equal(
        pl.select(a=pl.lit(1, dtype=pl.Int64), b=pl.lit(1, dtype=pl.Int64))
        .group_by("a")
        .agg(pl.col("b").first().slice(99, 0)),
        pl.DataFrame({"a": [1], "b": [[]]}, schema_overrides={"b": pl.List(pl.Int64)}),
    )


def test_agg_scalar_empty_groups_20115() -> None:
    assert_frame_equal(
        (
            pl.DataFrame({"key": [123], "value": [456]})
            .group_by("key")
            .agg(pl.col("value").slice(1, 1).first())
        ),
        pl.select(key=pl.lit(123, pl.Int64), value=pl.lit(None, pl.Int64)),
    )


def test_agg_expr_returns_list_type_15574() -> None:
    assert (
        pl.LazyFrame({"a": [1, None], "b": [1, 2]})
        .group_by("b")
        .agg(pl.col("a").drop_nulls())
        .collect_schema()
    ) == {"b": pl.Int64, "a": pl.List(pl.Int64)}


def test_empty_agg_22005() -> None:
    out = (
        pl.concat([pl.LazyFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]})])
        .limit(0)
        .select(pl.col("a").sum())
    )
    assert_frame_equal(out.collect(), pl.DataFrame({"a": 0}))


@pytest.mark.parametrize("wrap_numerical", [True, False])
@pytest.mark.parametrize("strict_cast", [True, False])
def test_agg_with_filter_then_cast_23682(
    strict_cast: bool, wrap_numerical: bool
) -> None:
    assert_frame_equal(
        pl.DataFrame([{"a": 123, "b": 12}, {"a": 123, "b": 257}])
        .group_by("a")
        .agg(
            pl.col("b")
            .filter(pl.col("b") < 256)
            .cast(pl.UInt8, strict=strict_cast, wrap_numerical=wrap_numerical)
        ),
        pl.DataFrame(
            [{"a": 123, "b": [12]}], schema={"a": pl.Int64, "b": pl.List(pl.UInt8)}
        ),
    )


@pytest.mark.parametrize("wrap_numerical", [True, False])
@pytest.mark.parametrize("strict_cast", [True, False])
def test_agg_with_slice_then_cast_23682(
    strict_cast: bool, wrap_numerical: bool
) -> None:
    assert_frame_equal(
        pl.DataFrame([{"a": 123, "b": 12}, {"a": 123, "b": 257}])
        .group_by("a")
        .agg(
            pl.col("b")
            .slice(0, 1)
            .cast(pl.UInt8, strict=strict_cast, wrap_numerical=wrap_numerical)
        ),
        pl.DataFrame(
            [{"a": 123, "b": [12]}], schema={"a": pl.Int64, "b": pl.List(pl.UInt8)}
        ),
    )


@pytest.mark.parametrize(
    ("op", "expr"),
    [
        ("any", pl.all().cast(pl.Boolean).any()),
        ("all", pl.all().cast(pl.Boolean).all()),
        ("arg_max", pl.all().arg_max()),
        ("arg_min", pl.all().arg_min()),
        ("min", pl.all().min()),
        ("max", pl.all().max()),
        ("mean", pl.all().mean()),
        ("median", pl.all().median()),
        ("product", pl.all().product()),
        ("quantile", pl.all().quantile(0.5)),
        ("std", pl.all().std()),
        ("var", pl.all().var()),
        ("sum", pl.all().sum()),
        ("first", pl.all().first()),
        ("last", pl.all().last()),
        ("approx_n_unique", pl.all().approx_n_unique()),
        ("bitwise_and", pl.all().bitwise_and()),
        ("bitwise_or", pl.all().bitwise_or()),
        ("bitwise_xor", pl.all().bitwise_xor()),
    ],
)
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"a": [[10]]}, schema={"a": pl.Array(shape=(1,), inner=pl.Int32)}),
        pl.DataFrame({"a": [[1]]}, schema={"a": pl.Struct(fields={"a": pl.Int32})}),
        pl.DataFrame({"a": [True]}, schema={"a": pl.Boolean}),
        pl.DataFrame({"a": ["a"]}, schema={"a": pl.Categorical}),
        pl.DataFrame({"a": [b"a"]}, schema={"a": pl.Binary}),
        pl.DataFrame({"a": ["a"]}, schema={"a": pl.Utf8}),
        pl.DataFrame({"a": [10]}, schema={"a": pl.Int32}),
        pl.DataFrame({"a": [10]}, schema={"a": pl.Float16}),
        pl.DataFrame({"a": [10]}, schema={"a": pl.Float32}),
        pl.DataFrame({"a": [10]}, schema={"a": pl.Float64}),
        pl.DataFrame({"a": [10]}, schema={"a": pl.Int128}),
        pl.DataFrame({"a": [10]}, schema={"a": pl.UInt128}),
        pl.DataFrame({"a": ["a"]}, schema={"a": pl.String}),
        pl.DataFrame({"a": [None]}, schema={"a": pl.Null}),
        pl.DataFrame({"a": [10]}, schema={"a": pl.Decimal()}),
        pl.DataFrame({"a": [datetime.now()]}, schema={"a": pl.Datetime}),
        pl.DataFrame({"a": [date.today()]}, schema={"a": pl.Date}),
        pl.DataFrame({"a": [timedelta(seconds=10)]}, schema={"a": pl.Duration}),
    ],
)
def test_agg_invalid_same_engines_behavior(
    op: str, expr: pl.Expr, df: pl.DataFrame
) -> None:
    # If the in-memory engine produces a good result, then the streaming engine
    # should also produce a good result, and then it should match the in-memory result.

    if isinstance(df.schema["a"], pl.Struct) and op in {"any", "all"}:
        # TODO: Remove this exception when #24509 is resolved
        pytest.skip("polars/#24509")

    if isinstance(df.schema["a"], pl.Duration) and op in {"std", "var"}:
        # TODO: Remove this exception when std & var are implemented for Duration
        pytest.skip(f"'{op}' aggregation not yet implemented for Duration")

    inmemory_result, inmemory_error = None, None
    streaming_result, streaming_error = None, None

    try:
        inmemory_result = df.select(expr)
    except pl.exceptions.PolarsError as e:
        inmemory_error = e

    try:
        streaming_result = df.lazy().select(expr).collect(engine="streaming")
    except pl.exceptions.PolarsError as e:
        streaming_error = e

    assert (streaming_error is None) == (inmemory_error is None), (
        f"mismatch in errors for: {streaming_error} != {inmemory_error}"
    )
    if inmemory_error:
        assert streaming_error, (
            f"streaming engine did not error (expected in-memory error: {inmemory_error})"
        )
        assert streaming_error.__class__ == inmemory_error.__class__

    if not inmemory_error:
        assert streaming_result is not None
        assert inmemory_result is not None
        assert_frame_equal(streaming_result, inmemory_result)


@pytest.mark.parametrize(
    ("op", "expr"),
    [
        ("sum", pl.all().sum()),
        ("mean", pl.all().mean()),
        ("median", pl.all().median()),
        ("std", pl.all().std()),
        ("var", pl.all().var()),
        ("quantile", pl.all().quantile(0.5)),
        ("cum_sum", pl.all().cum_sum()),
    ],
)
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"a": [[10]]}, schema={"a": pl.Array(shape=(1), inner=pl.Int32)}),
        pl.DataFrame({"a": [[1]]}, schema={"a": pl.Struct(fields={"a": pl.Int32})}),
        pl.DataFrame({"a": ["a"]}, schema={"a": pl.Categorical}),
        pl.DataFrame({"a": [b"a"]}, schema={"a": pl.Binary}),
        pl.DataFrame({"a": ["a"]}, schema={"a": pl.Utf8}),
        pl.DataFrame({"a": ["a"]}, schema={"a": pl.String}),
    ],
)
def test_invalid_agg_dtypes_should_raise(
    op: str, expr: pl.Expr, df: pl.DataFrame
) -> None:
    with pytest.raises(
        pl.exceptions.PolarsError, match=rf"`{op}` operation not supported for dtype"
    ):
        df.select(expr)
    with pytest.raises(
        pl.exceptions.PolarsError, match=rf"`{op}` operation not supported for dtype"
    ):
        df.lazy().select(expr).collect(engine="streaming")


@given(
    df=dataframes(
        min_size=1,
        max_size=1,
        excluded_dtypes=[
            # TODO: polars/#24936
            pl.Struct,
        ],
    )
)
def test_single(df: pl.DataFrame) -> None:
    q = df.lazy().select(pl.all(ignore_nulls=False).item())
    assert_frame_equal(q.collect(), df)
    assert_frame_equal(q.collect(engine="streaming"), df)


@given(df=dataframes(max_size=0))
def test_single_empty(df: pl.DataFrame) -> None:
    q = df.lazy().select(pl.all().item())
    match = "aggregation 'item' expected a single value, got none"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect(engine="streaming")


@given(df=dataframes(min_size=2))
def test_item_too_many(df: pl.DataFrame) -> None:
    q = df.lazy().select(pl.all(ignore_nulls=False).item())
    match = f"aggregation 'item' expected a single value, got {df.height} values"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect(engine="streaming")


@given(
    df=dataframes(
        min_size=1,
        max_size=1,
        allow_null=False,
        excluded_dtypes=[
            # TODO: polars/#24936
            pl.Struct,
        ],
    )
)
def test_item_on_groups(df: pl.DataFrame) -> None:
    df = df.with_columns(pl.col("col0").alias("key"))
    q = df.lazy().group_by("col0").agg(pl.all(ignore_nulls=False).item())
    assert_frame_equal(q.collect(), df)
    assert_frame_equal(q.collect(engine="streaming"), df)


def test_item_on_groups_empty() -> None:
    df = pl.DataFrame({"col0": [[]]})
    q = df.lazy().select(pl.all().list.item())
    match = "aggregation 'item' expected a single value, got none"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect(engine="streaming")


def test_item_on_groups_too_many() -> None:
    df = pl.DataFrame({"col0": [[1, 2, 3]]})
    q = df.lazy().select(pl.all().list.item())
    match = "aggregation 'item' expected a single value, got 3 values"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect()
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect(engine="streaming")


def test_all_any_on_list_raises_error() -> None:
    # Ensure boolean reductions on non-boolean columns raise an error.
    # (regression for #24942).
    lf = pl.LazyFrame({"x": [[True]]}, schema={"x": pl.List(pl.Boolean)})

    # for in-memory engine
    for expr in (pl.col("x").all(), pl.col("x").any()):
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match=r"expected boolean"
        ):
            lf.select(expr).collect()

    # for streaming engine
    for expr in (pl.col("x").all(), pl.col("x").any()):
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match=r"expected boolean"
        ):
            lf.select(expr).collect(engine="streaming")


@pytest.mark.parametrize("null_endpoints", [True, False])
@pytest.mark.parametrize("ignore_nulls", [True, False])
@pytest.mark.parametrize(
    ("dtype", "first_value", "last_value"),
    [
        # Struct
        (
            pl.Struct({"x": pl.Enum(["c0", "c1"]), "y": pl.Float32}),
            {"x": "c0", "y": 1.2},
            {"x": "c1", "y": 3.4},
        ),
        # List
        (pl.List(pl.UInt8), [1], [2]),
        # Array
        (pl.Array(pl.Int16, 2), [1, 2], [3, 4]),
        # Date (logical test)
        (pl.Date, date(2025, 1, 1), date(2025, 1, 2)),
        # Float (primitive test)
        (pl.Float32, 1.0, 2.0),
    ],
)
def test_first_last_nested(
    null_endpoints: bool,
    ignore_nulls: bool,
    dtype: PolarsDataType,
    first_value: Any,
    last_value: Any,
) -> None:
    s = pl.Series([first_value, last_value], dtype=dtype)
    if null_endpoints:
        # Test the case where the first/last value is null
        null = pl.Series([None], dtype=dtype)
        s = pl.concat((null, s, null))

    lf = pl.LazyFrame({"a": s})

    # first
    result = lf.select(pl.col("a").first(ignore_nulls=ignore_nulls)).collect()
    expected = pl.DataFrame(
        {
            "a": pl.Series(
                [None if null_endpoints and not ignore_nulls else first_value],
                dtype=dtype,
            )
        }
    )
    assert_frame_equal(result, expected)

    # last
    result = lf.select(pl.col("a").last(ignore_nulls=ignore_nulls)).collect()
    expected = pl.DataFrame(
        {
            "a": pl.Series(
                [None if null_endpoints and not ignore_nulls else last_value],
                dtype=dtype,
            ),
        }
    )
    assert_frame_equal(result, expected)


def test_struct_enum_agg_streaming_24936() -> None:
    s = (
        pl.Series(
            "a",
            [{"f0": "c0"}],
            dtype=pl.Struct({"f0": pl.Enum(categories=["c0"])}),
        ),
    )
    df = pl.DataFrame(s)

    q = df.lazy().select(pl.all(ignore_nulls=False).first())
    assert_frame_equal(q.collect(), df)


def test_sum_inf_not_nan_25849() -> None:
    data = [10.0, None, 10.0, 10.0, 10.0, 10.0, float("inf"), 10.0, 10.0]
    df = pl.DataFrame({"x": data, "g": ["X"] * len(data)})
    assert df.group_by("g").agg(pl.col("x").sum())["x"].item() == float("inf")


COLS = ["flt", "dec", "int", "str", "cat", "enum", "date", "dt"]


@pytest.mark.parametrize(
    "agg_funcs", [(pl.Expr.min_by, pl.Expr.min), (pl.Expr.max_by, pl.Expr.max)]
)
@pytest.mark.parametrize("by_col", COLS)
def test_min_max_by(agg_funcs: Any, by_col: str) -> None:
    agg_by, agg = agg_funcs
    df = pl.DataFrame(
        {
            "flt": [3.0, 2.0, float("nan"), 5.0, None, 4.0],
            "dec": [3, 2, None, 5, None, 4],
            "int": [3, 2, None, 5, None, 4],
            "str": ["c", "b", None, "e", None, "d"],
            "cat": ["c", "b", None, "e", None, "d"],
            "enum": ["c", "b", None, "e", None, "d"],
            "date": [
                date(2023, 3, 3),
                date(2023, 2, 2),
                None,
                date(2023, 5, 5),
                None,
                date(2023, 4, 4),
            ],
            "dt": [
                datetime(2023, 3, 3),
                datetime(2023, 2, 2),
                None,
                datetime(2023, 5, 5),
                None,
                datetime(2023, 4, 4),
            ],
            "g": [1, 1, 1, 2, 2, 2],
        },
        schema_overrides={
            "dec": pl.Decimal(scale=5),
            "cat": pl.Categorical,
            "enum": pl.Enum(["a", "b", "c", "d", "e", "f"]),
        },
    )

    result = df.select([agg_by(pl.col(c), pl.col(by_col)) for c in COLS])
    expected = df.select([agg(pl.col(c)) for c in COLS])
    assert_frame_equal(result, expected)

    # TODO: remove after https://github.com/pola-rs/polars/issues/25906.
    if by_col != "cat":
        df = df.drop("cat")
        cols = [c for c in COLS if c != "cat"]

        result = df.group_by("g").agg([agg_by(pl.col(c), pl.col(by_col)) for c in cols])
        expected = df.group_by("g").agg([agg(pl.col(c)) for c in cols])
        assert_frame_equal(result, expected, check_row_order=False)


@pytest.mark.parametrize(("agg", "expected"), [("max", 2), ("min", 0)])
def test_grouped_minmax_after_reverse_on_sorted_column_26141(
    agg: str, expected: int
) -> None:
    df = pl.DataFrame({"a": [0, 1, 2]}).sort("a")

    expr = getattr(pl.col("a").reverse(), agg)()
    out = df.group_by(1).agg(expr)

    expected_df = pl.DataFrame(
        {
            "literal": pl.Series([1], dtype=pl.Int32),
            "a": [expected],
        }
    )
    assert_frame_equal(out, expected_df)


@pytest.mark.may_fail_auto_streaming
@pytest.mark.parametrize("agg_by", [pl.Expr.min_by, pl.Expr.max_by])
def test_min_max_by_series_length_mismatch_26049(
    agg_by: Callable[[pl.Expr, pl.Expr], pl.Expr],
) -> None:
    lf = pl.LazyFrame(
        {
            "a": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "b": [18, 5, 8, 8, 4, 5, 6, 8, 1, -10],
            "group": ["A", "A", "A", "A", "A", "B", "B", "C", "C", "C"],
        }
    )

    q = lf.with_columns(
        agg_by(pl.col("group").filter(pl.col("b") % 2 == 0), pl.col("a"))
    )

    with pytest.raises(
        pl.exceptions.ShapeError,
        match=r"^'by' column in (min|max)_by expression has incorrect length: expected \d+, got \d+$",
    ):
        q.collect(engine="in-memory")
    with pytest.raises(
        pl.exceptions.ShapeError,
        match=r"^zip node received non-equal length inputs$",
    ):
        q.collect(engine="streaming")

    actual = (
        lf.group_by("group")
        .agg(
            pl.col("a")
            .max_by(pl.col("b").filter(pl.col("b") < 20).abs())
            .alias("max_by")
        )
        .sort("group")
    ).collect()
    expected = pl.DataFrame(
        {
            "group": ["A", "B", "C"],
            "max_by": [0, 60, 90],
        }
    )
    assert_frame_equal(actual, expected)

    q = (
        lf.group_by("group")
        .agg(
            pl.col("a")
            .max_by(pl.col("b").filter(pl.col("b") < 7).abs())
            .alias("group_length_mismatch")
        )
        .sort("group")
    )
    with pytest.raises(
        pl.exceptions.ShapeError,
        match=r"^expressions must have matching group lengths$",
    ):
        q.collect(engine="in-memory")


@pytest.mark.parametrize(
    "by_expr",
    [
        pl.struct("b", "c"),
        pl.concat_list("b", "c"),
    ],
)
def test_min_by_max_by_nested_type_key_26268(by_expr: pl.Expr) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 6, 5], "c": [7, 5, 2]})

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="cannot use a nested type as `by` argument in `min_by`/`max_by`",
    ):
        df.select(pl.col("a").min_by(by_expr))

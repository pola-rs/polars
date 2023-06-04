import math
import typing
from datetime import date, datetime, timedelta

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_quantile_expr_input() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [0, 0, 0.3, 0.2, 0]})

    assert_frame_equal(
        df.select([pl.col("a").quantile(pl.col("b").sum() + 0.1)]),
        df.select(pl.col("a").quantile(0.6)),
    )


def test_boolean_aggs() -> None:
    df = pl.DataFrame({"bool": [True, False, None, True]})

    aggs = [
        pl.mean("bool").alias("mean"),
        pl.std("bool").alias("std"),
        pl.var("bool").alias("var"),
    ]
    assert df.select(aggs).to_dict(False) == {
        "mean": [0.6666666666666666],
        "std": [0.5773502588272095],
        "var": [0.3333333432674408],
    }

    assert df.groupby(pl.lit(1)).agg(aggs).to_dict(False) == {
        "literal": [1],
        "mean": [0.6666666666666666],
        "std": [0.5773502691896258],
        "var": [0.33333333333333337],
    }


def test_duration_aggs() -> None:
    df = pl.DataFrame(
        {
            "time1": pl.date_range(
                start=datetime(2022, 12, 12),
                end=datetime(2022, 12, 18),
                interval="1d",
                eager=True,
            ),
            "time2": pl.date_range(
                start=datetime(2023, 1, 12),
                end=datetime(2023, 1, 18),
                interval="1d",
                eager=True,
            ),
        }
    )

    df = df.with_columns((pl.col("time2") - pl.col("time1")).alias("time_difference"))

    assert df.select("time_difference").mean().to_dict(False) == {
        "time_difference": [timedelta(days=31)]
    }
    assert df.groupby(pl.lit(1)).agg(pl.mean("time_difference")).to_dict(False) == {
        "literal": [1],
        "time_difference": [timedelta(days=31)],
    }


def test_hmean_with_str_column() -> None:
    assert pl.DataFrame(
        {"int": [1, 2, 3], "bool": [True, True, None], "str": ["a", "b", "c"]}
    ).mean(axis=1).to_list() == [1.0, 1.5, 3.0]


def test_list_aggregation_that_filters_all_data_6017() -> None:
    out = (
        pl.DataFrame({"col_to_groupby": [2], "flt": [1672740910.967138], "col3": [1]})
        .groupby("col_to_groupby")
        .agg(
            (pl.col("flt").filter(pl.col("col3") == 0).diff() * 1000)
            .diff()
            .alias("calc")
        )
    )

    assert out.schema == {"col_to_groupby": pl.Int64, "calc": pl.List(pl.Float64)}
    assert out.to_dict(False) == {"col_to_groupby": [2], "calc": [[]]}


def test_median() -> None:
    s = pl.Series([1, 2, 3])
    assert s.median() == 2


def test_single_element_std() -> None:
    s = pl.Series([1])
    assert math.isnan(typing.cast(float, s.std(ddof=1)))
    assert s.std(ddof=0) == 0.0


def test_quantile() -> None:
    s = pl.Series([1, 2, 3])
    assert s.quantile(0.5, "nearest") == 2
    assert s.quantile(0.5, "lower") == 2
    assert s.quantile(0.5, "higher") == 2


@pytest.mark.slow()
@typing.no_type_check
def test_quantile_vs_numpy() -> None:
    for tp in [int, float]:
        for n in [1, 2, 10, 100]:
            a = np.random.randint(0, 50, n).astype(tp)
            np_result = np.median(a)
            # nan check
            if np_result != np_result:
                np_result = None
            median = pl.Series(a).median()
            if median is not None:
                assert np.isclose(median, np_result)
            else:
                assert np_result is None

            q = np.random.sample()
            try:
                np_result = np.quantile(a, q)
            except IndexError:
                np_result = None
                pass
            if np_result:
                # nan check
                if np_result != np_result:
                    np_result = None
                assert np.isclose(
                    pl.Series(a).quantile(q, interpolation="linear"), np_result
                )


@typing.no_type_check
def test_mean_overflow() -> None:
    assert np.isclose(
        pl.Series([9_223_372_036_854_775_800, 100]).mean(), 4.611686018427388e18
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
        df.groupby("A").agg(pl.col("B").search_sorted(ser)),
        pl.DataFrame(
            [
                pl.Series("A", [1], dtype=pl.Int64),
                pl.Series("B", [[1, 2, 2]], dtype=pl.List(pl.UInt32)),
            ]
        ),
    )


def test_duration_function_literal() -> None:
    df = pl.DataFrame(
        {
            "A": ["x", "x", "y", "y", "y"],
            "T": [date(2022, m, 1) for m in range(1, 6)],
            "S": [1, 2, 4, 8, 16],
        }
    ).with_columns(
        [
            pl.col("T").cast(pl.Datetime),
        ]
    )

    # this checks if the `pl.duration` is flagged as AggState::Literal
    assert df.groupby("A", maintain_order=True).agg(
        [((pl.col("T").max() + pl.duration(seconds=1)) - pl.col("T"))]
    ).to_dict(False) == {
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


def test_string_par_materialize_8207() -> None:
    df = pl.LazyFrame(
        {
            "a": ["a", "b", "d", "c", "e"],
            "b": ["P", "L", "R", "T", "a long string"],
        }
    )

    assert df.groupby(["a"]).agg(pl.min("b")).sort("a").collect().to_dict(False) == {
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
        df.groupby("id")
        .agg(pl.all().exclude("id").std())
        .select(["no_nulls", "nulls"]),
        df.select(pl.all().exclude("id").std()),
    )


def test_err_on_implode_and_agg() -> None:
    df = pl.DataFrame({"type": ["water", "fire", "water", "earth"]})

    # this would OOB
    with pytest.raises(
        pl.InvalidOperationError,
        match=r"'implode' followed by an aggregation is not allowed",
    ):
        df.groupby("type").agg(pl.col("type").implode().first().alias("foo"))

    # implode + function should be allowed in groupby
    assert df.groupby("type", maintain_order=True).agg(
        pl.col("type").implode().list.head().alias("foo")
    ).to_dict(False) == {
        "type": ["water", "fire", "earth"],
        "foo": [["water", "water"], ["fire"], ["earth"]],
    }

    # but not during a window function as the groups cannot be mapped back
    with pytest.raises(
        pl.InvalidOperationError,
        match=r"'implode' followed by an aggregation is not allowed",
    ):
        df.lazy().select(pl.col("type").implode().list.head(1).over("type")).collect()


def test_mapped_literal_to_literal_9217() -> None:
    df = pl.DataFrame({"unique_id": ["a", "b"]})
    assert df.groupby(True).agg(
        pl.struct(pl.lit("unique_id").alias("unique_id"))
    ).to_dict(False) == {"literal": [True], "unique_id": [{"unique_id": "unique_id"}]}

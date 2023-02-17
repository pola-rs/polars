import typing
from datetime import datetime, timedelta

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
                low=datetime(2022, 12, 12), high=datetime(2022, 12, 18), interval="1d"
            ),
            "time2": pl.date_range(
                low=datetime(2023, 1, 12), high=datetime(2023, 1, 18), interval="1d"
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

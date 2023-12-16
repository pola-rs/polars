from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    import numpy.typing as npt


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
    assert df.select(aggs).to_dict(as_series=False) == {
        "mean": [0.6666666666666666],
        "std": [0.5773502588272095],
        "var": [0.3333333432674408],
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


def test_mean_horizontal_with_str_column() -> None:
    assert pl.DataFrame(
        {"int": [1, 2, 3], "bool": [True, True, None], "str": ["a", "b", "c"]}
    ).mean_horizontal().to_list() == [1.0, 1.5, 3.0]


def test_list_aggregation_that_filters_all_data_6017() -> None:
    out = (
        pl.DataFrame({"col_to_group_by": [2], "flt": [1672740910.967138], "col3": [1]})
        .group_by("col_to_group_by")
        .agg(
            (pl.col("flt").filter(pl.col("col3") == 0).diff() * 1000)
            .diff()
            .alias("calc")
        )
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


@pytest.mark.slow()
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
                pl.Series("B", [[1, 2, 2]], dtype=pl.List(pl.UInt32)),
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


def test_err_on_implode_and_agg() -> None:
    df = pl.DataFrame({"type": ["water", "fire", "water", "earth"]})

    # this would OOB
    with pytest.raises(
        pl.InvalidOperationError,
        match=r"'implode' followed by an aggregation is not allowed",
    ):
        df.group_by("type").agg(pl.col("type").implode().first().alias("foo"))

    # implode + function should be allowed in group_by
    assert df.group_by("type", maintain_order=True).agg(
        pl.col("type").implode().list.head().alias("foo")
    ).to_dict(as_series=False) == {
        "type": ["water", "fire", "earth"],
        "foo": [[["water", "water"]], [["fire"]], [["earth"]]],
    }

    # but not during a window function as the groups cannot be mapped back
    with pytest.raises(
        pl.InvalidOperationError,
        match=r"'implode' followed by an aggregation is not allowed",
    ):
        df.lazy().select(pl.col("type").implode().list.head(1).over("type")).collect()


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
        [pl.sum_horizontal(["a", "b"])]
    ).to_series().to_list() == [11, 5]


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
            .collect(simplify_expression=simplify_expression),
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
        ),
    )


@pytest.mark.parametrize("dtype", [pl.Int16, pl.UInt16])
def test_int16_max_12904(dtype: pl.PolarsDataType) -> None:
    s = pl.Series([None, 1], dtype=dtype)

    assert s.min() == 1
    assert s.max() == 1

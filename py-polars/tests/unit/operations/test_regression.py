from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType

ENGINES: list[EngineType] = ["in-memory", "streaming"]


def regr_exprs() -> list[pl.Expr]:
    return [
        pl.regr_slope("y", "x").alias("slope"),
        pl.regr_intercept("y", "x").alias("intercept"),
        pl.regr_r2("y", "x").alias("r2"),
        pl.regr_count("y", "x").alias("count"),
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_regr_basic(engine: EngineType) -> None:
    lf = pl.LazyFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    result = lf.select(regr_exprs()).collect(engine=engine)
    expected = pl.DataFrame(
        {
            "slope": [0.6],
            "intercept": [2.2],
            "r2": [0.6],
            "count": pl.Series([5], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_regr_ignores_null_pairs(engine: EngineType) -> None:
    lf = pl.LazyFrame({"x": [1, 2, None, 4, 5], "y": [2, 4, 5, None, 5]})
    result = lf.select(regr_exprs()).collect(engine=engine)
    expected = pl.DataFrame(
        {
            "slope": [17 / 26],
            "intercept": [25 / 13],
            "r2": [289 / 364],
            "count": pl.Series([3], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_regr_empty_and_all_null(engine: EngineType) -> None:
    lazy_frames = [
        pl.LazyFrame(schema={"x": pl.Int64, "y": pl.Int64}),
        pl.LazyFrame(
            {"x": [None, None], "y": [None, None]},
            schema={"x": pl.Int64, "y": pl.Int64},
        ),
        pl.LazyFrame({"x": [1, None], "y": [None, 2]}),
    ]
    expected = pl.DataFrame(
        {
            "slope": pl.Series([None], dtype=pl.Float64),
            "intercept": pl.Series([None], dtype=pl.Float64),
            "r2": pl.Series([None], dtype=pl.Float64),
            "count": pl.Series([0], dtype=pl.get_index_type()),
        }
    )
    for lf in lazy_frames:
        result = lf.select(regr_exprs()).collect(engine=engine)
        assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_regr_zero_variance(engine: EngineType) -> None:
    # Zero variance in x -> slope/intercept/r2 are all null.
    lf = pl.LazyFrame({"x": [2, 2, 2], "y": [1, 2, 3]})
    result = lf.select(regr_exprs()).collect(engine=engine)
    expected = pl.DataFrame(
        {
            "slope": pl.Series([None], dtype=pl.Float64),
            "intercept": pl.Series([None], dtype=pl.Float64),
            "r2": pl.Series([None], dtype=pl.Float64),
            "count": pl.Series([3], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)

    # Zero variance in y only -> slope/intercept are 0, r2 is 1.
    lf = pl.LazyFrame({"x": [1, 2, 3], "y": [7, 7, 7]})
    result = lf.select(regr_exprs()).collect(engine=engine)
    expected = pl.DataFrame(
        {
            "slope": [0.0],
            "intercept": [7.0],
            "r2": [1.0],
            "count": pl.Series([3], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_regr_group_by(engine: EngineType) -> None:
    lf = pl.LazyFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b", "c"],
            "x": [1, 2, 3, 1, 2, 3, 1],
            "y": [1, 2, 3, 2, 4, 8, 5],
        }
    )
    result = lf.group_by("g").agg(regr_exprs()).sort("g").collect(engine=engine)
    expected = pl.DataFrame(
        {
            "g": ["a", "b", "c"],
            "slope": [1.0, 3.0, None],
            "intercept": [0.0, -4 / 3, None],
            "r2": [1.0, 27 / 28, None],
            "count": pl.Series([3, 3, 1], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_regr_vs_corr_cov(engine: EngineType) -> None:
    lf = pl.LazyFrame(
        {
            "x": [1.5, 2.0, 8.0, 4.5, 6.25, 3.0],
            "y": [2.25, 4.0, 5.5, 4.0, 5.75, 3.5],
        }
    )
    slope_ref = pl.cov("x", "y") / pl.col("x").var()
    result = lf.select(
        pl.regr_slope("y", "x").alias("slope"),
        pl.regr_intercept("y", "x").alias("intercept"),
        pl.regr_r2("y", "x").alias("r2"),
        slope_ref.alias("slope_ref"),
        (pl.col("y").mean() - slope_ref * pl.col("x").mean()).alias("intercept_ref"),
        pl.corr("x", "y").pow(2).alias("r2_ref"),
    ).collect(engine=engine)
    row = result.row(0, named=True)
    assert row["slope"] == pytest.approx(row["slope_ref"])
    assert row["intercept"] == pytest.approx(row["intercept_ref"])
    assert row["r2"] == pytest.approx(row["r2_ref"])


@pytest.mark.parametrize("engine", ENGINES)
def test_regr_casts_to_float64(engine: EngineType) -> None:
    lf = pl.LazyFrame(
        {
            "x": pl.Series([1, 2, 3], dtype=pl.UInt8),
            "y": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
        }
    )
    result = lf.select(regr_exprs()).collect(engine=engine)
    assert result.schema["slope"] == pl.Float64
    assert result.schema["intercept"] == pl.Float64
    assert result.schema["r2"] == pl.Float64
    assert result.schema["count"] == pl.get_index_type()
    assert result.row(0) == (1.0, 0.0, 1.0, 3)


def test_regr_broadcast_scalar() -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    result = df.select(
        pl.regr_slope("y", pl.lit(1.0)).alias("slope"),
        pl.regr_count("y", pl.lit(1.0)).alias("count"),
    )
    assert result.row(0) == (None, 3)


def test_regr_eager() -> None:
    x = pl.Series("x", [1, 2, 3, 4, 5])
    y = pl.Series("y", [2, 4, 5, 4, 5])
    assert pl.regr_slope(y, x, eager=True).item() == pytest.approx(0.6)
    assert pl.regr_intercept(y, x, eager=True).item() == pytest.approx(2.2)
    assert pl.regr_r2(y, x, eager=True).item() == pytest.approx(0.6)
    assert pl.regr_count(y, x, eager=True).item() == 5

    with pytest.raises(ValueError, match="expected at least one Series"):
        pl.regr_slope("y", "x", eager=True)

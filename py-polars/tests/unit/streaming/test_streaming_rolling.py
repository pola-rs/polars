from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")

ROLLING_FUNCTIONS: dict[str, Callable[..., pl.Expr]] = {
    "min": lambda e, **kw: e.rolling_min(**kw),
    "max": lambda e, **kw: e.rolling_max(**kw),
    "mean": lambda e, **kw: e.rolling_mean(**kw),
    "sum": lambda e, **kw: e.rolling_sum(**kw),
    "var": lambda e, **kw: e.rolling_var(**kw),
    "std": lambda e, **kw: e.rolling_std(**kw, ddof=0),
    "median": lambda e, **kw: e.rolling_median(**kw),
    "quantile": lambda e, **kw: e.rolling_quantile(0.3, "linear", **kw),
    "skew": lambda e, **kw: e.rolling_skew(**kw),
    "kurtosis": lambda e, **kw: e.rolling_kurtosis(**kw),
    "rank": lambda e, **kw: e.rolling_rank(**kw, method="max"),
    "map": lambda e, **kw: e.rolling_map(lambda s: s.sum(), **kw),
}
WEIGHTED_FUNCTIONS = ["min", "max", "mean", "sum", "var", "std", "quantile", "map"]


def rolling_frame(n: int = 100) -> pl.LazyFrame:
    rng = np.random.default_rng(0)
    values = rng.normal(size=n).round(3)
    return pl.LazyFrame(
        {
            "f": [
                None if i % 11 == 3 or 40 <= i < 45 else v for i, v in enumerate(values)
            ],
            "i": rng.integers(0, 20, size=n),
        }
    )


def physical_plan(lf: pl.LazyFrame) -> str:
    plan = lf.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True, show=False
    )
    assert isinstance(plan, str)
    return plan


def assert_streaming_matches(lf: pl.LazyFrame) -> None:
    plan = physical_plan(lf)
    assert "rolling-fixed-window-function" in plan
    assert "columnar-function" not in plan

    expected = lf.collect(engine="in-memory")
    assert_frame_equal(lf.collect(engine="streaming"), expected, rel_tol=1e-6)


@pytest.fixture(autouse=True)
def small_morsels(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "7")


@pytest.mark.parametrize("func", ROLLING_FUNCTIONS)
@pytest.mark.parametrize("center", [False, True])
def test_streaming_rolling(func: str, center: bool) -> None:
    f = ROLLING_FUNCTIONS[func]
    inputs = {
        "f": pl.col("f"),
        "i": pl.col("i"),
        "expr": pl.col("f").fill_null(0.0) * 2,
    }
    lf = rolling_frame().select(
        f(e, window_size=w, center=center, min_samples=m).alias(f"{name}_{w}_{m}")
        for w in [1, 4, 7, 23, 150]
        for m in [None, 1]
        for name, e in inputs.items()
    )
    assert_streaming_matches(lf)


@pytest.mark.parametrize("func", WEIGHTED_FUNCTIONS)
@pytest.mark.parametrize("center", [False, True])
def test_streaming_rolling_weights(func: str, center: bool) -> None:
    f = ROLLING_FUNCTIONS[func]
    x = pl.col("i").cast(pl.Float64)
    lf = rolling_frame().select(
        f(
            x,
            window_size=w,
            center=center,
            weights=[float(j % 3) for j in range(w)],
            min_samples=m,
        ).alias(f"i_{w}_{m}")
        for w in [3, 8, 20]
        for m in [None, 2]
    )
    assert_streaming_matches(lf)


def test_streaming_rolling_corr_cov() -> None:
    g = pl.col("f").fill_null(0.0) + pl.col("i")

    def exprs(**kw: Any) -> dict[str, pl.Expr]:
        return {
            "corr": pl.rolling_corr("f", g, **kw),
            "corr_rev": pl.rolling_corr(g, pl.col("f").shift(3), **kw),
            "cov0": pl.rolling_cov("f", pl.col("f").shift(3) + g, ddof=0, **kw),
            "cov1": pl.rolling_cov(g, "i", ddof=1, **kw),
            "int": pl.rolling_cov("i", pl.col("i") * 3 % 7, **kw),
            "cov": pl.rolling_cov("i", g, **kw),
        }

    lf = rolling_frame().select(
        e.alias(f"{name}_{w}_{m}")
        for w in [1, 2, 5, 23, 150]
        for m in [None, 1]
        for name, e in exprs(window_size=w, min_samples=m).items()
    )
    assert_streaming_matches(lf)


@pytest.mark.parametrize("morsel_size", ["1", "3", "1000"])
def test_streaming_rolling_morsel_sizes(
    plmonkeypatch: PlMonkeyPatch, morsel_size: str
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", morsel_size)
    x = pl.col("i").cast(pl.Float64)
    lf = rolling_frame(300).select(
        e.alias(f"{name}_{w}")
        for w in [2, 5, 64]
        for name, e in {
            "sum": pl.col("f").rolling_sum(w, min_samples=1),
            "mean": pl.col("f").rolling_mean(w, center=True),
            "q": pl.col("i").rolling_quantile(0.7, window_size=w),
            "map": pl.col("i").rolling_map(lambda s: s.max(), w, center=True),
            # `rolling_map` applies all weights, so each batch must hold a full window.
            **{
                f"map_weights_{center}": x.rolling_map(
                    lambda s: s.sum(),
                    w,
                    weights=[float(j + 1) for j in range(w)],
                    min_samples=1,
                    center=center,
                )
                for center in [False, True]
            },
        }.items()
    )
    assert_streaming_matches(lf)


def test_streaming_rolling_map_weights_default_morsel_size(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.delenv("POLARS_IDEAL_MORSEL_SIZE")
    lf = pl.LazyFrame({"b": [float(i % 5) for i in range(120)]}).select(
        pl.col("b").rolling_map(
            lambda s: s.sum(), 13, weights=[1.0] * 13, min_samples=1
        )
    )
    assert_streaming_matches(lf)


@pytest.mark.parametrize("center", [False, True])
def test_streaming_rolling_rank_methods(center: bool) -> None:
    methods: list[Any] = ["average", "min", "max", "dense"]
    lf = rolling_frame().select(
        e.alias(f"{name}_{method}")
        for method in methods
        for name, e in {
            "i": pl.col("i").rolling_rank(9, method=method, center=center),
            "f": pl.col("f").rolling_rank(
                9, method=method, center=center, min_samples=1
            ),
        }.items()
    )
    assert_streaming_matches(lf)


@pytest.mark.parametrize("center", [False, True])
def test_streaming_rolling_temporal(center: bool) -> None:
    n = 50
    lf = pl.LazyFrame(
        {
            "date": [date(2020, 1, 1) + timedelta(days=(i * 7) % 30) for i in range(n)],
            "dt": [datetime(2020, 1, 1) + timedelta(hours=i * i) for i in range(n)],
            "dur": [timedelta(seconds=(i * 13) % 17) for i in range(n)],
        }
    ).with_columns(pl.col("dt").dt.replace_time_zone("Europe/Amsterdam"))
    lf = lf.select(
        *(
            getattr(pl.col(c), f"rolling_{func}")(5, center=center).name.suffix(
                f"_{func}"
            )
            for c in ["date", "dt", "dur"]
            for func in ["min", "max", "mean", "median"]
        ),
        pl.col("dt").rolling_quantile(0.5, window_size=5, center=center).alias("q"),
    )
    assert_streaming_matches(lf)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_streaming_rolling_short_input(n: int) -> None:
    lf = pl.LazyFrame({"a": [float(i) for i in range(n)]}, schema={"a": pl.Float64})
    lf = lf.select(
        pl.col("a").rolling_sum(3).alias("sum"),
        pl.col("a").rolling_mean(3, min_samples=1, center=True).alias("mean"),
        pl.col("a").rolling_map(lambda s: s.sum(), 3, min_samples=1).alias("map"),
    )
    assert_streaming_matches(lf)


def test_streaming_rolling_scalar() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]}).select(
        pl.col("a"), pl.lit(2.0).rolling_mean(3, min_samples=1).alias("lit")
    )
    assert_streaming_matches(lf)


def test_streaming_rolling_fallback() -> None:
    lf = rolling_frame().select(pl.col("i").rolling_rank(5, method="random", seed=1))
    assert "rolling-fixed-window-function" not in physical_plan(lf)


def test_streaming_rolling_rank_random_unseeded() -> None:
    # Without ties the random method is deterministic.
    values = np.random.default_rng(0).permutation(100)
    lf = pl.LazyFrame({"a": values}).select(
        pl.col("a").rolling_rank(5, method="random")
    )
    assert_streaming_matches(lf)

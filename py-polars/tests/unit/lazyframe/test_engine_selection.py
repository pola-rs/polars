from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.lazyframe.frame import _plan_engine, _select_engine
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType


@pytest.fixture
def df() -> pl.LazyFrame:
    return pl.LazyFrame({"a": [1, 2, 3]})


@pytest.fixture(params=["gpu", pl.GPUEngine()])
def engine(request: pytest.FixtureRequest) -> EngineType:
    value: EngineType = request.param
    return value


def test_engine_selection_invalid_raises(df: pl.LazyFrame) -> None:
    with pytest.raises(ValueError):
        df.collect(engine="unknown")  # type: ignore[call-overload]


def test_engine_selection_background_warns(
    df: pl.LazyFrame, engine: EngineType
) -> None:
    expect = df.collect()
    with pytest.warns(
        UserWarning,
        match="GPU engine does not support background",
    ):
        got = df.collect(engine=engine, background=True)
    assert_frame_equal(expect, got.fetch_blocking())


def test_engine_selection_eager_quiet(df: pl.LazyFrame, engine: EngineType) -> None:
    expect = df.collect()
    # _eager collection turns off GPU engine quietly
    got = df.collect(engine=engine, optimizations=pl.QueryOptFlags._eager())
    assert_frame_equal(expect, got)


def test_engine_import_error_raises(df: pl.LazyFrame, engine: EngineType) -> None:
    with pytest.raises(
        ImportError,
        match="GPU engine requested",
    ):
        df.collect(engine=engine)


def test_engine_affinity_object_is_selected(df: pl.LazyFrame) -> None:
    engine = pl.GPUEngine(device=1)
    with pl.Config(engine_affinity=engine):
        assert _select_engine("auto") is engine
        # an explicit engine still wins over the affinity
        assert _select_engine("streaming") == "streaming"
        with pytest.raises(ImportError, match="GPU engine requested"):
            df.collect()


def test_engine_affinity_name_is_selected(df: pl.LazyFrame) -> None:
    with pl.Config(engine_affinity="streaming"):
        assert _select_engine("auto") == "streaming"


def test_collect_local_ignores_object_affinity(df: pl.LazyFrame) -> None:
    with pl.Config(engine_affinity=pl.RemoteEngine()):
        assert df._collect_local().height == 3


def test_remote_plan_engine_does_not_inherit_local_affinity() -> None:
    with pl.Config(engine_affinity="streaming"):
        assert _plan_engine(pl.RemoteEngine()) == "auto"

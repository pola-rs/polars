from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType


@pytest.fixture()
def df() -> pl.LazyFrame:
    return pl.LazyFrame({"a": [1, 2, 3]})


@pytest.fixture(params=["gpu", pl.GPUEngine()])
def engine(request: pytest.FixtureRequest) -> EngineType:
    value: EngineType = request.param
    return value


def test_engine_selection_invalid_raises(df: pl.LazyFrame) -> None:
    with pytest.raises(ValueError):
        df.collect(engine="unknown")  # type: ignore[call-overload]


def test_engine_selection_streaming_warns(df: pl.LazyFrame, engine: EngineType) -> None:
    expect = df.collect()
    with pytest.warns(
        UserWarning, match="GPU engine does not support streaming or background"
    ):
        got = df.collect(engine=engine, streaming=True)
        assert_frame_equal(expect, got)


def test_engine_selection_background_warns(
    df: pl.LazyFrame, engine: EngineType
) -> None:
    expect = df.collect()
    with pytest.warns(
        UserWarning, match="GPU engine does not support streaming or background"
    ):
        got = df.collect(engine=engine, background=True)
        assert_frame_equal(expect, got.fetch_blocking())


def test_engine_selection_eager_quiet(df: pl.LazyFrame, engine: EngineType) -> None:
    expect = df.collect()
    # _eager collection turns off GPU engine quietly
    got = df.collect(engine=engine, _eager=True)
    assert_frame_equal(expect, got)


def test_engine_import_error_raises(df: pl.LazyFrame, engine: EngineType) -> None:
    with pytest.raises(ImportError, match="GPU engine requested"):
        df.collect(engine=engine)

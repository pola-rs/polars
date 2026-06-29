from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType
    from tests.conftest import PlMonkeyPatch


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


@pytest.mark.parametrize(
    "engine_affinity",
    ["in-memory", "streaming", None],
)
def test_default_engine_and_affinity(
    engine_affinity: str | None,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_STREAMING", "0")
    plmonkeypatch.setenv("POLARS_AUTO_STREAMING", "0")

    if engine_affinity is not None:
        plmonkeypatch.setenv("POLARS_ENGINE_AFFINITY", engine_affinity)
    else:
        with contextlib.suppress(KeyError):
            plmonkeypatch.delenv("POLARS_ENGINE_AFFINITY")

    capfd.readouterr()
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    pl.LazyFrame().collect()
    capture = capfd.readouterr().err
    plmonkeypatch.setenv("POLARS_VERBOSE", "0")

    used_engine = (
        "streaming" if "polars-stream: updating graph state" in capture else "in-memory"
    )

    if engine_affinity is not None:
        assert used_engine == engine_affinity
    else:
        assert used_engine == "streaming"

    capfd.readouterr()

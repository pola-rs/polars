from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

import polars as pl
from tests.unit.utils.pathlike import HostilePathLike

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def query() -> pl.LazyFrame:
    return (
        pl.LazyFrame(
            {
                "a": ["a", "b", "a", "b", "b", "c"],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [6, 5, 4, 3, 2, 1],
            }
        )
        .group_by("a", maintain_order=True)
        .agg(pl.all().sum())
        .sort("a")
    )


def test_show_graph_ir(query: pl.LazyFrame) -> None:
    # only test raw output, otherwise we need graphviz and matplotlib
    out = query.show_graph(raw_output=True, plan_stage="ir")
    assert isinstance(out, str)


def test_show_graph_phys_streaming(query: pl.LazyFrame) -> None:
    # only test raw output, otherwise we need graphviz and matplotlib
    out = query.show_graph(raw_output=True, plan_stage="physical", engine="streaming")
    assert isinstance(out, str)


def test_show_graph_phys_not_streaming(query: pl.LazyFrame) -> None:
    # only test raw output, otherwise we need graphviz and matplotlib
    out_ir = query.show_graph(raw_output=True, plan_stage="ir", engine="in-memory")
    out_phys = query.show_graph(
        raw_output=True, plan_stage="physical", engine="in-memory"
    )
    assert isinstance(out_ir, str)
    assert isinstance(out_phys, str)
    assert out_ir == out_phys


def test_show_graph_invalid_stage(query: pl.LazyFrame) -> None:
    with pytest.raises(TypeError, match="invalid plan stage 'invalid-stage'"):
        query.show_graph(raw_output=True, plan_stage="invalid-stage")  # type: ignore[call-overload]


@pytest.mark.write_disk
@pytest.mark.skipif(
    shutil.which("dot") is None, reason="graphviz `dot` binary is required"
)
def test_show_graph_output_path_os_pathlike_17828(
    query: pl.LazyFrame, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)
    output_path = tmp_path / "graph.png"

    # `output_path` must accept an `os.PathLike`; the `.svg` suffix check uses
    # `os.fspath`, not `str`.
    result = query.show_graph(show=False, output_path=HostilePathLike(output_path))

    assert result is None
    assert output_path.exists()
    assert output_path.stat().st_size > 0

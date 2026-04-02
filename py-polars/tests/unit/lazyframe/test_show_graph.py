import pytest

import polars as pl


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
        query.show_graph(raw_output=True, plan_stage="invalid-stage")  # type: ignore[arg-type]

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from deltalake import DeltaTable

import polars as pl
from polars.io.delta._scan_resolver import DeltaScanResolver
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=["plugin", "deltalake", "pyarrow"])
def planner(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "plugin":
        pytest.importorskip("polars_delta")
    monkeypatch.setenv(
        "POLARS_DELTA_PLANNER",
        "deltalake" if request.param == "pyarrow" else request.param,
    )
    return request.param


def test_resolver_operation_order_and_empty_projection(
    tmp_path: Path, planner: str
) -> None:
    frame = pl.DataFrame({"a": range(8), "b": range(8, 16)})
    frame.write_delta(tmp_path)
    scan = pl.scan_delta(tmp_path, use_pyarrow=planner == "pyarrow")
    for query, expected in [
        (scan.head(3).filter(pl.col("a") > 1), frame.head(3).filter(pl.col("a") > 1)),
        (scan.filter(pl.col("a") > 1).head(3), frame.filter(pl.col("a") > 1).head(3)),
        (scan.slice(2, 3), frame.slice(2, 3)),
        (scan.slice(-3, 2), frame.slice(-3, 2)),
        (
            scan.filter(pl.col("b") > 10).select("a"),
            frame.filter(pl.col("b") > 10).select("a"),
        ),
        (scan.select(pl.len()), pl.DataFrame({"len": [8]}, schema={"len": pl.UInt32})),
        (scan.select(pl.col("b").sum()), frame.select(pl.col("b").sum())),
    ]:
        assert_frame_equal(query.collect(), expected)
    assert scan.select(pl.len()).collect().item() == 8
    assert scan.drop("*").collect().height == frame.height


@pytest.mark.parametrize("operation", ["self_join", "concat"])
def test_resolver_shared_frame_single_resolution(
    tmp_path: Path, planner: str, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    frame = pl.DataFrame({"a": [1, 2, 3]})
    frame.write_delta(tmp_path)
    calls = []
    original = DeltaScanResolver.resolve_lazyframe

    def capture(self: DeltaScanResolver, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(DeltaScanResolver, "resolve_lazyframe", capture)
    scan = pl.scan_delta(tmp_path, use_pyarrow=planner == "pyarrow")
    query = (
        scan.join(scan, on="a") if operation == "self_join" else pl.concat([scan, scan])
    )
    assert query.collect().height == (3 if operation == "self_join" else 6)
    assert len(calls) == 1
    plan = query.explain()
    assert plan.count("CACHE[id:") == 2


def test_resolver_live_schema_change_rejected(tmp_path: Path, planner: str) -> None:
    pl.DataFrame({"a": [1]}).write_delta(tmp_path)
    table = DeltaTable(tmp_path)
    scan = pl.scan_delta(table, use_pyarrow=planner == "pyarrow")
    assert scan.collect_schema() == {"a": pl.Int64}
    pl.DataFrame({"a": ["changed"]}).write_delta(
        table, mode="overwrite", delta_write_options={"schema_mode": "overwrite"}
    )
    with pytest.raises(pl.exceptions.SchemaError, match="schema changed"):
        scan.collect()


def test_resolver_fallback_dtype_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("polars_delta")
    monkeypatch.delenv("POLARS_DELTA_PLANNER", raising=False)
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).cast({"a": pl.Int32})
    frame.write_delta(tmp_path)
    scan = pl.scan_delta(tmp_path)
    assert scan.collect_schema() == frame.schema

    def unsupported(*args: Any, **kwargs: Any) -> Any:
        msg = "supported fallback test"
        raise NotImplementedError(msg)

    monkeypatch.setattr("polars.io.delta._plugin.plugin_scan", unsupported)
    result = scan.collect()
    assert result.schema == frame.schema
    assert_frame_equal(result, frame)


def test_independent_plugin_capsule_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delta = pytest.importorskip("polars_delta")
    iceberg = pytest.importorskip("polars_iceberg")
    assert tuple(delta._polars_io_plugin_ids) == ("polars.io_plugin.delta.v2",)
    assert tuple(iceberg._polars_io_plugin_ids) == ("polars.io_plugin.iceberg.v1",)
    for plugin, wrong_id in [
        (delta, "polars.io_plugin.iceberg.v1"),
        (iceberg, "polars.io_plugin.delta.v2"),
    ]:
        with pytest.raises(ValueError):
            plugin._capsule(wrong_id)
    pl.DataFrame({"a": [1]}).write_delta(tmp_path)
    monkeypatch.setenv("POLARS_DELTA_PLANNER", "plugin")
    monkeypatch.setattr(
        delta, "_capsule", lambda _: iceberg._capsule("polars.io_plugin.iceberg.v1")
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="unsupported polars_delta plugin ID"
    ):
        pl.scan_delta(tmp_path).collect()


def test_resolver_runtime_filter_mixed_scan_and_repeated_collect(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame({"x": range(300)})
    frame.write_delta(tmp_path / "delta")
    frame.write_parquet(tmp_path / "data.parquet")
    query = pl.concat(
        [
            pl.scan_delta(tmp_path / "delta"),
            pl.scan_parquet(tmp_path / "data.parquet"),
        ]
    ).join(pl.LazyFrame({"x": [150, 160]}).filter(pl.col("x") > 0), on="x")
    assert query.explain(engine="streaming").count("dynamic_predicate()") == 2
    expected = pl.DataFrame({"x": [150, 150, 160, 160]})
    for _ in range(2):
        assert_frame_equal(query.collect(engine="streaming").sort("x"), expected)
        assert query.explain(engine="streaming").count("dynamic_predicate()") == 2


def test_resolver_shared_scan_independent_runtime_filters(tmp_path: Path) -> None:
    frame = pl.DataFrame({"x": range(300)})
    frame.write_delta(tmp_path)
    scan = pl.scan_delta(tmp_path)
    left = scan.join(pl.LazyFrame({"x": [150, 160]}).filter(pl.col("x") > 0), on="x")
    right = scan.join(pl.LazyFrame({"x": [50, 60]}).filter(pl.col("x") > 0), on="x")
    query = pl.concat([left, right])
    expected = pl.DataFrame({"x": [50, 60, 150, 160]})
    for _ in range(2):
        assert_frame_equal(query.collect(engine="streaming").sort("x"), expected)
    assert query.explain(engine="streaming").count("CACHE[id:") == 2


def test_resolver_metadata_count_needs_no_data_file(
    tmp_path: Path, planner: str
) -> None:
    if planner == "pyarrow":
        pytest.skip("PyArrow reads its own file metadata")
    frame = pl.DataFrame({"a": range(8)})
    frame.write_delta(tmp_path)
    scan = pl.scan_delta(tmp_path)
    scan.collect_schema()
    for path in tmp_path.glob("*.parquet"):
        path.unlink()
    assert scan.select(pl.len()).collect().item() == 8


@pytest.mark.parametrize(
    "counts", [[None], [-1], ["bad"], [True], [1 << 64], [1, None]]
)
def test_resolver_ignores_incomplete_or_malformed_metadata_counts(
    counts: list[Any],
) -> None:
    from polars.io.delta._scan_resolver import _metadata_row_count

    # Object dtype permits values outside the integer contract in this unit probe.
    statistics = pl.DataFrame({"len": pl.Series(counts, dtype=pl.Object)})
    assert (
        _metadata_row_count(statistics, len(counts), None, has_deletion_vectors=False)
        is None
    )


def test_resolver_metadata_count_completeness_and_deletions() -> None:
    from polars.io.delta._scan_resolver import _metadata_row_count

    assert _metadata_row_count(None, 1, None, has_deletion_vectors=False) is None
    assert (
        _metadata_row_count(
            pl.DataFrame({"other": [1]}), 1, None, has_deletion_vectors=False
        )
        is None
    )
    assert (
        _metadata_row_count(
            pl.DataFrame({"len": [8]}), 2, None, has_deletion_vectors=False
        )
        is None
    )
    assert (
        _metadata_row_count(
            pl.DataFrame({"len": [None]}), 1, None, has_deletion_vectors=False
        )
        is None
    )
    assert _metadata_row_count(
        pl.DataFrame({"len": []}), 0, None, has_deletion_vectors=False
    ) == (0, 0)
    assert (
        _metadata_row_count(
            pl.DataFrame({"len": [8]}), 1, None, has_deletion_vectors=True
        )
        is None
    )
    vectors = ("delta-deletion-vector-native", {0: (b"", 0, 0, 8)})
    assert _metadata_row_count(
        pl.DataFrame({"len": [8]}), 1, vectors, has_deletion_vectors=True
    ) == (8, 8)
    with pytest.raises(pl.exceptions.ComputeError, match="cardinality exceeds"):
        _metadata_row_count(
            pl.DataFrame({"len": [7]}), 1, vectors, has_deletion_vectors=True
        )

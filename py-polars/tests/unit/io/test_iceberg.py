# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path

import pytest

import polars as pl
from polars.io.iceberg import _convert_predicate, _to_ast


@pytest.fixture()
def iceberg_path(io_files_path: Path) -> str:
    # Iceberg requires absolute paths, so we'll symlink
    # the test table into /tmp/iceberg/t1/
    Path("/tmp/iceberg").mkdir(parents=True, exist_ok=True)
    current_path = Path(__file__).parent.resolve()

    with contextlib.suppress(FileExistsError):
        os.symlink(f"{current_path}/files/iceberg-table", "/tmp/iceberg/t1")

    iceberg_path = io_files_path / "iceberg-table" / "metadata" / "v2.metadata.json"
    return f"file://{iceberg_path.resolve()}"


@pytest.mark.slow()
@pytest.mark.write_disk()
@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
@pytest.mark.ci_only()
class TestIcebergScanIO:
    """Test coverage for `iceberg` scan ops."""

    def test_scan_iceberg_plain(self, iceberg_path: str) -> None:
        df = pl.scan_iceberg(iceberg_path)
        assert len(df.collect()) == 3
        assert df.collect_schema() == {
            "id": pl.Int32,
            "str": pl.String,
            "ts": pl.Datetime(time_unit="us", time_zone=None),
        }

    def test_scan_iceberg_filter_on_partition(self, iceberg_path: str) -> None:
        ts1 = datetime(2023, 3, 1, 18, 15)
        ts2 = datetime(2023, 3, 1, 19, 25)
        ts3 = datetime(2023, 3, 2, 22, 0)

        lf = pl.scan_iceberg(iceberg_path)

        res = lf.filter(pl.col("ts") >= ts2)
        assert len(res.collect()) == 2

        res = lf.filter(pl.col("ts") > ts2).select(pl.col("id"))
        assert res.collect().rows() == [(3,)]

        res = lf.filter(pl.col("ts") <= ts2).select("id", "ts")
        assert res.collect().rows(named=True) == [
            {"id": 1, "ts": ts1},
            {"id": 2, "ts": ts2},
        ]

        res = lf.filter(pl.col("ts") > ts3)
        assert len(res.collect()) == 0

        for constraint in (
            (pl.col("ts") == ts1) | (pl.col("ts") == ts3),
            pl.col("ts").is_in([ts1, ts3]),
        ):
            res = lf.filter(constraint).select("id")
            assert res.collect().rows() == [(1,), (3,)]

    def test_scan_iceberg_filter_on_column(self, iceberg_path: str) -> None:
        lf = pl.scan_iceberg(iceberg_path)
        res = lf.filter(pl.col("id") < 2)
        assert res.collect().rows() == [(1, "1", datetime(2023, 3, 1, 18, 15))]

        res = lf.filter(pl.col("id") == 2)
        assert res.collect().rows() == [(2, "2", datetime(2023, 3, 1, 19, 25))]

        res = lf.filter(pl.col("id").is_in([1, 3]))
        assert res.collect().rows() == [
            (1, "1", datetime(2023, 3, 1, 18, 15)),
            (3, "3", datetime(2023, 3, 2, 22, 0)),
        ]


@pytest.mark.ci_only()
class TestIcebergExpressions:
    """Test coverage for `iceberg` expressions comprehension."""

    def test_is_null_expression(self) -> None:
        from pyiceberg.expressions import IsNull

        expr = _to_ast("(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == IsNull("id")

    def test_is_not_null_expression(self) -> None:
        from pyiceberg.expressions import IsNull, Not

        expr = _to_ast("~(pa.compute.field('id')).is_null()")
        assert _convert_predicate(expr) == Not(IsNull("id"))

    def test_isin_expression(self) -> None:
        from pyiceberg.expressions import In, literal

        expr = _to_ast("(pa.compute.field('id')).isin([1,2,3])")
        assert _convert_predicate(expr) == In(
            "id", {literal(1), literal(2), literal(3)}
        )

    def test_parse_combined_expression(self) -> None:
        from pyiceberg.expressions import (
            And,
            EqualTo,
            GreaterThan,
            In,
            Or,
            Reference,
            literal,
        )

        expr = _to_ast(
            "(((pa.compute.field('str') == '2') & (pa.compute.field('id') > 10)) | (pa.compute.field('id')).isin([1,2,3]))"
        )
        assert _convert_predicate(expr) == Or(
            left=And(
                left=EqualTo(term=Reference(name="str"), literal=literal("2")),
                right=GreaterThan(term="id", literal=literal(10)),
            ),
            right=In("id", {literal(1), literal(2), literal(3)}),
        )

    def test_parse_gt(self) -> None:
        from pyiceberg.expressions import GreaterThan

        expr = _to_ast("(pa.compute.field('ts') > '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThan("ts", "2023-08-08")

    def test_parse_gteq(self) -> None:
        from pyiceberg.expressions import GreaterThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') >= '2023-08-08')")
        assert _convert_predicate(expr) == GreaterThanOrEqual("ts", "2023-08-08")

    def test_parse_eq(self) -> None:
        from pyiceberg.expressions import EqualTo

        expr = _to_ast("(pa.compute.field('ts') == '2023-08-08')")
        assert _convert_predicate(expr) == EqualTo("ts", "2023-08-08")

    def test_parse_lt(self) -> None:
        from pyiceberg.expressions import LessThan

        expr = _to_ast("(pa.compute.field('ts') < '2023-08-08')")
        assert _convert_predicate(expr) == LessThan("ts", "2023-08-08")

    def test_parse_lteq(self) -> None:
        from pyiceberg.expressions import LessThanOrEqual

        expr = _to_ast("(pa.compute.field('ts') <= '2023-08-08')")
        assert _convert_predicate(expr) == LessThanOrEqual("ts", "2023-08-08")

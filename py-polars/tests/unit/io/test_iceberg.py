from __future__ import annotations

import contextlib
import os
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


@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
def test_scan_iceberg_plain(iceberg_path: str) -> None:
    df = pl.scan_iceberg(iceberg_path)
    assert len(df.collect()) == 3
    assert df.schema == {
        "id": pl.Int32,
        "str": pl.Utf8,
        "ts": pl.Datetime(time_unit="us", time_zone=None),
    }


@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
def test_scan_iceberg_filter_on_partition(iceberg_path: str) -> None:
    df = pl.scan_iceberg(iceberg_path)
    df = df.filter(pl.col("ts") > "2023-03-02T00:00:00")
    assert len(df.collect()) == 1


@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
def test_scan_iceberg_filter_on_column(iceberg_path: str) -> None:
    df = pl.scan_iceberg(iceberg_path)
    df = df.filter(pl.col("id") < 2)
    assert len(df.collect()) == 1


def test_is_null_expression() -> None:
    from pyiceberg.expressions import IsNull

    expr = _to_ast("(pa.compute.field('id')).is_null()")
    assert _convert_predicate(expr) == IsNull("id")


def test_is_not_null_expression() -> None:
    from pyiceberg.expressions import IsNull, Not

    expr = _to_ast("~(pa.compute.field('id')).is_null()")
    assert _convert_predicate(expr) == Not(IsNull("id"))


def test_isin_expression() -> None:
    from pyiceberg.expressions import In, literal  # type: ignore[attr-defined]

    expr = _to_ast("(pa.compute.field('id')).isin([1,2,3])")
    assert _convert_predicate(expr) == In("id", {literal(1), literal(2), literal(3)})


def test_parse_combined_expression() -> None:
    from pyiceberg.expressions import (  # type: ignore[attr-defined]
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


def test_parse_gt() -> None:
    from pyiceberg.expressions import GreaterThan

    expr = _to_ast("(pa.compute.field('ts') > '2023-08-08')")
    assert _convert_predicate(expr) == GreaterThan("ts", "2023-08-08")


def test_parse_gteq() -> None:
    from pyiceberg.expressions import GreaterThanOrEqual

    expr = _to_ast("(pa.compute.field('ts') >= '2023-08-08')")
    assert _convert_predicate(expr) == GreaterThanOrEqual("ts", "2023-08-08")


def test_parse_eq() -> None:
    from pyiceberg.expressions import EqualTo

    expr = _to_ast("(pa.compute.field('ts') == '2023-08-08')")
    assert _convert_predicate(expr) == EqualTo("ts", "2023-08-08")


def test_parse_lt() -> None:
    from pyiceberg.expressions import LessThan

    expr = _to_ast("(pa.compute.field('ts') < '2023-08-08')")
    assert _convert_predicate(expr) == LessThan("ts", "2023-08-08")


def test_parse_lteq() -> None:
    from pyiceberg.expressions import LessThanOrEqual

    expr = _to_ast("(pa.compute.field('ts') <= '2023-08-08')")
    assert _convert_predicate(expr) == LessThanOrEqual("ts", "2023-08-08")

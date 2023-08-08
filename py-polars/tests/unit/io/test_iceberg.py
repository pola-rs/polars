from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

import polars as pl
from polars.io.iceberg import _convert_predicate, _to_ast


@pytest.fixture()
def iceberg_path() -> str:
    # Iceberg requires absolute paths, so we'll symlink
    # the test table into /tmp/iceberg/t1/
    Path("/tmp/iceberg").mkdir(parents=True, exist_ok=True)
    current_path = Path.cwd()
    with contextlib.suppress(FileExistsError):
        os.symlink(f"{current_path}/files/iceberg-table", "/tmp/iceberg/t1")

    return "file:///tmp/iceberg/t1/metadata/00001-55cdf97b-255c-4983-b9f3-0e468fadfe9e.metadata.json"


def test_scan_iceberg_plain(iceberg_path: str) -> None:
    df = pl.scan_iceberg(iceberg_path)
    assert len(df.collect()) == 233822


def test_scan_iceberg_filter_on_partition(iceberg_path: str) -> None:
    df = pl.scan_iceberg(iceberg_path)
    df = df.filter(pl.col("tpep_pickup_datetime") >= "2022-03-02T00:00:00+00:00")
    assert len(df.collect()) == 119863


def test_scan_iceberg_filter_on_column(iceberg_path: str) -> None:
    df = pl.scan_iceberg(iceberg_path)
    df = df.filter(pl.col("fare_amount") < 0.0)
    assert len(df.collect()) == 1192


def test_true_expression() -> None:
    from pyiceberg.expressions import (
        AlwaysTrue,
    )

    expr = _to_ast("pa.compute.scalar(True)")
    assert _convert_predicate(expr) == AlwaysTrue()


def test_false_expression() -> None:
    from pyiceberg.expressions import (
        AlwaysFalse,
    )

    expr = _to_ast("pa.compute.scalar(False)")
    assert _convert_predicate(expr) == AlwaysFalse()


def test_is_null_expression() -> None:
    from pyiceberg.expressions import (
        IsNull,
    )

    expr = _to_ast("(pa.compute.field('borough')).is_null()")
    assert _convert_predicate(expr) == IsNull("borough")


def test_is_not_null_expression() -> None:
    from pyiceberg.expressions import (
        IsNull,
        Not,
    )

    expr = _to_ast("~(pa.compute.field('location_id')).is_null()")
    assert _convert_predicate(expr) == Not(IsNull("location_id"))


def test_is_nan_expression() -> None:
    from pyiceberg.expressions import (
        IsNaN,
    )

    expr = _to_ast("(pa.compute.field('borough')).is_nan()")
    assert _convert_predicate(expr) == IsNaN("borough")


def test_is_not_nan_expression() -> None:
    from pyiceberg.expressions import (
        IsNaN,
        Not,
    )

    expr = _to_ast("~(pa.compute.field('location_id')).is_nan()")
    assert _convert_predicate(expr) == Not(IsNaN("location_id"))


def test_isin_expression() -> None:
    from pyiceberg.expressions import (
        In,
        literal,
    )

    expr = _to_ast("(pa.compute.field('location_id')).isin([1,2,3])")
    assert _convert_predicate(expr) == In(
        "location_id", {literal(1), literal(2), literal(3)}
    )


def test_parse_combined_expression() -> None:
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
        "(((pa.compute.field('borough') == 'Manhattan') & (pa.compute.field('location_id') > 10)) | (pa.compute.field('location_id')).isin([1,2,3]))"
    )
    assert _convert_predicate(expr) == Or(
        left=And(
            left=EqualTo(term=Reference(name="borough"), literal=literal("Manhattan")),
            right=GreaterThan(term="location_id", literal=literal(10)),
        ),
        right=In("location_id", {literal(1), literal(2), literal(3)}),
    )


def test_parse_gt() -> None:
    from pyiceberg.expressions import (
        GreaterThan,
    )

    expr = _to_ast("(pa.compute.field('dt') > '2023-08-08')")
    assert _convert_predicate(expr) == GreaterThan("dt", "2023-08-08")


def test_parse_gteq() -> None:
    from pyiceberg.expressions import (
        GreaterThanOrEqual,
    )

    expr = _to_ast("(pa.compute.field('dt') >= '2023-08-08')")
    assert _convert_predicate(expr) == GreaterThanOrEqual("dt", "2023-08-08")


def test_parse_eq() -> None:
    from pyiceberg.expressions import (
        EqualTo,
    )

    expr = _to_ast("(pa.compute.field('dt') == '2023-08-08')")
    assert _convert_predicate(expr) == EqualTo("dt", "2023-08-08")


def test_parse_lt() -> None:
    from pyiceberg.expressions import (
        LessThan,
    )

    expr = _to_ast("(pa.compute.field('dt') < '2023-08-08')")
    assert _convert_predicate(expr) == LessThan("dt", "2023-08-08")


def test_parse_lteq() -> None:
    from pyiceberg.expressions import (
        LessThanOrEqual,
    )

    expr = _to_ast("(pa.compute.field('dt') <= '2023-08-08')")
    assert _convert_predicate(expr) == LessThanOrEqual("dt", "2023-08-08")

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from tests.unit.utils.pathlike import HostilePathLike

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("foo").sum().over("bar"),
        pl.col("foo").rolling_quantile(0.25, window_size=5),
        pl.col("foo").rolling_var(window_size=4, ddof=2),
        pl.col("foo").rolling_min(window_size=2),
        pl.col("foo").rolling_quantile_by("bar", window_size="1mo", quantile=0.75),
    ],
)
def test_expr_serde_roundtrip_binary(expr: pl.Expr) -> None:
    json = expr.meta.serialize(format="binary")
    round_tripped = pl.Expr.deserialize(io.BytesIO(json), format="binary")
    assert round_tripped.meta == expr


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("foo").sum().over("bar"),
        pl.col("foo").rolling_quantile(0.25, window_size=5),
        pl.col("foo").rolling_var(window_size=4, ddof=2),
        pl.col("foo").rolling_min(window_size=2),
        pl.col("foo").rolling_quantile_by("bar", window_size="1mo", quantile=0.75),
    ],
)
def test_expr_serde_roundtrip_json(expr: pl.Expr) -> None:
    expr = pl.col("foo").sum().over("bar")
    json = expr.meta.serialize(format="json")
    round_tripped = pl.Expr.deserialize(io.StringIO(json), format="json")
    assert round_tripped.meta == expr


def test_expr_deserialize_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        pl.Expr.deserialize("abcdef")


@pytest.mark.write_disk
def test_expr_serde_to_from_os_pathlike_17828(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    expr = pl.col("foo").sum().over("bar")
    file_path = tmp_path / "expr.bin"
    expr.meta.serialize(HostilePathLike(file_path))
    round_tripped = pl.Expr.deserialize(HostilePathLike(file_path))

    assert round_tripped.meta == expr


def test_expr_deserialize_invalid_json() -> None:
    with pytest.raises(
        ComputeError, match="could not deserialize input into an expression"
    ):
        pl.Expr.deserialize(io.StringIO("abcdef"), format="json")


def test_expression_json_13991() -> None:
    expr = pl.col("foo").cast(pl.Decimal(38, 10))
    json = expr.meta.serialize(format="json")

    round_tripped = pl.Expr.deserialize(io.StringIO(json), format="json")
    assert round_tripped.meta == expr


def test_expr_write_json_from_json_deprecated() -> None:
    expr = pl.col("foo").sum().over("bar")

    with pytest.deprecated_call():
        json = expr.meta.write_json()

    with pytest.deprecated_call():
        round_tripped = pl.Expr.from_json(json)

    assert round_tripped.meta == expr

import io

import pytest

import polars as pl
from polars.exceptions import ComputeError


def test_expr_serialization_roundtrip() -> None:
    expr = pl.col("foo").sum().over("bar")
    json = expr.meta.serialize()
    round_tripped = pl.Expr.deserialize(io.StringIO(json))
    assert round_tripped.meta == expr


def test_expr_deserialize_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        pl.Expr.deserialize("abcdef")


def test_expr_deserialize_invalid_json() -> None:
    with pytest.raises(
        ComputeError, match="could not deserialize input into an expression"
    ):
        pl.Expr.deserialize(io.StringIO("abcdef"))


def test_expr_write_json_from_json_deprecated() -> None:
    expr = pl.col("foo").sum().over("bar")

    with pytest.deprecated_call():
        json = expr.meta.write_json()

    with pytest.deprecated_call():
        round_tripped = pl.Expr.from_json(json)

    assert round_tripped.meta == expr


def test_expression_json_13991() -> None:
    expr = pl.col("foo").cast(pl.Decimal)
    json = expr.meta.serialize()

    round_tripped = pl.Expr.deserialize(io.StringIO(json))
    assert round_tripped.meta == expr

from __future__ import annotations

import pytest

import polars as pl


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [20, 32, 50, 88, 128],
            "y": [-128, 0, 10, -1, None],
        }
    )


def test_bitwise_and(df: pl.DataFrame) -> None:
    res = df.sql(
        """
        SELECT
          x & y AS x_bitand_op_y,
          BITAND(y, x) AS y_bitand_x,
          BIT_AND(x, y) AS x_bitand_y,
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "x_bitand_op_y": [0, 0, 2, 88, None],
        "y_bitand_x": [0, 0, 2, 88, None],
        "x_bitand_y": [0, 0, 2, 88, None],
    }


def test_bitwise_count(df: pl.DataFrame) -> None:
    res = df.sql(
        """
        SELECT
          BITCOUNT(x) AS x_bits_set,
          BIT_COUNT(y) AS y_bits_set,
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "x_bits_set": [2, 1, 3, 3, 1],
        "y_bits_set": [57, 0, 2, 64, None],
    }


def test_bitwise_or(df: pl.DataFrame) -> None:
    res = df.sql(
        """
        SELECT
          x | y AS x_bitor_op_y,
          BITOR(y, x) AS y_bitor_x,
          BIT_OR(x, y) AS x_bitor_y,
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "x_bitor_op_y": [-108, 32, 58, -1, None],
        "y_bitor_x": [-108, 32, 58, -1, None],
        "x_bitor_y": [-108, 32, 58, -1, None],
    }


def test_bitwise_xor(df: pl.DataFrame) -> None:
    res = df.sql(
        """
        SELECT
          x XOR y AS x_bitxor_op_y,
          BITXOR(y, x) AS y_bitxor_x,
          BIT_XOR(x, y) AS x_bitxor_y,
        FROM self
        """
    )
    assert res.to_dict(as_series=False) == {
        "x_bitxor_op_y": [-108, 32, 56, -89, None],
        "y_bitxor_x": [-108, 32, 56, -89, None],
        "x_bitxor_y": [-108, 32, 56, -89, None],
    }

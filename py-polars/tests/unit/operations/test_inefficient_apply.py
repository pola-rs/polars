from __future__ import annotations

from typing import Any, Callable
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest

import polars as pl
from polars.exceptions import PolarsInefficientApplyWarning
from polars.utils.udfs import (
    _get_bytecode_ops,
    _is_inefficient,
    _rewrite_as_polars_expr,
    _simple_signature,
)


def _get_suggestion(
    func: Callable[[Any], Any], col: str, apply_target: str
) -> str | None:
    return _rewrite_as_polars_expr(_get_bytecode_ops(func), col, apply_target)


@pytest.mark.parametrize(
    "func",
    [
        np.sin,
        lambda x: np.sin(x),
        lambda x, y: x + y,
    ],
)
def test_non_simple_function(func: Callable[[Any], Any]) -> None:
    assert not _simple_signature(func) or not _is_inefficient(
        _get_bytecode_ops(func), apply_target="expr"
    )


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x,
        lambda x: x + 1 - (2 / 3),
        lambda x: x // 1 % 2,
        lambda x: x & True,
        lambda x: x | False,
        lambda x: x != 3,
        lambda x: x > 1,
        lambda x: not (x > 1),
        lambda x: x is None,
        lambda x: x is not None,
        lambda x: (x * x) ** x,
        lambda x: x * (x**x),
        lambda x: (x / x) + ((x * x) - x),
        lambda x: (10 - x) / (((x * 4) - x) // (2 + (x * (x - 1)))),
        lambda x: x in (2, 3, 4),
        lambda x: x not in (2, 3, 4),
    ],
)
def test_expr_apply_produces_warning(func: Callable[[Any], Any]) -> None:
    with catch_warnings():
        simplefilter("ignore", PolarsInefficientApplyWarning)

        suggestion = _get_suggestion(func, col="a", apply_target="expr")
        assert suggestion is not None

        df = pl.DataFrame({"a": [1, 2, 3]})
        result = df.select(
            x="a",
            y=eval(suggestion),
        )
        expected = df.select(
            x=pl.col("a"),
            y=pl.col("a").apply(func),
        )
        assert result.rows() == expected.rows()


def test_expr_apply_produces_warning_misc() -> None:
    # note: can also identify inefficient functions and methods as well as lambdas
    class Test:
        def x10(self, x: pl.Expr) -> pl.Expr:
            return x * 10

    suggestion = _get_suggestion(Test().x10, col="colx", apply_target="expr")
    assert suggestion == '(pl.col("colx") * 10)'

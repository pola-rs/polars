from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_foldv_runs_in_an_eager_query() -> None:
    # Eager selection executes the JIT expression and returns its scalar state.
    df = pl.DataFrame({"l": [1, 2, 3]})
    expr = pl.foldv(lambda state: state * 10 + pl.col("l"), initial_state=0).alias(
        "result"
    )
    expected = pl.DataFrame({"result": [1, 12, 123]})

    assert_frame_equal(df.select(expr), expected)


def test_foldv_runs_in_a_lazy_query() -> None:
    # Lazy selection executes the same ordered state updates during collection.
    df = pl.DataFrame({"l": [1, 2, 3]})
    expr = pl.foldv(lambda state: state * 10 + pl.col("l"), initial_state=0).alias(
        "result"
    )
    expected = pl.DataFrame({"result": [1, 12, 123]})

    assert_frame_equal(df.lazy().select(expr).collect(), expected)


def test_foldv_calls_the_python_function_once_during_construction() -> None:
    # Native execution reuses the symbolic body without calling Python per row.
    calls = 0

    def body(state: pl.Expr) -> pl.Expr:
        nonlocal calls
        calls += 1
        return state + pl.col("l")

    expr = pl.foldv(body, initial_state=10)
    assert calls == 1
    assert pl.DataFrame({"l": [1, 2, 3]}).select(expr).to_series().to_list() == [
        11,
        13,
        16,
    ]
    assert calls == 1


def test_foldv_returns_an_empty_result_for_empty_input() -> None:
    # An empty input never evaluates the body.
    df = pl.DataFrame({"l": pl.Series([], dtype=pl.Int64)})
    result = df.select(pl.foldv(lambda state: state + pl.col("l"), initial_state=42))

    assert result.is_empty()


def test_foldv_requires_the_function_to_return_an_expression() -> None:
    # Invalid callback results fail while constructing the expression.
    with pytest.raises(TypeError, match="must return a Polars expression"):
        pl.foldv(lambda state: 1, initial_state=0)  # type: ignore[return-value]

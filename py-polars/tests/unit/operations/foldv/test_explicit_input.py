from __future__ import annotations

import pytest

import polars as pl


def test_expr_foldv_runs_in_an_eager_query() -> None:
    # The method form passes the current row value as the callback's second expression.
    result = pl.DataFrame({"l": [1, 2, 3]}).select(
        pl.col("l").foldv(
            lambda state, value: state * 10 + value,
            initial_state=0,
        )
    )

    assert result.to_series().to_list() == [1, 12, 123]


def test_expr_foldv_runs_in_a_lazy_query() -> None:
    # Lazy execution preserves the same ordered state updates as eager execution.
    result = (
        pl.DataFrame({"l": [1, 2, 3]})
        .lazy()
        .select(
            pl.col("l").foldv(
                lambda state, value: state + value,
                initial_state=10,
            )
        )
        .collect()
    )

    assert result.to_series().to_list() == [11, 13, 16]


def test_expr_foldv_folds_a_chained_input_expression() -> None:
    # The value callback receives the complete input expression.
    result = pl.DataFrame({"l": [1, 2, 3]}).select(
        (pl.col("l") * 2).foldv(
            lambda state, value: state + value,
            initial_state=0,
        )
    )

    assert result.to_series().to_list() == [2, 6, 12]


def test_expr_foldv_can_reference_an_additional_column() -> None:
    # Method callbacks can combine their explicit value with other row inputs.
    result = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).select(
        pl.col("a").foldv(
            lambda state, value: state + value * pl.col("b"),
            initial_state=0,
        )
    )

    assert result.to_series().to_list() == [4, 14, 32]


def test_expr_foldv_calls_the_callback_once() -> None:
    # Construction invokes Python once to create the symbolic state-update expression.
    calls = 0

    def body(state: pl.Expr, value: pl.Expr) -> pl.Expr:
        nonlocal calls
        calls += 1
        return state + value

    expression = pl.col("l").foldv(body, initial_state=0)

    assert calls == 1
    assert pl.DataFrame({"l": [1, 2, 3]}).select(expression).to_series().to_list() == [
        1,
        3,
        6,
    ]
    assert calls == 1


def test_expr_foldv_runs_once_per_input_row_when_the_value_is_unused() -> None:
    # The input sets the number of rows even if the callback never reads its value.
    df = pl.DataFrame({"l": [10, 20, 30]})
    result = df.select(
        pl.col("l").foldv(lambda state, value: state + 1, initial_state=0)
    )

    assert result.to_series().to_list() == [1, 2, 3]


def test_expr_foldv_is_named_after_its_input() -> None:
    # Like other expression methods, the result keeps the input's name.
    df = pl.DataFrame({"l": [1, 2, 3]})
    expression = pl.col("l").foldv(lambda state, value: state + value, initial_state=0)

    assert df.select(expression).columns == ["l"]
    assert df.lazy().select(expression).collect_schema().names() == ["l"]


def test_expr_foldv_rejects_another_column_of_a_different_length() -> None:
    # A shortened input cannot be combined row by row with a full-length column.
    df = pl.DataFrame({"l": [1, 2, 3], "b": [1, 1, 1]})
    expression = (
        pl.col("l")
        .head(2)
        .foldv(
            lambda state, value: state + value + pl.col("b"),
            initial_state=0,
        )
    )

    with pytest.raises(pl.exceptions.ShapeError):
        df.select(expression)


def test_expr_foldv_supports_output_aliases() -> None:
    # Aliasing the scalar expression controls its result-column name.
    result = pl.DataFrame({"l": [1, 2, 3]}).select(
        pl.col("l")
        .foldv(lambda state, value: state + value, initial_state=0)
        .alias("total")
    )

    assert result.columns == ["total"]
    assert result.to_series().to_list() == [1, 3, 6]


def test_expr_foldv_requires_the_callback_to_return_an_expression() -> None:
    # Invalid callback results fail while constructing the method expression.
    with pytest.raises(TypeError, match="must return a Polars expression"):
        pl.col("l").foldv(
            lambda state, value: 1,  # type: ignore[return-value]
            initial_state=0,
        )


def test_expr_foldv_rejects_a_non_scalar_initial_state() -> None:
    # A state column with one value per row cannot initialize a scalar fold.
    df = pl.DataFrame({"l": [1, 2, 3], "state": [0, 0, 0]})
    expression = pl.col("l").foldv(
        lambda state, value: state + value,
        initial_state=pl.col("state"),
    )

    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="initial state must contain exactly one value",
    ):
        df.select(expression)

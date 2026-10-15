from __future__ import annotations

import polars as pl


def test_foldv_applies_conditional_updates() -> None:
    # The condition is evaluated against each row inside the generated loop.
    result = pl.DataFrame({"l": [-5, 2, -3, 7]}).select(
        pl.foldv(
            lambda state: pl.when(pl.col("l") > 0)
            .then(state + pl.col("l"))
            .otherwise(state),
            s=10,
        )
    )

    assert result.to_series().to_list() == [10, 12, 12, 19]


def test_foldv_supports_nested_state_dependent_conditions() -> None:
    # Nested branches can depend on both the current row and carried state.
    result = pl.DataFrame({"l": [1, 4, 9]}).select(
        pl.foldv(
            lambda state: pl.when((pl.col("l") > 0) & (state <= 5))
            .then(
                pl.when(pl.col("l") >= 3)
                .then(state + pl.col("l"))
                .otherwise(state + 1)
            )
            .otherwise(state),
            s=0,
        )
    )

    assert result.to_series().to_list() == [1, 5, 14]

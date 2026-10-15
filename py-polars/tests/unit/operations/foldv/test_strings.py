from __future__ import annotations

import pytest

import polars as pl


def test_foldv_concatenates_rows_into_a_string_state() -> None:
    # Each row's state is a new string built from the previous state.
    df = pl.DataFrame({"s": ["a", "é", "", "日本"]})
    result = df.select(pl.foldv(lambda state: state + pl.col("s"), initial_state=""))

    assert result.to_series().to_list() == ["a", "aé", "aé", "aé日本"]


def test_expr_foldv_folds_over_a_string_column() -> None:
    # The method form reads each string through its value argument.
    df = pl.DataFrame({"s": ["x", "y", "z"]})
    result = df.select(
        pl.col("s").foldv(lambda state, value: value + state, initial_state="")
    )

    assert result.columns == ["s"]
    assert result.to_series().to_list() == ["x", "yx", "zyx"]


def test_foldv_keeps_the_longest_string() -> None:
    # The state is replaced by an input value whenever that value is longer.
    df = pl.DataFrame({"s": ["ab", "x", "日本語", "abcd"]})
    expression = pl.foldv(
        lambda state: (
            pl.when(pl.col("s").str.len_chars() > state.str.len_chars())
            .then(pl.col("s"))
            .otherwise(state)
        ),
        initial_state="",
    )

    assert df.select(expression).to_series().to_list() == ["ab", "ab", "日本語", "abcd"]


def test_foldv_string_state_propagates_and_recovers_from_null() -> None:
    # A null input makes the concatenation null until the body restarts the state.
    df = pl.DataFrame({"s": ["a", None, "b"]})
    expression = pl.foldv(
        lambda state: (
            pl.when(state.is_null()).then(pl.col("s")).otherwise(state + pl.col("s"))
        ),
        initial_state="",
    )

    assert df.select(expression).to_series().to_list() == ["a", None, "b"]


def test_foldv_rejects_casts_between_strings_and_numbers() -> None:
    # String casts are not supported yet and fail with a clear error.
    df = pl.DataFrame({"s": ["1"]})
    expression = pl.foldv(
        lambda state: state + pl.col("s").cast(pl.Int64), initial_state=0
    )

    with pytest.raises(pl.exceptions.InvalidOperationError, match="casts from str"):
        df.select(expression)


def test_foldv_treats_a_string_initial_state_as_a_value() -> None:
    # The initial state is a value: a Python string is the string itself, even when a
    # column has that name, while `pl.col` still reads a column.
    df = pl.DataFrame({"s": ["x", "y"], "start": ["column", "unused"]})
    as_value = pl.foldv(lambda state: state + pl.col("s"), initial_state="start")
    as_column = pl.foldv(
        lambda state: state + pl.col("s"), initial_state=pl.col("start").first()
    )

    assert df.select(as_value).to_series().to_list() == ["startx", "startxy"]
    assert df.select(as_column).to_series().to_list() == ["columnx", "columnxy"]

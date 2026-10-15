from __future__ import annotations

import polars as pl


def test_foldv_carries_an_int32_state() -> None:
    # The Python API preserves an explicitly typed four-byte signed state.
    result = pl.DataFrame({"l": pl.Series([1, 2, 3], dtype=pl.Int32)}).select(
        pl.foldv(
            lambda state: state + pl.col("l"),
            s=pl.lit(0, dtype=pl.Int32),
        )
    )

    assert result.to_series().dtype == pl.Int32
    assert result.to_series().to_list() == [1, 3, 6]


def test_foldv_carries_a_uint32_state() -> None:
    # The Python API preserves an explicitly typed four-byte unsigned state.
    result = pl.DataFrame({"l": pl.Series([1, 2, 3], dtype=pl.UInt32)}).select(
        pl.foldv(
            lambda state: state + pl.col("l"),
            s=pl.lit(0, dtype=pl.UInt32),
        )
    )

    assert result.to_series().dtype == pl.UInt32
    assert result.to_series().to_list() == [1, 3, 6]


def test_foldv_carries_a_uint64_state() -> None:
    # The Python API preserves an explicitly typed eight-byte unsigned state.
    result = pl.DataFrame({"l": pl.Series([1, 2, 3], dtype=pl.UInt64)}).select(
        pl.foldv(
            lambda state: state + pl.col("l"),
            s=pl.lit(0, dtype=pl.UInt64),
        )
    )

    assert result.to_series().dtype == pl.UInt64
    assert result.to_series().to_list() == [1, 3, 6]


def test_foldv_carries_a_float32_state() -> None:
    # The Python API preserves Float32 calculations across every iteration.
    result = pl.DataFrame({"l": pl.Series([1.0, 2.5, 3.0], dtype=pl.Float32)}).select(
        pl.foldv(
            lambda state: state + pl.col("l"),
            s=pl.lit(0.5, dtype=pl.Float32),
        )
    )

    assert result.to_series().dtype == pl.Float32
    assert result.to_series().to_list() == [1.5, 4.0, 7.0]


def test_foldv_carries_a_float64_state() -> None:
    # The Python API preserves Float64 calculations across every iteration.
    result = pl.DataFrame({"l": pl.Series([1.0, 2.5, 3.0], dtype=pl.Float64)}).select(
        pl.foldv(lambda state: state + pl.col("l"), s=0.5)
    )

    assert result.to_series().dtype == pl.Float64
    assert result.to_series().to_list() == [1.5, 4.0, 7.0]


def test_foldv_carries_a_boolean_state() -> None:
    # Boolean state updates use the same public expression API as numeric updates.
    result = pl.DataFrame({"l": [True, True, False]}).select(
        pl.foldv(lambda state: state & pl.col("l"), s=True)
    )

    assert result.to_series().dtype == pl.Boolean
    assert result.to_series().to_list() == [True, True, False]


def test_foldv_coerces_int32_rows_into_a_float64_state() -> None:
    # Mixed input and state types follow Polars' expression coercion rules.
    result = pl.DataFrame({"l": pl.Series([1, 2, 3], dtype=pl.Int32)}).select(
        pl.foldv(lambda state: state + pl.col("l"), s=0.5)
    )

    assert result.to_series().dtype == pl.Float64
    assert result.to_series().to_list() == [1.5, 3.5, 6.5]


def test_foldv_widens_int32_rows_to_int64() -> None:
    # A widening integer cast remains non-null and executes in the generated program.
    result = pl.DataFrame({"l": pl.Series([1, 2, 3], dtype=pl.Int32)}).select(
        pl.foldv(
            lambda state: state + pl.col("l").cast(pl.Int64),
            s=pl.lit(0, dtype=pl.Int64),
        )
    )

    assert result.to_series().dtype == pl.Int64
    assert result.to_series().to_list() == [1, 3, 6]

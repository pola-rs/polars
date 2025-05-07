from __future__ import annotations

import io
from datetime import datetime
from typing import IO
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.parametrize(
    ("literal_values", "expected", "cast_options"),
    [
        (
            (pl.lit(1, dtype=pl.Int64), pl.lit(2, dtype=pl.Int32)),
            pl.Series([1, 2], dtype=pl.Int64),
            pl.ScanCastOptions(integer_cast="upcast"),
        ),
        (
            (pl.lit(1.0, dtype=pl.Float64), pl.lit(2.0, dtype=pl.Float32)),
            pl.Series([1, 2], dtype=pl.Float64),
            pl.ScanCastOptions(float_cast="upcast"),
        ),
        (
            (pl.lit(1.0, dtype=pl.Float32), pl.lit(2.0, dtype=pl.Float64)),
            pl.Series([1, 2], dtype=pl.Float32),
            pl.ScanCastOptions(float_cast=["upcast", "downcast"]),
        ),
        (
            (
                pl.lit(datetime(2025, 1, 1), dtype=pl.Datetime(time_unit="ms")),
                pl.lit(datetime(2025, 1, 2), dtype=pl.Datetime(time_unit="ns")),
            ),
            pl.Series(
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                dtype=pl.Datetime(time_unit="ms"),
            ),
            pl.ScanCastOptions(datetime_cast="nanosecond-downcast"),
        ),
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")),
                    dtype=pl.Datetime(time_unit="ms", time_zone="Europe/Amsterdam"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, tzinfo=ZoneInfo("Australia/Sydney")),
                    dtype=pl.Datetime(time_unit="ns", time_zone="Australia/Sydney"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")),
                    datetime(2025, 1, 1, 14, tzinfo=ZoneInfo("Europe/Amsterdam")),
                ],
                dtype=pl.Datetime(time_unit="ms", time_zone="Europe/Amsterdam"),
            ),
            pl.ScanCastOptions(
                datetime_cast=["nanosecond-downcast", "convert-timezone"]
            ),
        ),
        (
            (  # We also test nested primitive upcast policy with this one
                pl.lit(
                    {"a": [[1]], "b": 1},
                    dtype=pl.Struct(
                        {"a": pl.List(pl.Array(pl.Int32, 1)), "b": pl.Int32}
                    ),
                ),
                pl.lit(
                    {"a": [[2]]},
                    dtype=pl.Struct({"a": pl.List(pl.Array(pl.Int8, 1))}),
                ),
            ),
            pl.Series(
                [{"a": [[1]], "b": 1}, {"a": [[2]], "b": None}],
                dtype=pl.Struct({"a": pl.List(pl.Array(pl.Int32, 1)), "b": pl.Int32}),
            ),
            pl.ScanCastOptions(
                integer_cast="upcast",
                missing_struct_fields="insert",
            ),
        ),
        (
            (  # Test same set of struct fields but in different order
                pl.lit(
                    {"a": [[1]], "b": 1},
                    dtype=pl.Struct(
                        {"a": pl.List(pl.Array(pl.Int32, 1)), "b": pl.Int32}
                    ),
                ),
                pl.lit(
                    {"b": None, "a": [[2]]},
                    dtype=pl.Struct(
                        {"b": pl.Int32, "a": pl.List(pl.Array(pl.Int32, 1))}
                    ),
                ),
            ),
            pl.Series(
                [{"a": [[1]], "b": 1}, {"a": [[2]], "b": None}],
                dtype=pl.Struct({"a": pl.List(pl.Array(pl.Int32, 1)), "b": pl.Int32}),
            ),
            None,
        ),
    ],
)
def test_scan_cast_options(
    literal_values: tuple[pl.Expr, pl.Expr],
    expected: pl.Series,
    cast_options: pl.ScanCastOptions | None,
) -> None:
    expected = expected.alias("literal")
    lv1, lv2 = literal_values

    df1 = pl.select(lv1)
    df2 = pl.select(lv2)

    # `cast()` from the Python API should give the same results.
    assert_frame_equal(
        pl.concat(
            [
                df1.cast(expected.dtype),
                df2.cast(expected.dtype),
            ]
        ),
        expected.to_frame(),
    )

    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]

    df1.write_parquet(files[0])
    df2.write_parquet(files[1])

    for f in files:
        f.seek(0)

    # Note: Schema is taken from the first file

    if cast_options is not None:
        q = pl.scan_parquet(files)

        with pytest.raises(pl.exceptions.SchemaError, match=r"hint: pass "):
            q.collect()

    assert_frame_equal(
        pl.scan_parquet(files, cast_options=cast_options).collect(),
        expected.to_frame(),
    )


def test_scan_cast_options_forbid_int_downcast() -> None:
    # Test to ensure that passing `integer_cast='upcast'` does not accidentally
    # permit casting to smaller integer types.
    lv1, lv2 = pl.lit(1, dtype=pl.Int8), pl.lit(2, dtype=pl.Int32)

    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]

    df1 = pl.select(lv1)
    df2 = pl.select(lv2)

    df1.write_parquet(files[0])
    df2.write_parquet(files[1])

    for f in files:
        f.seek(0)

    q = pl.scan_parquet(files)

    with pytest.raises(pl.exceptions.SchemaError):
        q.collect()

    for f in files:
        f.seek(0)

    q = pl.scan_parquet(
        files,
        cast_options=pl.ScanCastOptions(integer_cast="upcast"),
    )

    with pytest.raises(pl.exceptions.SchemaError):
        q.collect()


def test_scan_cast_options_extra_struct_fields() -> None:
    cast_options = pl.ScanCastOptions(extra_struct_fields="ignore")

    expected = pl.Series([{"a": 1}, {"a": 2}], dtype=pl.Struct({"a": pl.Int32}))
    expected = expected.alias("literal")

    lv1, lv2 = (
        pl.lit({"a": 1}, dtype=pl.Struct({"a": pl.Int32})),
        pl.lit(
            {"a": 2, "extra_field": 1},
            dtype=pl.Struct({"a": pl.Int32, "extra_field": pl.Int32}),
        ),
    )

    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]

    df1 = pl.select(lv1)
    df2 = pl.select(lv2)

    df1.write_parquet(files[0])
    df2.write_parquet(files[1])

    for f in files:
        f.seek(0)

    q = pl.scan_parquet(files)

    with pytest.raises(pl.exceptions.SchemaError, match=r"hint: pass "):
        q.collect()

    assert_frame_equal(
        pl.scan_parquet(files, cast_options=cast_options).collect(),
        expected.to_frame(),
    )

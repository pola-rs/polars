from __future__ import annotations

import io
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.datatypes.group import FLOAT_DTYPES
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import EngineType


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
        # datetime: ns -> ms (nanosecond-downcast). The incoming value carries
        # sub-millisecond precision so a broken (non-scaling) cast is observable.
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=111000),
                    dtype=pl.Datetime(time_unit="ms"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=123456),
                    dtype=pl.Datetime(time_unit="ns"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=111000),
                    datetime(2025, 1, 2, microsecond=123000),
                ],
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
                datetime_cast=("nanosecond-downcast", "convert-timezone")
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
        # Test logical (datetime) type under list
        (
            (
                pl.lit(
                    [
                        {
                            "field": datetime(
                                2025, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")
                            )
                        }
                    ],
                    dtype=pl.List(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ms", time_zone="Europe/Amsterdam"
                                )
                            }
                        )
                    ),
                ),
                pl.lit(
                    [
                        {
                            "field": datetime(
                                2025, 1, 2, tzinfo=ZoneInfo("Australia/Sydney")
                            )
                        }
                    ],
                    dtype=pl.List(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ns", time_zone="Australia/Sydney"
                                )
                            }
                        )
                    ),
                ),
            ),
            pl.Series(
                [
                    [
                        {
                            "field": datetime(
                                2025, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")
                            )
                        }
                    ],
                    [
                        {
                            "field": datetime(
                                2025, 1, 1, 14, tzinfo=ZoneInfo("Europe/Amsterdam")
                            )
                        }
                    ],
                ],
                dtype=pl.List(
                    pl.Struct(
                        {
                            "field": pl.Datetime(
                                time_unit="ms", time_zone="Europe/Amsterdam"
                            )
                        }
                    )
                ),
            ),
            pl.ScanCastOptions(
                datetime_cast=("nanosecond-downcast", "convert-timezone")
            ),
        ),
        (
            (
                pl.lit(
                    [
                        {
                            "field": datetime(
                                2025, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")
                            )
                        }
                    ],
                    dtype=pl.Array(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ms", time_zone="Europe/Amsterdam"
                                )
                            }
                        ),
                        shape=1,
                    ),
                ),
                pl.lit(
                    [
                        {
                            "field": datetime(
                                2025, 1, 2, tzinfo=ZoneInfo("Australia/Sydney")
                            )
                        }
                    ],
                    dtype=pl.Array(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ns", time_zone="Australia/Sydney"
                                )
                            }
                        ),
                        shape=1,
                    ),
                ),
            ),
            pl.Series(
                [
                    [
                        {
                            "field": datetime(
                                2025, 1, 1, tzinfo=ZoneInfo("Europe/Amsterdam")
                            )
                        }
                    ],
                    [
                        {
                            "field": datetime(
                                2025, 1, 1, 14, tzinfo=ZoneInfo("Europe/Amsterdam")
                            )
                        }
                    ],
                ],
                dtype=pl.Array(
                    pl.Struct(
                        {
                            "field": pl.Datetime(
                                time_unit="ms", time_zone="Europe/Amsterdam"
                            )
                        }
                    ),
                    shape=1,
                ),
            ),
            pl.ScanCastOptions(
                datetime_cast=("nanosecond-downcast", "convert-timezone")
            ),
        ),
        # Test outer validity
        (
            (
                pl.lit(
                    None,
                    dtype=pl.List(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ms", time_zone="Europe/Amsterdam"
                                )
                            }
                        )
                    ),
                ),
                pl.lit(
                    [None],
                    dtype=pl.List(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ns", time_zone="Australia/Sydney"
                                )
                            }
                        )
                    ),
                ),
            ),
            pl.Series(
                [None, [None]],
                dtype=pl.List(
                    pl.Struct(
                        {
                            "field": pl.Datetime(
                                time_unit="ms", time_zone="Europe/Amsterdam"
                            )
                        }
                    )
                ),
            ),
            pl.ScanCastOptions(
                datetime_cast=("nanosecond-downcast", "convert-timezone")
            ),
        ),
        (
            (
                pl.lit(
                    None,
                    dtype=pl.Array(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ms", time_zone="Europe/Amsterdam"
                                )
                            }
                        ),
                        shape=1,
                    ),
                ),
                pl.lit(
                    [None],
                    dtype=pl.Array(
                        pl.Struct(
                            {
                                "field": pl.Datetime(
                                    time_unit="ns", time_zone="Australia/Sydney"
                                )
                            }
                        ),
                        shape=1,
                    ),
                ),
            ),
            pl.Series(
                [None, [None]],
                dtype=pl.Array(
                    pl.Struct(
                        {
                            "field": pl.Datetime(
                                time_unit="ms", time_zone="Europe/Amsterdam"
                            )
                        }
                    ),
                    shape=1,
                ),
            ),
            pl.ScanCastOptions(
                datetime_cast=("nanosecond-downcast", "convert-timezone")
            ),
        ),
        # datetime: us -> ms (microsecond-downcast)
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=111000),
                    dtype=pl.Datetime(time_unit="ms"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=123456),
                    dtype=pl.Datetime(time_unit="us"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=111000),
                    datetime(2025, 1, 2, microsecond=123000),
                ],
                dtype=pl.Datetime(time_unit="ms"),
            ),
            pl.ScanCastOptions(datetime_cast="microsecond-downcast"),
        ),
        # datetime: ms -> us (millisecond-upcast). The ms value must scale up by
        # 1000; a non-scaling cast would land 1000x off.
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=123456),
                    dtype=pl.Datetime(time_unit="us"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=789000),
                    dtype=pl.Datetime(time_unit="ms"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=123456),
                    datetime(2025, 1, 2, microsecond=789000),
                ],
                dtype=pl.Datetime(time_unit="us"),
            ),
            pl.ScanCastOptions(datetime_cast="millisecond-upcast"),
        ),
        # datetime: ms -> ns (millisecond-upcast).
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=123456),
                    dtype=pl.Datetime(time_unit="ns"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=789000),
                    dtype=pl.Datetime(time_unit="ms"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=123456),
                    datetime(2025, 1, 2, microsecond=789000),
                ],
                dtype=pl.Datetime(time_unit="ns"),
            ),
            pl.ScanCastOptions(datetime_cast="millisecond-upcast"),
        ),
        # datetime: us -> ns (microsecond-upcast).
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=123456),
                    dtype=pl.Datetime(time_unit="ns"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=654321),
                    dtype=pl.Datetime(time_unit="us"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=123456),
                    datetime(2025, 1, 2, microsecond=654321),
                ],
                dtype=pl.Datetime(time_unit="ns"),
            ),
            pl.ScanCastOptions(datetime_cast="microsecond-upcast"),
        ),
        # datetime: ms -> ns via generic 'upcast' aggregate.
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=123456),
                    dtype=pl.Datetime(time_unit="ns"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=789000),
                    dtype=pl.Datetime(time_unit="ms"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=123456),
                    datetime(2025, 1, 2, microsecond=789000),
                ],
                dtype=pl.Datetime(time_unit="ns"),
            ),
            pl.ScanCastOptions(datetime_cast="upcast"),
        ),
        # datetime: us -> ms via generic 'downcast' aggregate.
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=111000),
                    dtype=pl.Datetime(time_unit="ms"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=123456),
                    dtype=pl.Datetime(time_unit="us"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=111000),
                    datetime(2025, 1, 2, microsecond=123000),
                ],
                dtype=pl.Datetime(time_unit="ms"),
            ),
            pl.ScanCastOptions(datetime_cast="downcast"),
        ),
        # datetime: us -> ms with both directions enabled via list.
        (
            (
                pl.lit(
                    datetime(2025, 1, 1, microsecond=111000),
                    dtype=pl.Datetime(time_unit="ms"),
                ),
                pl.lit(
                    datetime(2025, 1, 2, microsecond=123456),
                    dtype=pl.Datetime(time_unit="us"),
                ),
            ),
            pl.Series(
                [
                    datetime(2025, 1, 1, microsecond=111000),
                    datetime(2025, 1, 2, microsecond=123000),
                ],
                dtype=pl.Datetime(time_unit="ms"),
            ),
            pl.ScanCastOptions(
                datetime_cast=("microsecond-downcast", "millisecond-upcast")
            ),
        ),
    ],
)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_scan_cast_options(
    literal_values: tuple[pl.Expr, pl.Expr],
    expected: pl.Series,
    cast_options: pl.ScanCastOptions | None,
    engine: EngineType,
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

        with pytest.raises(pl.exceptions.SchemaError, match=r"hint: .*pass"):
            q.collect(engine=engine)

    assert_frame_equal(
        pl.scan_parquet(files, cast_options=cast_options).collect(engine=engine),
        expected.to_frame(),
    )


@pytest.mark.parametrize(
    ("target_unit", "incoming_unit", "cast_str", "incoming_phys", "expected_phys"),
    [
        ("ms", "us", "microsecond-downcast", 1_734_567_891_123_456, 1_734_567_891_123),
        ("us", "ms", "millisecond-upcast", 1_734_567_891_789, 1_734_567_891_789_000),
        (
            "ns",
            "ms",
            "millisecond-upcast",
            1_734_567_891_789,
            1_734_567_891_789_000_000,
        ),
        (
            "ns",
            "us",
            "microsecond-upcast",
            1_734_567_891_123_456,
            1_734_567_891_123_456_000,
        ),
        (
            "ms",
            "ns",
            "nanosecond-downcast",
            1_734_567_891_123_456_789,
            1_734_567_891_123,
        ),
    ],
)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_scan_cast_options_datetime_physical_scaling(
    target_unit: str,
    incoming_unit: str,
    cast_str: str,
    incoming_phys: int,
    expected_phys: int,
    engine: EngineType,
) -> None:
    target = pl.Series([0], dtype=pl.Int64).cast(pl.Datetime(time_unit=target_unit))  # type: ignore[arg-type]
    incoming = pl.Series([incoming_phys], dtype=pl.Int64).cast(
        pl.Datetime(time_unit=incoming_unit)  # type: ignore[arg-type]
    )
    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]
    target.to_frame().write_parquet(files[0])
    incoming.to_frame().write_parquet(files[1])
    for f in files:
        f.seek(0)

    out = pl.scan_parquet(
        files,
        cast_options=pl.ScanCastOptions(datetime_cast=cast_str),  # type: ignore[arg-type]
    ).collect(engine=engine)

    result = out.to_series()
    assert result.dtype == pl.Datetime(time_unit=target_unit)  # type: ignore[arg-type]
    assert result.to_physical().to_list()[1] == expected_phys


@pytest.mark.parametrize(
    ("target_unit", "incoming_unit", "cast_str", "incoming_phys"),
    [
        # ms -> ns scales by 1e6; value is a valid datetime but overflows
        ("ns", "ms", "millisecond-upcast", 9_300_000_000_000),
        # us -> ns scales by 1e3; overflows once scaled (> i64::MAX / 1e3).
        ("ns", "us", "microsecond-upcast", 9_000_000_000_000_000_00),
    ],
)
# note: `nested` also exercises the nested cast path (datetime inside a list)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_scan_cast_options_datetime_upcast_overflow_raises(
    target_unit: str,
    incoming_unit: str,
    cast_str: str,
    incoming_phys: int,
    nested: bool,
    engine: EngineType,
) -> None:
    # upcasting to a finer unit multiplies the physical values and can overflow
    target = pl.Series([0], dtype=pl.Int64).cast(pl.Datetime(time_unit=target_unit))  # type: ignore[arg-type]
    incoming = pl.Series([incoming_phys], dtype=pl.Int64).cast(
        pl.Datetime(time_unit=incoming_unit)  # type: ignore[arg-type]
    )
    if nested:
        target = target.implode()
        incoming = incoming.implode()

    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]
    target.to_frame().write_parquet(files[0])
    incoming.to_frame().write_parquet(files[1])
    for f in files:
        f.seek(0)

    scan_opts = pl.ScanCastOptions(datetime_cast=cast_str)  # type: ignore[arg-type]
    lf = pl.scan_parquet(files, cast_options=scan_opts)
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match=r"conversion.*failed"
    ):
        lf.collect(engine=engine)


@pytest.mark.parametrize(
    ("target_unit", "incoming_unit", "cast_options"),
    [
        ("ns", "us", pl.ScanCastOptions(datetime_cast="microsecond-downcast")),
        ("ms", "us", pl.ScanCastOptions(datetime_cast="microsecond-upcast")),
        ("ms", "us", pl.ScanCastOptions(datetime_cast="millisecond-upcast")),
        ("ms", "us", pl.ScanCastOptions(datetime_cast="upcast")),
        ("ns", "ms", pl.ScanCastOptions(datetime_cast="downcast")),
    ],
)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_scan_cast_options_datetime_wrong_direction(
    target_unit: str,
    incoming_unit: str,
    cast_options: pl.ScanCastOptions,
    engine: EngineType,
) -> None:
    # an option enabling one direction should not silently permit a cast in
    # the other direction (eg: enabling upcasts should still forbid downcasts)
    lv1 = pl.lit(datetime(2025, 1, 1), dtype=pl.Datetime(time_unit=target_unit))  # type: ignore[arg-type]
    lv2 = pl.lit(datetime(2025, 1, 2), dtype=pl.Datetime(time_unit=incoming_unit))  # type: ignore[arg-type]

    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]

    pl.select(lv1).write_parquet(files[0])
    pl.select(lv2).write_parquet(files[1])
    for f in files:
        f.seek(0)

    q = pl.scan_parquet(files, cast_options=cast_options)
    with pytest.raises(pl.exceptions.SchemaError, match=r"hint: .*pass"):
        q.collect(engine=engine)


def test_scan_cast_options_datetime_unknown_option() -> None:
    f = io.BytesIO()
    pl.select(
        pl.lit(datetime(2025, 1, 1), dtype=pl.Datetime(time_unit="ms"))
    ).write_parquet(f)
    f.seek(0)

    with pytest.raises(TypeError) as excinfo:
        pl.scan_parquet(
            f,
            cast_options=pl.ScanCastOptions(datetime_cast="bogus"),  # type: ignore[arg-type]
        )

    assert "unknown option for datetime_cast: bogus" in str(excinfo.value.__cause__)


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

    with pytest.raises(pl.exceptions.SchemaError, match=r"hint: specify .*or pass"):
        q.collect()

    assert_frame_equal(
        pl.scan_parquet(files, cast_options=cast_options).collect(),
        expected.to_frame(),
    )


def test_cast_options_ignore_extra_columns() -> None:
    files: list[IO[bytes]] = [io.BytesIO(), io.BytesIO()]

    pl.DataFrame({"a": 1}).write_parquet(files[0])
    pl.DataFrame({"a": 2, "b": 1}).write_parquet(files[1])

    with pytest.raises(
        pl.exceptions.SchemaError,
        match=r"extra column in file outside of expected schema: b, hint: specify.* or pass",
    ):
        pl.scan_parquet(files, schema={"a": pl.Int64}).collect()

    assert_frame_equal(
        pl.scan_parquet(
            files,
            schema={"a": pl.Int64},
            extra_columns="ignore",
        ).collect(),
        pl.DataFrame({"a": [1, 2]}),
    )


@pytest.mark.parametrize(
    ("scan_func", "write_func"),
    [
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        # TODO: Fix for all other formats
        # (pl.scan_ipc, pl.DataFrame.write_ipc),
        # (pl.scan_csv, pl.DataFrame.write_csv),
        # (pl.scan_ndjson, pl.DataFrame.write_ndjson),
    ],
)
def test_scan_cast_options_extra_columns(
    scan_func: Callable[[Any], pl.LazyFrame],
    write_func: Callable[[pl.DataFrame, io.BytesIO], None],
) -> None:
    dfs = [pl.DataFrame({"a": 1, "b": 1}), pl.DataFrame({"a": 2, "b": 2, "c": 2})]
    files = [io.BytesIO(), io.BytesIO()]

    write_func(dfs[0], files[0])
    write_func(dfs[1], files[1])

    with pytest.raises(
        pl.exceptions.SchemaError,
        match=r"extra column in file outside of expected schema: c, hint: ",
    ):
        scan_func(files).collect()

    assert_frame_equal(
        scan_func(files, extra_columns="ignore").collect(),  # type: ignore[call-arg]
        pl.DataFrame({"a": [1, 2], "b": [1, 2]}),
    )


@pytest.mark.parametrize("float_dtype", sorted(FLOAT_DTYPES, key=repr))
def test_scan_cast_options_integer_to_float(float_dtype: pl.DataType) -> None:
    df = pl.DataFrame({"a": [1]}, schema={"a": pl.Int64})
    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)

    assert_frame_equal(
        pl.scan_parquet(f).collect(),
        pl.DataFrame({"a": [1]}, schema={"a": pl.Int64}),
    )

    q = pl.scan_parquet(f, schema={"a": float_dtype})

    with pytest.raises(SchemaError):
        q.collect()

    f.seek(0)

    assert_frame_equal(
        pl.scan_parquet(
            f,
            schema={"a": float_dtype},
            cast_options=pl.ScanCastOptions(integer_cast="allow-float"),
        ).collect(),
        pl.DataFrame({"a": [1.0]}, schema={"a": float_dtype}),
    )

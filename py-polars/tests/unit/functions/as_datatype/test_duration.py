from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import TimeUnit


def test_empty_duration() -> None:
    s = pl.DataFrame([], {"days": pl.Int32}).select(pl.duration(days="days"))
    assert s.dtypes == [pl.Duration("us")]
    assert s.shape == (0, 1)


@pytest.mark.parametrize(
    ("time_unit", "expected"),
    [
        ("ms", timedelta(days=1, minutes=2, seconds=3, milliseconds=4)),
        ("us", timedelta(days=1, minutes=2, seconds=3, milliseconds=4, microseconds=5)),
        ("ns", timedelta(days=1, minutes=2, seconds=3, milliseconds=4, microseconds=5)),
    ],
)
def test_duration_time_units(time_unit: TimeUnit, expected: timedelta) -> None:
    result = pl.LazyFrame().select(
        duration=pl.duration(
            days=1,
            minutes=2,
            seconds=3,
            milliseconds=4,
            microseconds=5,
            nanoseconds=6,
            time_unit=time_unit,
        )
    )
    assert result.collect_schema()["duration"] == pl.Duration(time_unit)
    assert result.collect()["duration"].item() == expected
    if time_unit == "ns":
        assert (
            result.collect()["duration"].dt.total_nanoseconds().item() == 86523004005006
        )


def test_datetime_duration_offset() -> None:
    df = pl.DataFrame(
        {
            "datetime": [
                datetime(1999, 1, 1, 7),
                datetime(2022, 1, 2, 14),
                datetime(3000, 12, 31, 21),
            ],
            "add": [1, 2, -1],
        }
    )
    out = df.select(
        (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
        (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
        (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
        (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
        (pl.col("datetime") + pl.duration(microseconds=pl.col("add") * 1000)).alias(
            "add_usecs"
        ),
    )
    expected = pl.DataFrame(
        {
            "add_weeks": [
                datetime(1999, 1, 8, 7),
                datetime(2022, 1, 16, 14),
                datetime(3000, 12, 24, 21),
            ],
            "add_days": [
                datetime(1999, 1, 2, 7),
                datetime(2022, 1, 4, 14),
                datetime(3000, 12, 30, 21),
            ],
            "add_hours": [
                datetime(1999, 1, 1, hour=8),
                datetime(2022, 1, 2, hour=16),
                datetime(3000, 12, 31, hour=20),
            ],
            "add_seconds": [
                datetime(1999, 1, 1, 7, second=1),
                datetime(2022, 1, 2, 14, second=2),
                datetime(3000, 12, 31, 20, 59, 59),
            ],
            "add_usecs": [
                datetime(1999, 1, 1, 7, microsecond=1000),
                datetime(2022, 1, 2, 14, microsecond=2000),
                datetime(3000, 12, 31, 20, 59, 59, 999000),
            ],
        }
    )
    assert_frame_equal(out, expected)


def test_date_duration_offset() -> None:
    df = pl.DataFrame(
        {
            "date": [date(10, 1, 1), date(2000, 7, 5), date(9990, 12, 31)],
            "offset": [365, 7, -31],
        }
    )
    out = df.select(
        (pl.col("date") + pl.duration(days="offset")).alias("add_days"),
        (pl.col("date") - pl.duration(days="offset")).alias("sub_days"),
        (pl.col("date") + pl.duration(weeks="offset")).alias("add_weeks"),
        (pl.col("date") - pl.duration(weeks="offset")).alias("sub_weeks"),
    )
    assert out.to_dict(as_series=False) == {
        "add_days": [date(11, 1, 1), date(2000, 7, 12), date(9990, 11, 30)],
        "sub_days": [date(9, 1, 1), date(2000, 6, 28), date(9991, 1, 31)],
        "add_weeks": [date(16, 12, 30), date(2000, 8, 23), date(9990, 5, 28)],
        "sub_weeks": [date(3, 1, 3), date(2000, 5, 17), date(9991, 8, 5)],
    }


def test_add_duration_3786() -> None:
    df = pl.DataFrame(
        {
            "datetime": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
            "add": [1, 2],
        }
    )
    assert df.slice(0, 1).with_columns(
        (pl.col("datetime") + pl.duration(weeks="add")).alias("add_weeks"),
        (pl.col("datetime") + pl.duration(days="add")).alias("add_days"),
        (pl.col("datetime") + pl.duration(seconds="add")).alias("add_seconds"),
        (pl.col("datetime") + pl.duration(milliseconds="add")).alias(
            "add_milliseconds"
        ),
        (pl.col("datetime") + pl.duration(hours="add")).alias("add_hours"),
    ).to_dict(as_series=False) == {
        "datetime": [datetime(2022, 1, 1, 0, 0)],
        "add": [1],
        "add_weeks": [datetime(2022, 1, 8, 0, 0)],
        "add_days": [datetime(2022, 1, 2, 0, 0)],
        "add_seconds": [datetime(2022, 1, 1, 0, 0, 1)],
        "add_milliseconds": [datetime(2022, 1, 1, 0, 0, 0, 1000)],
        "add_hours": [datetime(2022, 1, 1, 1, 0)],
    }


@pytest.mark.parametrize(
    ("time_unit", "ms", "us", "ns"),
    [
        ("ms", 11, 0, 0),
        ("us", 0, 11_007, 0),
        ("ns", 0, 0, 11_007_003),
    ],
)
def test_duration_subseconds_us(time_unit: TimeUnit, ms: int, us: int, ns: int) -> None:
    result = pl.duration(
        milliseconds=6, microseconds=4_005, nanoseconds=1_002_003, time_unit=time_unit
    )
    expected = pl.duration(
        milliseconds=ms, microseconds=us, nanoseconds=ns, time_unit=time_unit
    )
    assert_frame_equal(pl.select(result), pl.select(expected))


def test_duration_time_unit_ns() -> None:
    result = pl.duration(milliseconds=4, microseconds=3_000, nanoseconds=10)
    expected = pl.duration(
        milliseconds=4, microseconds=3_000, nanoseconds=10, time_unit="ns"
    )
    assert_frame_equal(pl.select(result), pl.select(expected))


def test_duration_time_unit_us() -> None:
    result = pl.duration(milliseconds=4, microseconds=3_000)
    expected = pl.duration(milliseconds=4, microseconds=3_000, time_unit="us")
    assert_frame_equal(pl.select(result), pl.select(expected))


def test_duration_time_unit_ms() -> None:
    result = pl.duration(milliseconds=4)
    expected = pl.duration(milliseconds=4, time_unit="us")
    assert_frame_equal(pl.select(result), pl.select(expected))


def test_duration_wildcard_expansion() -> None:
    # Test that wildcard expansions occurs correctly in pl.duration
    # https://github.com/pola-rs/polars/issues/19007
    df = df = pl.DataFrame({"a": [1], "b": [2]})
    assert df.select(pl.duration(hours=pl.all()).name.keep()).to_dict(
        as_series=False
    ) == {"a": [timedelta(seconds=3600)], "b": [timedelta(seconds=7200)]}


def test_duration_from_i128_23050() -> None:
    assert_frame_equal(
        pl.DataFrame(
            [
                pl.Series("a", [1478262, 84879], pl.Int128),
                pl.Series("b", [532440, 309209], pl.Int128),
            ]
        ).select(
            pl.duration(microseconds=pl.col.a + pl.col.b).alias("c"),
        ),
        pl.select(
            c=pl.duration(
                seconds=pl.lit(pl.Series([2, 0])),
                microseconds=pl.lit(pl.Series([10702, 394088])),
            )
        ),
    )


def test_duration_overflow() -> None:
    # (~300 years cannot be represented with "ns" precision, so the total is null)
    df = pl.DataFrame({"days": [300 * 365, 1], "hours": [None, 1]})
    result = df.select(
        pl.duration(
            days="days", hours="hours", nanoseconds=1, seconds=0.5, time_unit="us"
        )
    )
    assert result.to_series().to_list() == [None, timedelta(1, hours=1, seconds=0.5)]

    # (the floats of every row are added, also when the integers are scalars)
    result = pl.DataFrame({"f": [0.5, 1.5]}).select(
        pl.duration(days=200_000, nanoseconds=1, seconds="f", time_unit="us")
    )
    expected = [17_280_000_000_500_000, 17_280_000_001_500_000]
    assert result.to_series().to_physical().to_list() == expected

    # A wide scalar integer subtotal is broadcast before the final float addition.
    result = pl.DataFrame({"hours": [-4_800_000.0, -4_799_999.0]}).select(
        pl.duration(days=200_000, hours="hours", time_unit="ns")
    )
    assert result.to_series().to_physical().to_list() == [0, 3_600_000_000_000]

    # (masked values do not matter)
    masked = pl.when(pl.col("days") < 1_000).then("days")
    for days, hours in ((pl.col("days"), pl.col("hours")), (masked, pl.lit(1))):
        result = df.select(pl.duration(days=days, hours=hours, time_unit="ns"))
        assert result.to_series().to_list() == [None, timedelta(days=1, hours=1)]

    # (unsigned) values outside the i64 range raise
    with pytest.raises(InvalidOperationError, match=r"conversion from `u64` to `i64`"):
        pl.select(pl.duration(seconds=pl.lit(pl.Series([2**63], dtype=pl.UInt64))))


def test_duration_chunks_and_scalars() -> None:
    df = pl.concat(
        [
            pl.DataFrame({"d": [1, None], "f": [0.5, 1.0]}),
            pl.DataFrame({"d": [3], "f": [None]}),
        ],
        rechunk=False,
    ).with_columns(h=pl.lit(2), s=pl.lit(1.5), n=pl.lit(None, dtype=pl.Int64))
    result = df.select(
        chunks=pl.duration(days="d", hours=1),
        first_scalar=pl.duration(weeks=1, days="d"),
        float_nulls=pl.duration(days=1, seconds="f"),
        null=pl.duration(days="d", hours="n"),  # (a null scalar nulls every result)
        null_float=pl.duration(days="d", seconds=pl.lit(None, dtype=pl.Float64)),
        scalar=pl.duration(days=pl.col("d").first(), hours="h", seconds="s"),
        null_scalar=pl.duration(hours="h", seconds="n"),
    )
    assert result.to_dict(as_series=False) == {
        "chunks": [timedelta(days=1, hours=1), None, timedelta(days=3, hours=1)],
        "first_scalar": [timedelta(days=8), None, timedelta(days=10)],
        "float_nulls": [
            timedelta(days=1, seconds=0.5),
            timedelta(days=1, seconds=1),
            None,
        ],
        "null": [None] * 3,
        "null_float": [None] * 3,
        "scalar": [timedelta(days=1, hours=2, seconds=1.5)] * 3,
        "null_scalar": [None] * 3,
    }


def test_duration_misaligned_chunks_with_overflow() -> None:
    days = pl.concat(
        [pl.Series("days", [200_000, 200_000]), pl.Series("days", [None, 3, 4])],
        rechunk=False,
    )
    hours = pl.concat(
        [pl.Series("hours", [-4_800_000, -4_800_000, 2]), pl.Series("hours", [4, 5])],
        rechunk=False,
    )
    df = pl.DataFrame([days, hours, pl.Series("seconds", [0.25, 0.5, 1.5, 2.5, 3.5])])

    result = (
        df.lazy()
        .select_seq(
            pl.duration(days="days", hours="hours", seconds="seconds", time_unit="ns")
        )
        .collect(engine="in-memory")
        .to_series()
    )
    assert result.n_chunks() > 1
    assert result.to_list() == [
        timedelta(seconds=0.25),
        timedelta(seconds=0.5),
        None,
        timedelta(days=3, hours=4, seconds=2.5),
        timedelta(days=4, hours=5, seconds=3.5),
    ]


NANOSECONDS_PER = {
    "weeks": 7 * 86_400 * 10**9,
    "days": 86_400 * 10**9,
    "hours": 3_600 * 10**9,
    "minutes": 60 * 10**9,
    "seconds": 10**9,
    "milliseconds": 10**6,
    "microseconds": 10**3,
    "nanoseconds": 1,
}


def _expected_duration(kwargs: dict[str, float], unit_ns: int) -> int | None:
    # integers are summed exactly and truncated (towards zero), then the floats are
    # added (in component order) with float precision; null if out of range
    ints = sum(v * NANOSECONDS_PER[k] for k, v in kwargs.items() if isinstance(v, int))
    integer = int(Fraction(ints, unit_ns))
    floats = 0.0
    for k, ns in NANOSECONDS_PER.items():
        if isinstance(v := kwargs.get(k), float):
            floats += v * (ns // unit_ns) if ns >= unit_ns else v / (unit_ns // ns)
    total = integer + floats if floats else integer
    return int(total) if math.isfinite(total) and -(2**63) <= total < 2**63 else None


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"nanoseconds": 999_999, "microseconds": 1},
        {"milliseconds": 2, "microseconds": -1, "nanoseconds": -1_999_999},
        {"seconds": 1, "microseconds": -1},
        {"weeks": 1, "days": -3, "hours": 25, "seconds": -1, "nanoseconds": 1_500},
        {"nanoseconds": 1.5e6, "microseconds": 500.0},
        {"seconds": 1.0, "microseconds": -1.0},
        {"days": 2, "hours": 1.5, "milliseconds": -0.75, "nanoseconds": 7},
        {"milliseconds": 1, "microseconds": -1, "seconds": 0.5},
        # (finer floats are divided exactly)
        {"nanoseconds": 4413651200609999.5},
        {"nanoseconds": 71, "microseconds": 0.9289999999999999},
        # (the inputs can exceed the "ns" range if the total does not)
        {"weeks": 50_000, "days": -349_999, "nanoseconds": -1},
        {"weeks": -50_000, "days": 349_999, "nanoseconds": 1},
        {"weeks": 50_000, "days": -349_999, "seconds": 0.5},
        {"days": 300 * 365, "nanoseconds": 1, "seconds": 0.5},
        {"days": 300 * 365, "nanoseconds": 1},
        # (integers are truncated exactly before floats are added)
        {"days": 200, "nanoseconds": 1, "seconds": 0.0},
        {"seconds": 10**10, "nanoseconds": 1_000, "microseconds": 0.0},
        {"days": 200_000, "hours": -4_800_000.0},
        {"nanoseconds": 1_700_000_000_000_000_999, "seconds": 0.5},
        {"nanoseconds": -1_700_000_000_000_000_999, "seconds": -0.5},
        {"nanoseconds": 999, "microseconds": 0.9995},
        {"nanoseconds": -(9 * 10**18), "seconds": 1.8e10},
        {"seconds": 7_931_731_533, "milliseconds": 0.25, "nanoseconds": 0},
        # (totals that are non-finite or out of range are null)
        {"days": 300 * 365},
        {"weeks": 15_000, "days": 2_000},
        {"days": 300 * 365, "seconds": 1.0},
        {"days": 1, "seconds": float("nan")},
        {"milliseconds": float("inf")},
        {"milliseconds": 2.0**63},
        {"milliseconds": 2.0**63 - 1024},  # (the largest float below 2**63)
    ],
)
def test_duration_combines_components(
    kwargs: dict[str, float], time_unit: TimeUnit
) -> None:
    # The same result for columns and for literals, including zero float columns.
    unit_ns = {"ns": 1, "us": 1_000, "ms": 1_000_000}[time_unit]
    other = {k: type(v)(1) for k, v in kwargs.items()}
    df = pl.DataFrame({k: [v, other[k]] for k, v in kwargs.items()})
    expr = pl.duration(
        **{k: pl.col(k) for k in kwargs},
        time_unit=time_unit,  # type: ignore[arg-type]
    ).to_physical()
    evaluated = df.select(expr)
    expected = _expected_duration(kwargs, unit_ns)
    assert evaluated.to_series().to_list() == [
        expected,
        _expected_duration(other, unit_ns),
    ]
    assert df.head(1).select(expr).to_series().item() == expected

    folded = pl.select(pl.duration(**kwargs, time_unit=time_unit))  # type: ignore[arg-type]
    assert folded.columns == ["duration"]  # (folded into a literal)
    assert folded.to_series().to_physical().item() == expected


@pytest.mark.parametrize(
    "value", [True, pl.lit("1"), pl.lit(Decimal("1.5")), pl.lit(date(2020, 1, 1))]
)
def test_duration_invalid_dtype(value: Any) -> None:
    # (literals are not folded, so they raise like columns)
    for days in (value, pl.col("x")):
        with pytest.raises(
            InvalidOperationError, match=r"expected integer or float dtype"
        ):
            pl.select(x=value).select(pl.duration(days=days))

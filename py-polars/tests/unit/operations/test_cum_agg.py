from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import PolarsDataType
    from tests.conftest import PlMonkeyPatch

KINDS = ["sum", "count", "prod", "min", "max"]


@pytest.fixture(autouse=True)
def _small_morsels(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "7")


def is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def cum_agg(values: list[Any], kind: str) -> list[Any]:
    """Compute the cumulative aggregate sequentially; min and max ignore NaN."""
    out: list[Any] = []
    acc: Any = None
    count = 0
    for v in values:
        if kind == "count":
            count += v is not None
            out.append(count)
            continue
        if v is None:
            out.append(None)
            continue
        if isinstance(v, bool) and kind in ("sum", "prod"):
            v = int(v)
        if acc is None or (kind in ("min", "max") and is_nan(acc)):
            acc = v
        elif kind == "sum":
            acc += v
        elif kind == "prod":
            acc *= v
        elif not is_nan(v):
            acc = min(acc, v) if kind == "min" else max(acc, v)
        out.append(acc)
    return out


def cum_expr(kind: str) -> pl.Expr:
    return getattr(pl.col("a"), f"cum_{kind}")()  # type: ignore[no-any-return]


def check_cum_agg(
    values: list[Any],
    kind: str,
    dtype: PolarsDataType,
    out_dtype: PolarsDataType | None = None,
) -> None:
    if out_dtype is None:
        out_dtype = pl.get_index_type() if kind == "count" else dtype
    lf = pl.LazyFrame({"a": values}, schema={"a": dtype})
    expected = pl.DataFrame({"a": cum_agg(values, kind)}, schema={"a": out_dtype})
    assert_frame_equal(lf.select(cum_expr(kind)).collect(), expected)


def zig_zag(i: int) -> int:
    return i if i % 2 == 0 else -i


def base_value(kind: str, i: int) -> int:
    """Values whose cumulative aggregate keeps changing without overflowing."""
    if kind == "prod":
        return [1, 2, 1, -1, 1][i % 5]
    return zig_zag(i)


@pytest.mark.parametrize(
    ("dtype", "out_dtype", "to_value"),
    [
        (pl.Int8, pl.Int64, lambda x: x),
        (pl.UInt16, pl.Int64, lambda x: x),
        (pl.Int32, pl.Int32, lambda x: x),
        (pl.UInt64, pl.UInt64, lambda x: x),
        (pl.Int128, pl.Int128, lambda x: x),
        (pl.Float16, pl.Float16, float),
        (pl.Float32, pl.Float32, float),
        (pl.Float64, pl.Float64, float),
        (pl.Boolean, pl.UInt32, lambda x: x % 3 == 0),
        (pl.Duration("us"), pl.Duration("us"), lambda x: timedelta(seconds=x)),
        (pl.Decimal(10, 2), pl.Decimal(38, 2), lambda x: Decimal(x) / 4),
    ],
)
def test_cum_sum_dtypes(
    dtype: PolarsDataType, out_dtype: PolarsDataType, to_value: Callable[[int], Any]
) -> None:
    values = [None if i % 11 == 3 else to_value(i % 10) for i in range(200)]
    check_cum_agg(values, "sum", dtype, out_dtype)


@pytest.mark.parametrize(
    ("dtype", "out_dtype"),
    [
        (pl.Boolean, pl.Int64),
        (pl.Int8, pl.Int64),
        (pl.UInt32, pl.Int64),
        (pl.UInt64, pl.UInt64),
        (pl.Int128, pl.Int128),
        (pl.Float16, pl.Float16),
        (pl.Float64, pl.Float64),
    ],
)
def test_cum_prod_dtypes(dtype: PolarsDataType, out_dtype: PolarsDataType) -> None:
    if dtype == pl.Boolean:
        cycle: list[Any] = [True] * 36 + [False]
    elif dtype.is_unsigned_integer():
        cycle = [1, 2, 1, 1, 1, 1, 1, 1]
    elif dtype.is_integer():
        cycle = [1, 2, 1, -1, 1, 1, 1, 1]
    else:
        cycle = [1.0, 2.0, 1.0, -1.0, 1.0, 0.5, 1.0, 1.0, 2.0]
    values = [None if i % 11 == 3 else cycle[i % len(cycle)] for i in range(100)]
    check_cum_agg(values, "prod", dtype, out_dtype)


@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize(
    ("dtype", "to_value"),
    [
        (pl.Int8, lambda x: x),
        (pl.UInt64, lambda x: x + 120),
        (pl.Int128, lambda x: x),
        (pl.Float16, lambda x: x / 2),
        (pl.Float64, lambda x: x / 2),
        (pl.String, lambda x: f"{x + 500:04d}"),
        (pl.Binary, lambda x: f"{x + 500:04d}".encode()),
        (pl.Date, lambda x: date(2020, 1, 1) + timedelta(days=x)),
        (pl.Datetime("us"), lambda x: datetime(2020, 1, 1) + timedelta(hours=x)),
        (
            pl.Datetime("us", "UTC"),
            lambda x: datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=x),
        ),
        (pl.Duration("ms"), lambda x: timedelta(seconds=x)),
        (pl.Time, lambda x: time(12 + x // 60, x % 60)),
        (pl.Decimal(10, 2), lambda x: Decimal(x) / 4),
    ],
)
def test_cum_min_max_dtypes(
    kind: str, dtype: PolarsDataType, to_value: Callable[[int], Any]
) -> None:
    values = [None if i % 11 == 3 else to_value(zig_zag(i)) for i in range(120)]
    check_cum_agg(values, kind, dtype)


@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize("first_half", [True, False])
def test_cum_min_max_boolean(kind: str, first_half: bool) -> None:
    values = [None if i % 13 == 0 else (i < 50) == first_half for i in range(100)]
    check_cum_agg(values, kind, pl.Boolean)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "is_null",
    [
        lambda i: 14 <= i < 21,
        lambda i: i < 25,
        lambda i: i >= 40,
        lambda i: True,
        lambda i: i != 20,
    ],
)
def test_cum_agg_nulls(kind: str, is_null: Callable[[int], bool]) -> None:
    values = [None if is_null(i) else base_value(kind, i) for i in range(70)]
    check_cum_agg(values, kind, pl.Int64)


def expected_frame(values: list[Any], kind: str) -> pl.DataFrame:
    return pl.DataFrame({"a": cum_agg(values, kind)}, schema={"a": pl.Int64})


@pytest.mark.parametrize("kind", ["sum", "min"])
@pytest.mark.parametrize(
    "keep",
    [
        lambda i: not (7 <= i < 35 or 50 <= i < 60),
        lambda i: False,
    ],
)
def test_cum_agg_empty_morsels(kind: str, keep: Callable[[int], bool]) -> None:
    values = [zig_zag(i) for i in range(100)]
    keep_mask = [keep(i) for i in range(100)]
    lf = pl.LazyFrame({"a": values, "keep": keep_mask})
    out = lf.filter(pl.col("keep")).select(cum_expr(kind)).collect()
    kept = [v for v, k in zip(values, keep_mask, strict=True) if k]
    assert_frame_equal(out, expected_frame(kept, kind))


@pytest.mark.parametrize("kind", ["sum", "min"])
def test_cum_agg_multiple_inputs(kind: str) -> None:
    parts = [
        [None if i % 5 == 2 else zig_zag(i) for i in range(30 * k, 30 * k + 30)]
        for k in range(3)
    ]
    lf = pl.concat([pl.LazyFrame({"a": p}) for p in parts])
    values = [v for p in parts for v in p]
    assert_frame_equal(
        lf.select(cum_expr(kind)).collect(), expected_frame(values, kind)
    )


@pytest.mark.parametrize("kind", ["sum", "min"])
def test_cum_agg_with_columns(kind: str) -> None:
    a = [None if i % 6 == 1 else zig_zag(i) for i in range(80)]
    b = list(range(100, 180))
    lf = pl.LazyFrame({"a": a, "b": b})
    expected = pl.DataFrame({"a": a, "b": b}).with_columns(
        c=expected_frame(a, kind).to_series()
    )
    assert_frame_equal(lf.with_columns(c=cum_expr(kind)).collect(), expected)


@pytest.mark.parametrize("kind", ["sum", "min"])
def test_cum_agg_head(kind: str) -> None:
    values = [None if i % 5 == 2 else zig_zag(i) for i in range(10_000)]
    lf = pl.LazyFrame({"a": values})
    out = lf.select(cum_expr(kind)).head(10).collect()
    assert_frame_equal(out, expected_frame(values[:10], kind))


@pytest.mark.parametrize("kind", ["sum", "prod"])
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_cum_sum_prod_float(kind: str, dtype: PolarsDataType) -> None:
    rng = random.Random(0)
    if kind == "sum":
        values: list[float | None] = [
            None
            if rng.random() < 0.1
            else rng.uniform(-1, 1) * 10 ** rng.randint(-3, 6)
            for _ in range(500)
        ]
    else:
        values = [
            None if rng.random() < 0.1 else rng.uniform(0.9, 1.1) for _ in range(500)
        ]
    if dtype == pl.Float32:
        # Round the inputs to float32 first, so the ground truth uses the same values.
        values = pl.Series(values, dtype=pl.Float32).to_list()

    expected: list[float | None] = []
    seen: list[float] = []
    for v in values:
        if v is None:
            expected.append(None)
        else:
            seen.append(v)
            expected.append(math.fsum(seen) if kind == "sum" else math.prod(seen))

    lf = pl.LazyFrame({"a": values}, schema={"a": dtype})
    out = lf.select(cum_expr(kind)).collect()
    assert_series_equal(
        out.to_series(),
        pl.Series("a", expected, dtype=dtype),
        rel_tol=1e-4 if dtype == pl.Float32 else 1e-12,
        abs_tol=1e-6,
    )


@pytest.mark.parametrize("kind", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize(
    "special",
    [
        {10: float("inf"), 30: float("-inf")},
        {10: 0.0, 30: float("inf")},
        {10: float("nan")},
        dict.fromkeys(range(20), float("nan")),
    ],
)
def test_cum_agg_float_special_values(
    kind: str, dtype: PolarsDataType, special: dict[int, float]
) -> None:
    cycle = [1.0, 2.0, 0.5, -1.0, 3.0]
    values = [
        special.get(
            i, None if i % 9 == 4 else cycle[i % 5] * (i if kind != "prod" else 1)
        )
        for i in range(50)
    ]
    check_cum_agg(values, kind, dtype)

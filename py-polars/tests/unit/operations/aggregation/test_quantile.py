from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from polars._typing import QuantileMethod

METHODS: list[QuantileMethod] = [
    "nearest",
    "higher",
    "lower",
    "midpoint",
    "linear",
    "equiprobable",
]

QUANTILE_SETS = [
    [0.5],
    [0.0, 1.0],
    [0.1, 0.9],
    [i / 20 for i in range(21)],
]


def _sort_key(v: Any) -> tuple[bool, Any]:
    # Polars' total order: NaN is the largest value and equal to itself.
    if isinstance(v, float) and math.isnan(v):
        return (True, 0)
    return (False, v)


def _oracle(values: list[Any], q: float, method: QuantileMethod) -> float | None:
    vals = [
        float(v) for v in sorted((v for v in values if v is not None), key=_sort_key)
    ]
    n = len(vals)
    if n == 0:
        return None
    f = (n - 1) * q
    lo = math.floor(f)
    hi = min(math.ceil(f), n - 1)
    lower, upper = vals[lo], vals[hi]
    if method == "nearest":
        return vals[lo + (f - lo >= 0.5)]
    if method == "lower":
        return lower
    if method == "higher":
        return upper
    if method == "equiprobable":
        return vals[max(math.ceil(n * q) - 1, 0)]
    if lower == upper:
        return lower
    if method == "midpoint":
        return (lower + upper) / 2
    return (f - lo) * (upper - lower) + lower


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return (math.isnan(a) and math.isnan(b)) or a == b


def _chunked(values: list[Any], dtype: pl.DataType, n_chunks: int = 3) -> pl.Series:
    size = max(len(values) // n_chunks, 1)
    parts = [values[i : i + size] for i in range(0, len(values), size)] or [[]]
    s = pl.concat([pl.Series("a", p, dtype=dtype) for p in parts], rechunk=False)
    assert s.n_chunks() == len(parts)
    return s


def _inputs() -> list[tuple[list[Any], pl.DataType]]:
    rng = random.Random(0)
    ints = [rng.randint(-50, 50) for _ in range(200)]
    ties = [rng.choice([0, 1, 2]) for _ in range(200)]
    floats = [
        rng.choice([math.nan, -0.0, 0.0, -math.inf, math.inf, None])
        if rng.random() < 0.2
        else rng.uniform(-10, 10)
        for _ in range(200)
    ]
    return [
        ([v if i % 7 else None for i, v in enumerate(ints)], pl.Int64()),
        (ints, pl.Int32()),
        ([v if i % 5 else None for i, v in enumerate(ties)], pl.UInt8()),
        (floats, pl.Float64()),
        ([3.0] * 50 + [None] * 5, pl.Float64()),
        ([None, 1, None], pl.Int64()),
        ([None, None], pl.Int64()),
    ]


@pytest.mark.parametrize("method", METHODS)
def test_quantile_matches_sorted_oracle(method: QuantileMethod) -> None:
    for values, dtype in _inputs():
        s = _chunked(values, dtype)
        for qs in QUANTILE_SETS:
            for q in qs:
                assert _same(s.quantile(q, method), _oracle(values, q, method))

            result = s.to_frame().select(pl.col("a").quantile(qs, interpolation=method))
            expected = [_oracle(values, q, method) for q in qs]
            out = result.item().to_list()
            assert all(_same(a, b) for a, b in zip(out, expected, strict=True))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_quantile_sorted_flag(
    method: QuantileMethod, descending: bool, nulls_last: bool
) -> None:
    rng = random.Random(1)
    values: list[float | None] = sorted(
        [rng.choice([math.nan, -0.0, 0.0, rng.uniform(-5, 5)]) for _ in range(60)],
        key=_sort_key,
        reverse=descending,
    )
    nulls: list[float | None] = [None] * 7
    values = values + nulls if nulls_last else nulls + values
    s = _chunked(values, pl.Float64(), n_chunks=6).set_sorted(descending=descending)
    for qs in QUANTILE_SETS:
        for q in qs:
            assert _same(s.quantile(q, method), _oracle(values, q, method))
        result = s.to_frame().select(pl.col("a").quantile(qs, interpolation=method))
        expected = [_oracle(values, q, method) for q in qs]
        out = result.item().to_list()
        assert all(_same(a, b) for a, b in zip(out, expected, strict=True))


def _check_group_by(df: pl.DataFrame, method: QuantileMethod) -> None:
    groups: dict[Any, list[Any]] = {}
    for g, v in df.iter_rows():
        groups.setdefault(g, []).append(v)
    for q in [0.0, 0.3, 0.5, 1.0]:
        result = df.group_by("g", maintain_order=True).agg(
            pl.col("v").quantile(q, interpolation=method).alias("q"),
            pl.col("v").median().alias("median"),
        )
        for g, out, median in result.iter_rows():
            assert _same(out, _oracle(groups[g], q, method))
            assert _same(median, _oracle(groups[g], 0.5, "linear"))


@pytest.mark.parametrize("method", METHODS)
def test_quantile_group_by(method: QuantileMethod) -> None:
    rng = random.Random(2)
    n = 500
    # Unsorted keys give index groups; small groups and groups of nulls included.
    for n_groups in [5, 150]:
        df = pl.DataFrame(
            {
                "g": [rng.randrange(n_groups) for _ in range(n)],
                "v": [
                    None
                    if rng.random() < 0.2
                    else rng.choice([1.5, -0.0, 0.0, math.nan])
                    if rng.random() < 0.3
                    else rng.uniform(-10, 10)
                    for _ in range(n)
                ],
            }
        )
        _check_group_by(df, method)
        _check_group_by(
            df.with_columns(pl.col("v").cast(pl.Int64, strict=False)), method
        )

    # A sorted key gives slice groups, here over unsorted multi-chunk values.
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "g": sorted(rng.randrange(40) for _ in range(100)),
                    "v": [
                        None if rng.random() < 0.2 else rng.randint(0, 9)
                        for _ in range(100)
                    ],
                }
            )
            for _ in range(3)
        ],
        rechunk=False,
    ).with_columns(pl.col("g") + pl.int_range(pl.len()) // 100 * 100)
    df = df.with_columns(pl.col("g").set_sorted())
    _check_group_by(df, method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_quantile_group_by_sorted_values(
    method: QuantileMethod, descending: bool, nulls_last: bool
) -> None:
    rng = random.Random(3)
    ints = sorted((rng.randint(0, 100) for _ in range(300)), reverse=descending)
    nulls: list[int | None] = [None] * 20
    v = [*ints, *nulls] if nulls_last else [*nulls, *ints]
    df = pl.DataFrame({"g": [i // 7 for i in range(len(v))], "v": v}).with_columns(
        pl.col("g").set_sorted(), pl.col("v").set_sorted(descending=descending)
    )
    _check_group_by(df, method)


@pytest.mark.parametrize("method", METHODS)
def test_quantile_large_matches_sorted(method: QuantileMethod) -> None:
    # Large enough for the selection to recurse in parallel; the sorted input takes the
    # independent lookup path.
    rng = random.Random(4)
    values = [
        None if rng.random() < 0.05 else rng.randint(-1000, 1000)
        for _ in range(300_000)
    ]
    s = _chunked(values, pl.Int64())
    qs = [i / 100 for i in range(101)]
    expected = (
        s.sort().to_frame().select(pl.col("a").quantile(qs, interpolation=method))
    )
    result = s.to_frame().select(pl.col("a").quantile(qs, interpolation=method))
    assert result.item().to_list() == expected.item().to_list()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["mostly_null", "two_values", "constant", "heavy"])
def test_quantile_large_single_matches_sorted(
    method: QuantileMethod, kind: str
) -> None:
    # A single quantile of a large column only copies the values near it, and counts
    # those equal to the bounds around it when values repeat; mostly-null chunks are
    # filtered first.
    rng = random.Random(10)
    n = 300_000
    values: list[float | None]
    if kind == "mostly_null":
        values = [None if rng.random() < 0.7 else rng.uniform(-1, 1) for _ in range(n)]
    elif kind == "two_values":
        # The median lands just past the zeros, among the values equal to the upper
        # bound.
        values = [0.0 if i < n // 2 - 10 else 1.0 for i in range(n)]
        rng.shuffle(values)
    elif kind == "constant":
        values = [2.5 for _ in range(n)]
    else:
        # The quantiles around 0.28 land at the start of the run of sevens.
        values = [7.0 if rng.random() < 0.6 else rng.uniform(0, 10) for _ in range(n)]
    s = _chunked(values, pl.Float64())
    expected = s.sort()
    for q in [0.0, 0.2, 0.27, 0.28, 0.29, 0.5, 0.7, 1.0]:
        assert s.quantile(q, method) == expected.quantile(q, method)


@pytest.mark.parametrize("method", METHODS)
def test_quantile_large_floats_matches_sorted(method: QuantileMethod) -> None:
    # The parallel selection with NaN, signed zeros and infinities, whose tie
    # classes and NaN pivots the integer test doesn't reach.
    rng = random.Random(5)
    specials = [float("nan"), -0.0, 0.0, float("inf"), float("-inf")]

    def value() -> float | None:
        r = rng.random()
        if r < 0.05:
            return None
        if r < 0.2:
            return rng.choice(specials)
        return rng.uniform(-1000, 1000)

    values = [value() for _ in range(300_000)]
    s = _chunked(values, pl.Float64())
    qs = [i / 100 for i in range(101)]
    expected = (
        s.sort().to_frame().select(pl.col("a").quantile(qs, interpolation=method))
    )
    result = s.to_frame().select(pl.col("a").quantile(qs, interpolation=method))
    assert_series_equal(result.item(), expected.item())


@pytest.mark.parametrize("method", METHODS)
def test_quantile_group_by_large_group(method: QuantileMethod) -> None:
    # A group large enough for the selection to recurse in parallel inside the group-by.
    rng = random.Random(6)
    n = 200_000
    df = pl.DataFrame(
        {
            "g": [0] * n + [1] * 5,
            "v": [
                None if rng.random() < 0.05 else rng.randint(-100, 100)
                for _ in range(n + 5)
            ],
        }
    ).sample(fraction=1.0, shuffle=True, seed=0)
    _check_group_by(df, method)

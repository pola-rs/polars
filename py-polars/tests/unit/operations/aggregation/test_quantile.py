from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

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

# Sorted, the non-null values are 1 1 2 3 4 5 6 9; the quantiles fall at positions
# 0, 1.75, 3.5, 5.25 and 7.
VALUES = [3, None, 1, 4, 1, 5, 9, 2, 6]
QS = [0.0, 0.25, 0.5, 0.75, 1.0]
EXPECTED: dict[QuantileMethod, list[float]] = {
    "nearest": [1, 2, 4, 5, 9],
    "higher": [1, 2, 4, 6, 9],
    "lower": [1, 1, 3, 5, 9],
    "midpoint": [1, 1.5, 3.5, 5.5, 9],
    "linear": [1, 1.75, 3.5, 5.25, 9],
    "equiprobable": [1, 1, 3, 5, 9],
}


def _chunked(values: list[Any], dtype: pl.DataType, n_chunks: int = 3) -> pl.Series:
    size = max(len(values) // n_chunks, 1)
    parts = [values[i : i + size] for i in range(0, len(values), size)] or [[]]
    s = pl.concat([pl.Series("a", p, dtype=dtype) for p in parts], rechunk=False)
    assert s.n_chunks() == len(parts)
    return s


def _quantiles(s: pl.Series, qs: list[float], method: QuantileMethod) -> pl.Series:
    out = s.to_frame().select(pl.col("a").quantile(qs, interpolation=method))
    return out.to_series().explode().rename("")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64()])
@pytest.mark.parametrize("order", [None, "asc", "desc"])
def test_quantile_methods(
    method: QuantileMethod, dtype: pl.DataType, order: str | None
) -> None:
    s = _chunked(VALUES, dtype)
    if order is not None:
        # Flagged sorted, the quantiles are read without selecting.
        descending = order == "desc"
        s = s.sort(descending=descending, nulls_last=descending)
        s = _chunked(s.to_list(), dtype).set_sorted(descending=descending)

    assert [s.quantile(q, method) for q in QS] == EXPECTED[method]
    assert_series_equal(
        _quantiles(s, QS, method), pl.Series(EXPECTED[method], dtype=pl.Float64)
    )


@pytest.mark.parametrize("method", ["lower", "higher"])
def test_quantile_floats_total_order(method: QuantileMethod) -> None:
    # Sorted: -inf, the two zeros, 1, 2, inf, NaN; NaN is the largest value.
    s = _chunked(
        [math.nan, 2.0, -math.inf, 0.0, None, -0.0, math.inf, 1.0], pl.Float64()
    )
    expected = {
        "lower": [-math.inf, 0.0, 1.0, math.inf, math.nan],
        "higher": [-math.inf, 0.0, 1.0, math.nan, math.nan],
    }[method]
    qs = [0.0, 0.25, 0.5, 0.9, 1.0]
    assert_series_equal(_quantiles(s, qs, method), pl.Series(expected))


@pytest.mark.parametrize("method", ["lower", "higher"])
@pytest.mark.parametrize(
    "kind", ["random", "few_distinct", "constant", "descending", "floats"]
)
def test_quantile_many_ranks_match_sort(method: QuantileMethod, kind: str) -> None:
    # Selecting many ranks at once, including every rank of a small column.
    rng = random.Random(0)
    n = 2000
    candidates: dict[str, list[Any]] = {
        "random": [rng.randint(-(2**62), 2**62) for _ in range(n)],
        "few_distinct": [rng.randint(0, 2) for _ in range(n)],
        "constant": [7] * n,
        "descending": list(range(n, 0, -1)),
        "floats": [
            rng.choice([math.nan, -0.0, 0.0, math.inf]) if i % 5 == 0 else rng.random()
            for i in range(n)
        ],
    }
    values = [None if i % 11 == 0 else v for i, v in enumerate(candidates[kind])]
    s = _chunked(values, pl.Float64() if kind == "floats" else pl.Int64())

    for qs in [[i / 100 for i in range(101)], [i / 9 for i in range(10)]]:
        expected = _quantiles(s.sort(), qs, method)
        assert_series_equal(_quantiles(s, qs, method), expected)
    small = s.head(50)
    all_ranks = [i / 49 for i in range(50)]
    assert_series_equal(
        _quantiles(small, all_ranks, method),
        _quantiles(small.sort(), all_ranks, method),
    )


def _check_group_by(df: pl.DataFrame, method: QuantileMethod) -> None:
    """Compares each group's quantiles with those of its sorted values."""
    qs = [0.0, 0.3, 0.5, 1.0]
    result = df.group_by("g", maintain_order=True).agg(
        *(pl.col("v").quantile(q, interpolation=method).alias(str(q)) for q in qs),
        pl.col("v").median().alias("median"),
    )
    groups = {
        g: sub["v"].sort() for (g,), sub in df.partition_by("g", as_dict=True).items()
    }
    expected = pl.DataFrame(
        {
            "g": result["g"],
            **{
                str(q): [groups[g].quantile(q, method) for g in result["g"]] for q in qs
            },
            "median": [groups[g].median() for g in result["g"]],
        },
        schema=result.schema,
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64()])
def test_quantile_group_by_fixed(method: QuantileMethod, dtype: pl.DataType) -> None:
    # Group 0 holds VALUES, group 1 only nulls and group 2 one value.
    rows = [(0, v) for v in VALUES] + [(1, None), (1, None), (2, 7)]
    expected = pl.DataFrame(
        {
            "g": [0, 1, 2],
            **{
                str(q): [e, None, 7.0]
                for q, e in zip(QS, EXPECTED[method], strict=True)
            },
        },
        schema_overrides={str(q): pl.Float64 for q in QS},
    )

    def run(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by("g")
            .agg(
                pl.col("v").quantile(q, interpolation=method).alias(str(q)) for q in QS
            )
            .sort("g")
        )

    # Shuffled keys give index groups; sorted keys over unsorted values give slices.
    shuffled = random.Random(1).sample(rows, len(rows))
    for df in [
        pl.DataFrame(shuffled, schema={"g": pl.Int64, "v": dtype}, orient="row"),
        pl.DataFrame(
            rows, schema={"g": pl.Int64, "v": dtype}, orient="row"
        ).with_columns(pl.col("g").set_sorted()),
    ]:
        assert_frame_equal(run(df), expected)


@pytest.mark.parametrize("method", METHODS)
def test_quantile_group_by_random(method: QuantileMethod) -> None:
    rng = random.Random(2)
    n = 500
    df = pl.DataFrame(
        {
            "g": [rng.randrange(150) for _ in range(n)],
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
    _check_group_by(df.with_columns(pl.col("v").cast(pl.Int64, strict=False)), method)
    _check_group_by(df.sort("g").with_columns(pl.col("g").set_sorted()), method)


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
        pl.col("g").set_sorted()
    )
    flagged = df.with_columns(pl.col("v").set_sorted(descending=descending))
    assert df["v"].flags["SORTED_ASC"] is False
    assert df["v"].flags["SORTED_DESC"] is False

    # Slices of a sorted column are read without selecting.
    def run(df: pl.DataFrame) -> pl.DataFrame:
        return df.group_by("g", maintain_order=True).agg(
            pl.col("v").quantile(q, interpolation=method).alias(str(q))
            for q in [0.0, 0.3, 0.5, 1.0]
        )

    assert_frame_equal(run(flagged), run(df))


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
    assert_series_equal(_quantiles(s, qs, method), _quantiles(s.sort(), qs, method))


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
    assert_series_equal(_quantiles(s, qs, method), _quantiles(s.sort(), qs, method))


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

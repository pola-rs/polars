from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from hypothesis import Phase, given, settings
from hypothesis import strategies as st

import polars as pl
from polars.meta import get_index_type
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric.strategies import series

if TYPE_CHECKING:
    from collections.abc import Sequence


class Case(TypedDict):
    """A test case for Skip Batch Predicate."""

    min: Any | None
    max: Any | None
    null_count: int | None
    len: int | None
    can_skip: bool


def assert_skp_series(
    name: str,
    dtype: pl.DataType,
    expr: pl.Expr,
    cases: Sequence[Case],
) -> None:
    sbp = expr._skip_batch_predicate({name: dtype})

    df = pl.DataFrame(
        [
            pl.Series(f"{name}_min", [i["min"] for i in cases], dtype),
            pl.Series(f"{name}_max", [i["max"] for i in cases], dtype),
            pl.Series(f"{name}_nc", [i["null_count"] for i in cases], get_index_type()),
            pl.Series("len", [i["len"] for i in cases], get_index_type()),
        ]
    )
    mask = pl.Series("can_skip", [i["can_skip"] for i in cases], pl.Boolean)

    out = df.select(can_skip=sbp).to_series()
    out = out.replace(None, False)

    try:
        assert_series_equal(out, mask)
    except AssertionError:
        print(sbp)
        raise


def test_equality() -> None:
    assert_skp_series(
        "a",
        pl.Int64,
        pl.col("a") == 5,
        [
            {"min": 1, "max": 2, "null_count": 0, "len": 42, "can_skip": True},
            {"min": 6, "max": 7, "null_count": 0, "len": 42, "can_skip": True},
            {"min": 1, "max": 7, "null_count": 0, "len": 42, "can_skip": False},
            {"min": None, "max": None, "null_count": 42, "len": 42, "can_skip": True},
        ],
    )

    assert_skp_series(
        "a",
        pl.Int64(),
        pl.col("a") != 0,
        [
            {"min": 0, "max": 0, "null_count": 6, "len": 7, "can_skip": False},
        ],
    )

    assert_skp_series(
        "a",
        pl.Struct(),
        pl.col("a") != 0,
        [
            {"min": None, "max": None, "null_count": 6, "len": 7, "can_skip": False},
        ],
    )


CHUNK_SIZE = 7
NUM_CHUNKS = 13
TOTAL_SIZE = CHUNK_SIZE * NUM_CHUNKS


@given(
    s=series(
        name="x",
        min_size=TOTAL_SIZE,
        max_size=TOTAL_SIZE,
        # allowed_dtypes=[
        #     pl.Int64,
        #     pl.String,
        #     pl.Date,
        #     pl.Datetime(time_zone=datetime.timezone.utc),
        #     pl.Time,
        # ],
    ),
    index_a=st.integers(0, TOTAL_SIZE - 1),
    index_b=st.integers(0, TOTAL_SIZE - 1),
)
@settings(
    report_multiple_bugs=False,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.explain),
)
def test_skip_batch_predicate_parametric(
    s: pl.Series, index_a: int, index_b: int
) -> None:
    name = "x"
    dtype = s.dtype

    value_a = s.slice(index_a, 1)
    value_b = s.slice(index_b, 1)

    lit_a = pl.lit(value_a[0], dtype)
    lit_b = pl.lit(value_b[0], dtype)

    exprs = [
        pl.col.x == lit_a,
        pl.col.x != lit_a,
        pl.col.x.eq_missing(lit_a),
        pl.col.x.ne_missing(lit_a),
        pl.col.x.is_null(),
        pl.col.x.is_not_null(),
    ]

    try:
        _ = s > value_a
        exprs += [
            pl.col.x > lit_a,
            pl.col.x >= lit_a,
            pl.col.x < lit_a,
            pl.col.x <= lit_a,
            pl.col.x.is_between(lit_a, lit_b),
            pl.col.x.is_in(pl.Series([value_a[0], value_b[0]], dtype=dtype)),
            pl.col.x.is_in(pl.Series([None, value_a[0]], dtype=dtype)),
        ]
    except Exception as _:
        pass

    for expr in exprs:
        sbp = expr._skip_batch_predicate({name: dtype})

        mins = [None] * NUM_CHUNKS
        try:
            mins = [
                s.slice(i * CHUNK_SIZE, CHUNK_SIZE).min() for i in range(NUM_CHUNKS)
            ]
        except Exception as _:
            mins = [None] * NUM_CHUNKS
        try:
            maxs = [
                s.slice(i * CHUNK_SIZE, CHUNK_SIZE).max() for i in range(NUM_CHUNKS)
            ]
        except Exception as _:
            maxs = [None] * NUM_CHUNKS
        null_counts = [
            s.slice(i * CHUNK_SIZE, CHUNK_SIZE).null_count() for i in range(NUM_CHUNKS)
        ]
        lengths = [CHUNK_SIZE] * NUM_CHUNKS

        df = pl.DataFrame(
            [
                pl.Series(f"{name}_min", mins, dtype),
                pl.Series(f"{name}_max", maxs, dtype),
                pl.Series(f"{name}_nc", null_counts, get_index_type()),
                pl.Series("len", lengths, get_index_type()),
            ]
        )

        out = df.select(can_skip=sbp).fill_null(False).to_series()

        included = []
        for i, can_skip in enumerate(out):
            if not can_skip:
                included += [s.slice(i * CHUNK_SIZE, CHUNK_SIZE).to_frame()]

        skipped_batches_df: pl.DataFrame
        if len(included) == 0:
            skipped_batches_df = s.head(0).to_frame()
        else:
            skipped_batches_df = pl.concat(included)

        try:
            assert_frame_equal(
                s.to_frame().filter(expr),
                skipped_batches_df.filter(expr),
            )
        except Exception as _:
            print(expr)
            print(sbp)
            raise

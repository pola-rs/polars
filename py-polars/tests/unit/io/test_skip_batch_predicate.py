from __future__ import annotations

import contextlib
import datetime
from typing import TYPE_CHECKING, Any, TypedDict

from hypothesis import Phase, given, settings

import polars as pl
from polars.meta import get_index_type
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric.strategies import series

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars._typing import PythonLiteral


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


def test_true_false_predicate() -> None:
    true_sbp = pl.lit(True)._skip_batch_predicate({})
    false_sbp = pl.lit(False)._skip_batch_predicate({})
    null_sbp = pl.lit(None)._skip_batch_predicate({})

    df = pl.DataFrame({"len": [1]})

    out = df.select(
        true=true_sbp,
        false=false_sbp,
        null=null_sbp,
    )

    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "true": [False],
                "false": [True],
                "null": [True],
            }
        ),
    )


def test_equality() -> None:
    assert_skp_series(
        "a",
        pl.Int64(),
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


def test_datetimes() -> None:
    d = datetime.datetime(2023, 4, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    td = datetime.timedelta

    assert_skp_series(
        "a",
        pl.Datetime(time_zone=datetime.timezone.utc),
        pl.col("a") == d,
        [
            {
                "min": d - td(days=2),
                "max": d - td(days=1),
                "null_count": 0,
                "len": 42,
                "can_skip": True,
            },
            {
                "min": d + td(days=1),
                "max": d - td(days=2),
                "null_count": 0,
                "len": 42,
                "can_skip": True,
            },
            {"min": d, "max": d, "null_count": 42, "len": 42, "can_skip": True},
            {"min": d, "max": d, "null_count": 0, "len": 42, "can_skip": False},
            {
                "min": d - td(days=2),
                "max": d + td(days=2),
                "null_count": 0,
                "len": 42,
                "can_skip": False,
            },
            {
                "min": d + td(days=1),
                "max": None,
                "null_count": None,
                "len": None,
                "can_skip": True,
            },
        ],
    )


@given(
    s=series(
        name="x",
        min_size=1,
    ),
)
@settings(
    report_multiple_bugs=False,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.explain),
)
def test_skip_batch_predicate_parametric(s: pl.Series) -> None:
    name = "x"
    dtype = s.dtype

    value_a = s.slice(0, 1)

    lit_a = pl.lit(value_a[0], dtype)

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
            pl.col.x.is_in(pl.Series([None, value_a[0]], dtype=dtype)),
        ]

        if s.len() > 1:
            value_b = s.slice(1, 1)
            lit_b = pl.lit(value_b[0], dtype)

            exprs += [
                pl.col.x.is_between(lit_a, lit_b),
                pl.col.x.is_in(pl.Series([value_a[0], value_b[0]], dtype=dtype)),
            ]
    except Exception as _:
        pass

    for expr in exprs:
        sbp = expr._skip_batch_predicate({name: dtype})

        if sbp is None:
            continue

        mins: list[PythonLiteral | None] = [None]
        with contextlib.suppress(Exception):
            mins = [s.min()]

        maxs: list[PythonLiteral | None] = [None]
        with contextlib.suppress(Exception):
            maxs = [s.max()]

        null_counts = [s.null_count()]
        lengths = [s.len()]

        df = pl.DataFrame(
            [
                pl.Series(f"{name}_min", mins, dtype),
                pl.Series(f"{name}_max", maxs, dtype),
                pl.Series(f"{name}_nc", null_counts, get_index_type()),
                pl.Series("len", lengths, get_index_type()),
            ]
        )

        can_skip = df.select(can_skip=sbp).fill_null(False).to_series()[0]
        if can_skip:
            try:
                assert s.to_frame().filter(expr).height == 0
            except Exception as _:
                print(expr)
                print(sbp)
                print(df)
                print(s.to_frame().filter(expr))

                raise

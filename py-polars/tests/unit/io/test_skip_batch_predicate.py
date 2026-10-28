from __future__ import annotations

import contextlib
import datetime
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from hypothesis import Phase, given, settings

import polars as pl
from polars.meta import get_index_type
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric.strategies import series

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polars._typing import PolarsDataType, PythonLiteral


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


def test_not_bool() -> None:
    assert_skp_series(
        "b",
        pl.Boolean(),
        ~pl.col("b"),
        [
            {"min": True, "max": True, "null_count": 0, "len": 42, "can_skip": True},
            {"min": True, "max": True, "null_count": 5, "len": 42, "can_skip": True},
            {"min": None, "max": None, "null_count": 42, "len": 42, "can_skip": True},
            {"min": False, "max": False, "null_count": 0, "len": 42, "can_skip": False},
            {"min": False, "max": True, "null_count": 0, "len": 42, "can_skip": False},
            {"min": False, "max": False, "null_count": 5, "len": 42, "can_skip": False},
            {"min": False, "max": True, "null_count": 5, "len": 42, "can_skip": False},
            {"min": None, "max": None, "null_count": 5, "len": 42, "can_skip": False},
        ],
    )


def test_datetimes() -> None:
    d = datetime.datetime(2023, 4, 1, 0, 0, 0, tzinfo=datetime.UTC)
    td = datetime.timedelta

    assert_skp_series(
        "a",
        pl.Datetime(time_zone=datetime.UTC),
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


def _null_count_dtype(dtype: PolarsDataType) -> PolarsDataType:
    """Statistics-frame dtype of the ``<col>_nc`` column, mirroring the Rust producer.

    Scalar (and non-struct nested) columns carry a single index-typed null count;
    struct columns carry a per-field count mirroring the column shape (each leaf
    replaced by the index type), so the predicate can read an individual field's
    count via ``col("<col>_nc").struct.field(..)``.
    """
    if isinstance(dtype, pl.Struct):
        return pl.Struct(
            [pl.Field(f.name, _null_count_dtype(f.dtype)) for f in dtype.fields]
        )
    return get_index_type()


def _null_count_value(s: pl.Series) -> Any:
    """Per-field null counts mirroring the struct shape.

    A null struct nulls every leaf, so each leaf count includes parent-null rows.
    """
    if isinstance(s.dtype, pl.Struct):
        return {
            f.name: _null_count_value(s.struct.field(f.name)) for f in s.dtype.fields
        }
    return s.null_count()


@given(
    s=series(
        name="x",
        min_size=1,
        excluded_dtypes=pl.Extension,  # literals with structs containing extensions are not supported yet
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

        null_counts = [_null_count_value(s)]
        lengths = [s.len()]

        df = pl.DataFrame(
            [
                pl.Series(f"{name}_min", mins, dtype),
                pl.Series(f"{name}_max", maxs, dtype),
                pl.Series(f"{name}_nc", null_counts, _null_count_dtype(dtype)),
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


def test_float_skip_batch_predicate() -> None:
    schema = {"x": pl.Float64()}
    NaN = float("nan")

    def sbp(e: pl.Expr) -> pl.Expr | None:
        return e._skip_batch_predicate(schema)

    assert sbp(pl.col("x") < 5.0) is not None  # Can skip. NaN never satisfies <.
    assert sbp(pl.col("x") < NaN) is not None  # Can skip. NaN never satisfies <.
    assert sbp(pl.col("x") <= 5.0) is not None  # Can skip. NaN never satisfies <=.
    assert sbp(pl.col("x") <= NaN) is not None  # Can skip. NaN never satisfies <=.
    assert sbp(pl.col("x") == 5.0) is not None  # Can skip. NaN != 5.0.
    assert sbp(pl.col("x") == NaN) is None  # No skip. Stats exclude NaN.
    assert sbp(pl.col("x") != 5.0) is None  # No skip. Hidden NaN != x is true.
    assert sbp(pl.col("x") != NaN) is None  # No skip. Stats exclude NaN.
    assert sbp(pl.col("x") > 5.0) is None  # No skip. Hidden NaN satisfies >.
    assert sbp(pl.col("x") > NaN) is not None  # Can skip. Nothing > NaN under TotalOrd.
    assert sbp(pl.col("x") >= 5.0) is None  # No skip. Hidden NaN satisfies >=.
    assert sbp(pl.col("x") >= NaN) is None  # No skip. Stats exclude NaN.
    assert (
        sbp(pl.lit(5.0) > pl.col("x")) is not None
    )  # Can skip. 5.0 > col is col < 5.0.
    assert sbp(pl.lit(5.0) < pl.col("x")) is None  # No skip. 5.0 < col is col > 5.0.
    assert (
        sbp(pl.col("x").is_between(2.0, 4.0)) is not None
    )  # Can skip. Non-NaN bounds.
    assert sbp(pl.col("x").is_between(NaN, 4.0)) is None  # No skip. NaN left bound.
    assert sbp(pl.col("x").is_between(1.0, NaN)) is None  # No skip. NaN right bound.


def test_is_in_skip_batch_predicate_is_exact_for_large_haystacks() -> None:
    """A haystack above the unrolled limit tests needles, not the haystack's hull.

    Needles 0..74 and 100..149: a batch on [75, 99] sits inside the hull
    [0, 149] but holds no needle, and must still be skippable.
    """
    dtype = pl.Int64()
    needles = [*range(75), *range(100, 150)]  # 125 needles, hull [0, 149]

    assert_skp_series(
        "x",
        dtype,
        pl.col("x").is_in(needles),
        [
            {"min": 0, "max": 74, "null_count": 0, "len": 10, "can_skip": False},
            {"min": 75, "max": 99, "null_count": 0, "len": 10, "can_skip": True},
            {"min": 100, "max": 149, "null_count": 0, "len": 10, "can_skip": False},
            {"min": 74, "max": 100, "null_count": 0, "len": 10, "can_skip": False},
            {"min": 150, "max": 199, "null_count": 0, "len": 10, "can_skip": True},
            # A single known statistic outside the hull still settles it.
            {"min": 150, "max": None, "null_count": 0, "len": 10, "can_skip": True},
            {"min": None, "max": -1, "null_count": 0, "len": 10, "can_skip": True},
            {"min": 80, "max": None, "null_count": 0, "len": 10, "can_skip": False},
            # Undefined statistics settle nothing.
            {"min": None, "max": None, "null_count": 0, "len": 10, "can_skip": False},
            # Every row null, and the haystack holds no null: nothing matches.
            {"min": None, "max": None, "null_count": 10, "len": 10, "can_skip": True},
        ],
    )


def test_is_in_skip_batch_predicate_strings() -> None:
    needles = [f"{i:04}" for i in [*range(75), *range(100, 150)]]

    assert_skp_series(
        "x",
        pl.String(),
        pl.col("x").is_in(needles),
        [
            {
                "min": "0075",
                "max": "0099",
                "null_count": 0,
                "len": 10,
                "can_skip": True,
            },
            {
                "min": "0070",
                "max": "0080",
                "null_count": 0,
                "len": 10,
                "can_skip": False,
            },
            {
                "min": "0150",
                "max": "9999",
                "null_count": 0,
                "len": 10,
                "can_skip": True,
            },
        ],
    )


def test_is_in_skip_batch_predicate_unsearchable_dtype() -> None:
    """Without a binary search, a haystack above the limit falls back to its hull."""
    dtype = pl.Int128()
    haystack = pl.List(dtype)
    small = pl.lit([*range(10), *range(100, 110)], dtype=haystack)
    large = pl.lit([*range(75), *range(100, 150)], dtype=haystack)

    assert_skp_series(
        "x",
        dtype,
        pl.col("x").is_in(small),
        [
            {"min": 20, "max": 90, "null_count": 0, "len": 10, "can_skip": True},
            {"min": 5, "max": 90, "null_count": 0, "len": 10, "can_skip": False},
        ],
    )
    assert_skp_series(
        "x",
        dtype,
        pl.col("x").is_in(large),
        [
            {"min": 75, "max": 99, "null_count": 0, "len": 10, "can_skip": False},
            {"min": 150, "max": 199, "null_count": 0, "len": 10, "can_skip": True},
        ],
    )


def test_is_in_skip_batch_predicate_floats() -> None:
    NaN = float("nan")
    schema = {"x": pl.Float64()}

    # Statistics leave out NaN, so a NaN needle cannot be ruled out.
    assert pl.col("x").is_in([1.0, NaN])._skip_batch_predicate(schema) is None

    assert_skp_series(
        "x",
        pl.Float64(),
        pl.col("x").is_in([1.0, 2.0, 5.0]),
        [
            {"min": 2.5, "max": 4.5, "null_count": 0, "len": 10, "can_skip": True},
            {"min": 1.5, "max": 2.5, "null_count": 0, "len": 10, "can_skip": False},
            {"min": 6.0, "max": 9.0, "null_count": 0, "len": 10, "can_skip": True},
            {"min": 3.0, "max": 3.0, "null_count": 0, "len": 10, "can_skip": True},
            # A writer that lets NaN into the statistics.
            {"min": NaN, "max": NaN, "null_count": 0, "len": 10, "can_skip": False},
            {"min": NaN, "max": 9.0, "null_count": 0, "len": 10, "can_skip": False},
        ],
    )


@pytest.mark.parametrize(
    "needles",
    [
        [0.0],
        [-0.0],
        [1.5, 3.0],
        [float(i) for i in range(0, 300, 2)],
        [float("nan")],
        [1.5, float("nan")],
        [float("inf"), float("-inf")],
    ],
)
def test_is_in_float_row_group_pruning_matches_eager(
    needles: list[float], tmp_path: Path
) -> None:
    NaN = float("nan")
    values = [
        *[-0.0, 0.0, 1.5],
        *[NaN, 3.0, NaN],
        *[float(i) for i in range(1, 150, 2)],
        *[float("inf"), float("-inf"), None],
        *[NaN] * 3,
        *[0.0] * 3,
    ]
    df = pl.DataFrame({"x": values}, schema={"x": pl.Float64})
    path = tmp_path / "f.parquet"
    df.write_parquet(path, row_group_size=3)

    expr = pl.col("x").is_in(needles)
    assert_frame_equal(pl.scan_parquet(path).filter(expr).collect(), df.filter(expr))

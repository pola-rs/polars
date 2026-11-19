from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ShapeError
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

# A single row, so that every non-scalar argument has the length of a scalar at runtime.
FRAME = pl.LazyFrame(
    {
        "a": [3.0],
        "v": [3],
        "s": ["xa"],
        "n": [1],
        "i": [0],
        "q": [0.5],
        "d": [date(2024, 1, 1)],
        "p": ["a"],
        "r": ["1"],
        "ps": [["a"]],
        "rs": [["1"]],
        "vl": [[3]],
    }
)
SCHEMA = FRAME.collect_schema()
# The flat column whose implode gives the value of a list column.
FLAT = {"ps": "p", "rs": "r", "vl": "v"}

# Name, the column holding a valid value of the argument, the expression taking it, and
# whether a Python value of the argument means the same as that column's value.
ARGUMENTS: list[tuple[str, str, Callable[[Any], pl.Expr], bool]] = [
    ("head", "n", lambda x: pl.col("a").head(x), True),
    ("tail", "n", lambda x: pl.col("a").tail(x), True),
    ("slice-offset", "i", lambda x: pl.col("a").slice(x, 1), True),
    ("slice-length", "n", lambda x: pl.col("a").slice(0, x), True),
    ("top_k", "n", lambda x: pl.col("a").top_k(x), True),
    ("bottom_k", "n", lambda x: pl.col("a").bottom_k(x), True),
    ("top_k_by", "n", lambda x: pl.col("a").top_k_by("v", x), True),
    ("bottom_k_by", "n", lambda x: pl.col("a").bottom_k_by("v", x), True),
    ("sample-n", "n", lambda x: pl.col("a").sample(n=x, seed=0), True),
    ("sample-fraction", "q", lambda x: pl.col("a").sample(fraction=x, seed=0), True),
    ("quantile", "q", lambda x: pl.col("a").quantile(x), True),
    ("approx_quantile", "q", lambda x: pl.col("a").approx_quantile(x), True),
    ("pct_change", "n", lambda x: pl.col("a").pct_change(x), True),
    ("get", "i", lambda x: pl.col("a").get(x), True),
    ("shift", "n", lambda x: pl.col("a").shift(x), True),
    ("shift-fill_value", "q", lambda x: pl.col("a").shift(1, fill_value=x), True),
    ("diff", "n", lambda x: pl.col("a").diff(x), True),
    ("extend_constant-value", "q", lambda x: pl.col("a").extend_constant(x, 1), True),
    ("extend_constant-n", "n", lambda x: pl.col("a").extend_constant(0.0, x), True),
    ("repeat-value", "n", lambda x: pl.repeat(x, 2), True),
    ("repeat-n", "n", lambda x: pl.repeat(0, x), True),
    ("index_of", "v", lambda x: pl.col("v").index_of(x), True),
    ("int_range-start", "n", lambda x: pl.int_range(x, 5), True),
    ("int_range-end", "n", lambda x: pl.int_range(0, x), True),
    ("linear_space-num_samples", "n", lambda x: pl.linear_space(0, 1, x), True),
    ("date_range-start", "d", lambda x: pl.date_range(x, date(2024, 1, 3)), True),
    ("list.count_matches", "v", lambda x: pl.col("vl").list.count_matches(x), True),
    (
        "arr.count_matches",
        "v",
        lambda x: pl.col("vl").cast(pl.Array(pl.Int64, 1)).arr.count_matches(x),
        True,
    ),
    ("str.replace", "p", lambda x: pl.col("s").str.replace(x, "_"), True),
    ("str.replace_all", "p", lambda x: pl.col("s").str.replace_all(x, "_"), True),
    ("str.contains_any", "ps", lambda x: pl.col("s").str.contains_any(x), True),
    (
        "str.replace_many-patterns",
        "ps",
        lambda x: pl.col("s").str.replace_many(x, ["1"]),
        True,
    ),
    (
        "str.replace_many-replace_with",
        "rs",
        lambda x: pl.col("s").str.replace_many(["a"], x),
        True,
    ),
    # A Python list here is a flat lookup column, not one list.
    ("replace-old", "vl", lambda x: pl.col("v").replace(x, 0), False),
    ("replace_strict-new", "vl", lambda x: pl.col("v").replace_strict(3, x), False),
]


def _series(column: str) -> pl.Series:
    return FRAME.select(column).collect().to_series()


def _scalar_forms(column: str, python_value: bool) -> dict[str, Any]:
    forms: dict[str, Any] = {
        "first": pl.col(column).first(),
        "item": pl.col(column).head(1).item(),
        "series-item": pl.lit(_series(column)).item(),
    }
    if column in FLAT:
        forms["implode"] = pl.col(FLAT[column]).implode()
    if python_value:
        forms["python"] = _series(column).to_list()[0]
    return forms


def _non_scalar_forms(column: str) -> dict[str, pl.Expr]:
    return {
        "column": pl.col(column),
        "head": pl.col(column).head(1),
        "series": pl.lit(_series(column)),
    }


SCALAR_CASES = [
    pytest.param(build, value, id=f"{name}-{form}")
    for name, column, build, python_value in ARGUMENTS
    for form, value in _scalar_forms(column, python_value).items()
]
NON_SCALAR_CASES = [
    pytest.param(build, value, id=f"{name}-{form}")
    for name, column, build, _ in ARGUMENTS
    for form, value in _non_scalar_forms(column).items()
]


@pytest.mark.parametrize(("build", "argument"), SCALAR_CASES)
def test_scalar_argument(build: Callable[[Any], pl.Expr], argument: Any) -> None:
    FRAME.select(build(argument)).collect()


@pytest.mark.parametrize(("build", "argument"), NON_SCALAR_CASES)
def test_non_scalar_argument_raises_at_planning(
    build: Callable[[Any], pl.Expr], argument: pl.Expr
) -> None:
    with pytest.raises(ShapeError, match="must be a scalar value"):
        FRAME.select(build(argument)).collect_schema()


def test_get_index_must_be_scalar() -> None:
    lf = pl.LazyFrame({"x": [1, 2, 3], "n": [1, 2, 3], "i": [0, 1, 2]})
    with pytest.raises(ShapeError, match="'index' must be a scalar value"):
        lf.select(pl.col("x").head(pl.col("n").get(pl.col("i")))).collect_schema()

    out = lf.select(pl.col("x").head(pl.col("n").get(pl.col("i").first())))
    assert out.collect().to_series().to_list() == [1]
    out = lf.select(pl.col("n").gather(pl.col("i")))
    assert out.collect().to_series().to_list() == [1, 2, 3]


def test_scalar_argument_per_group() -> None:
    lf = pl.LazyFrame({"g": [1, 1, 2], "a": [1.0, 2.0, 3.0], "n": [1, 2, 1]})

    out = lf.group_by("g", maintain_order=True).agg(
        pl.col("a").head(pl.col("n").first())
    )
    assert out.collect()["a"].to_list() == [[1.0], [3.0]]
    out = lf.select(pl.col("a").pct_change(pl.col("n").first()).over("g"))
    assert out.collect().to_series().to_list() == [None, 1.0, None]

    with pytest.raises(ShapeError, match="must be a scalar value"):
        lf.group_by("g").agg(pl.col("a").head(pl.col("n").head(1))).collect_schema()
    with pytest.raises(ShapeError, match="must be a scalar value"):
        lf.select(
            pl.col("a").pct_change(pl.col("n").head(1)).over("g")
        ).collect_schema()


def test_str_replace_value_may_vary_per_row() -> None:
    lf = pl.LazyFrame({"s": ["xa", "ya"], "p": ["a", "b"], "r": ["1", "2"]})
    pattern = pl.col("p").first()
    out = lf.select(
        pl.col("s").str.replace(pattern, pl.col("r")).alias("one"),
        pl.col("s").str.replace_all(pattern, pl.col("r")).alias("all"),
    ).collect()
    assert out["one"].to_list() == ["x1", "y2"]
    assert out["all"].to_list() == ["x1", "y2"]


def test_series_method_with_series_argument() -> None:
    s = pl.Series("a", [1.0, 2.0, 4.0])
    with pytest.raises(ShapeError, match="must be a scalar value"):
        s.pct_change(pl.Series([1]))
    with pytest.raises(ShapeError, match="must be a scalar value"):
        s.top_k(pl.Series([1]))  # type: ignore[arg-type]
    assert_series_equal(
        s.pct_change(pl.Series([1]).item()), pl.Series("a", [None, 1.0, 1.0])
    )

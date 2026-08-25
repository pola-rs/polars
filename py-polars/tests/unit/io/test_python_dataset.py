from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

import polars as pl
from polars._plr import PyLazyFrame
from polars._utils.wrap import wrap_ldf
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterator


class CapturingDataset:
    """Minimal dataset provider that records the predicate Polars lowers for it."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df
        self.arrow_schema = df.to_arrow().schema
        self.pyarrow_predicate: str | None = None

    def schema(self) -> pa.Schema:
        return self.arrow_schema

    def to_dataset_scan(
        self, *, pyarrow_predicate: str | None = None, **_kwargs: Any
    ) -> tuple[pl.LazyFrame, str]:
        self.pyarrow_predicate = pyarrow_predicate

        def impl(*_args: Any, **_kwargs: Any) -> tuple[Iterator[pl.DataFrame], bool]:
            # Return everything: the engine re-applies the predicate itself.
            return iter([self.df]), False

        lf = pl.LazyFrame._scan_python_function(
            self.arrow_schema, impl, pyarrow=True, is_pure=True
        )

        return lf, "v1"


def lowered_predicate(
    df: pl.DataFrame, predicate: pl.Expr, *, partial: bool = False
) -> str | None:
    """
    Return the predicate string handed to a dataset provider for `predicate`.

    Also asserts that the scan produces correct results, and - when something was
    lowered - that the string builds a PyArrow expression selecting the same rows
    (a superset of them, if only part of the predicate was lowered).
    """
    dataset = CapturingDataset(df)
    lf = wrap_ldf(PyLazyFrame.new_from_dataset_object(dataset))

    expected = df.filter(predicate)
    assert_frame_equal(lf.filter(predicate).collect(), expected)

    pyarrow_predicate = dataset.pyarrow_predicate

    if pyarrow_predicate is not None:
        from polars._utils.convert import (
            to_py_date,
            to_py_datetime,
            to_py_time,
            to_py_timedelta,
        )

        # The provider protocol says this is valid Python building a PyArrow
        # expression - the same environment `scan_delta` evaluates it in.
        expr = eval(
            pyarrow_predicate,
            {
                "pa": pa,
                "to_py_date": to_py_date,
                "to_py_datetime": to_py_datetime,
                "to_py_time": to_py_time,
                "to_py_timedelta": to_py_timedelta,
            },
        )
        filtered = pl.from_arrow(ds.dataset(df.to_arrow()).to_table(filter=expr))
        assert isinstance(filtered, pl.DataFrame)

        if partial:
            assert_frame_equal(filtered.filter(predicate), expected)
        else:
            assert_frame_equal(filtered, expected)

    return pyarrow_predicate


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "cat": ["alpha", "beta", "gamma", "delta", "alpha", "beta"],
            "val": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        }
    )


def test_dataset_provider_predicate_comparison(df: pl.DataFrame) -> None:
    assert lowered_predicate(df, pl.col("id") > 3) == "(pa.compute.field('id') > 3)"


def test_dataset_provider_predicate_is_in(df: pl.DataFrame) -> None:
    assert (
        lowered_predicate(df, pl.col("cat").is_in(["alpha", "beta"]))
        == '(pa.compute.field(\'cat\')).isin(["alpha","beta"])'
    )
    assert (
        lowered_predicate(df, pl.col("id").is_in([1, 3, 5]))
        == "(pa.compute.field('id')).isin([1,3,5])"
    )


def test_dataset_provider_predicate_is_in_empty(df: pl.DataFrame) -> None:
    assert lowered_predicate(df, pl.col("id").is_in([])) == "pa.compute.scalar(False)"


def test_dataset_provider_predicate_is_in_nulls_equal(df: pl.DataFrame) -> None:
    df = df.with_columns(pl.when(pl.col("id") > 3).then(pl.col("cat")).alias("cat"))

    # Nulls are dropped from the haystack unless they are to compare equal.
    assert (
        lowered_predicate(df, pl.col("cat").is_in(["alpha", None]))
        == "(pa.compute.field('cat')).isin([\"alpha\"])"
    )
    assert (
        lowered_predicate(df, pl.col("cat").is_in(["alpha", None], nulls_equal=True))
        == "(pa.compute.field('cat')).isin([\"alpha\",None])"
    )


def test_dataset_provider_predicate_arithmetic(df: pl.DataFrame) -> None:
    assert (
        lowered_predicate(df, pl.col("val") * 2 > 1.0)
        == "((pa.compute.field('val') * 2) > 1)"
    )
    assert (
        lowered_predicate(df, pl.col("val") + 1.0 <= 1.5)
        == "((pa.compute.field('val') + 1) <= 1.5)"
    )
    assert (
        lowered_predicate(df, pl.col("id") * pl.col("id") > 4)
        == "((pa.compute.field('id') * pa.compute.field('id')) > 4)"
    )
    # PyArrow expressions have no reflected arithmetic, so a literal on the left
    # has to become an expression of its own.
    assert (
        lowered_predicate(df, 2.0 - pl.col("val") > 1.0)
        == "((pa.compute.scalar(2) - pa.compute.field('val')) > 1)"
    )


def test_dataset_provider_predicate_true_divide(df: pl.DataFrame) -> None:
    # Polars' `/` is float division, PyArrow's follows the operand types, so the
    # dividend has to be cast for an integer column not to truncate.
    assert (
        lowered_predicate(df, pl.col("id") / 4 > 1.0)
        == "(((pa.compute.field('id')).cast('double') / 4) > 1)"
    )


def test_dataset_provider_predicate_not_lowered(df: pl.DataFrame) -> None:
    # `%` has no PyArrow expression form.
    assert lowered_predicate(df, pl.col("id") % 2 == 0) is None
    # Neither does a cast.
    assert lowered_predicate(df, pl.col("id").cast(pl.String) == "1") is None


def test_dataset_provider_predicate_partial(df: pl.DataFrame) -> None:
    # Unconvertible conjuncts are dropped, the rest is still lowered.
    assert (
        lowered_predicate(
            df, (pl.col("id") > 3) & (pl.col("id") % 2 == 0), partial=True
        )
        == "(pa.compute.field('id') > 3)"
    )

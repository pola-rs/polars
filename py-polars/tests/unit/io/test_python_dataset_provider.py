"""Tests for the Python dataset provider interface (`new_from_dataset_object`)."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

import polars as pl
from polars._plr import PyLazyFrame
from polars._utils.wrap import wrap_ldf
from polars.testing import assert_frame_equal

SCHEMA = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])
TABLE = pa.table({"a": [1, 2, 3, 4], "b": ["bear", "cat", "beetle", "dog"]})


class _Provider:
    """Records what Polars hands it, and serves the whole table regardless."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def schema(self) -> pa.Schema:
        return SCHEMA

    def _scan(self, kwargs: dict[str, Any]) -> tuple[pl.LazyFrame, str]:
        self.calls.append(kwargs)
        projection = kwargs.get("projection")

        def impl(*_args: Any, **_kwargs: Any) -> tuple[Any, bool]:
            table = TABLE if projection is None else TABLE.select(projection)
            return iter([pl.from_arrow(table)]), False

        schema = (
            SCHEMA
            if projection is None
            else pa.schema([SCHEMA.field(name) for name in projection])
        )
        lf = pl.LazyFrame._scan_python_function(
            schema, impl, pyarrow=True, is_pure=True
        )
        return lf, "v1"


class _KwargsProvider(_Provider):
    """A provider that takes whatever Polars passes."""

    def to_dataset_scan(self, **kwargs: Any) -> tuple[pl.LazyFrame, str]:
        return self._scan(kwargs)


class _NarrowProvider(_Provider):
    """A provider written against a Polars that did not pass a predicate."""

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
    ) -> tuple[pl.LazyFrame, str]:
        return self._scan(
            {
                "existing_resolved_version_key": existing_resolved_version_key,
                "limit": limit,
                "projection": projection,
                "filter_columns": filter_columns,
                "pyarrow_predicate": pyarrow_predicate,
            }
        )


class _PredicateProvider(_Provider):
    """A provider that asks for the whole predicate by name."""

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
        serialized_predicate: bytes | None = None,
    ) -> tuple[pl.LazyFrame, str]:
        return self._scan(
            {
                "existing_resolved_version_key": existing_resolved_version_key,
                "limit": limit,
                "projection": projection,
                "filter_columns": filter_columns,
                "pyarrow_predicate": pyarrow_predicate,
                "serialized_predicate": serialized_predicate,
            }
        )


def _lf(provider: _Provider) -> pl.LazyFrame:
    return wrap_ldf(PyLazyFrame.new_from_dataset_object(provider))


def test_serialized_predicate_is_passed_when_asked_for() -> None:
    provider = _PredicateProvider()
    predicate = pl.col("b").str.starts_with("be")

    out = _lf(provider).filter(predicate).collect()

    assert_frame_equal(out, pl.DataFrame(TABLE).filter(predicate))

    (call,) = [c for c in provider.calls if c["filter_columns"] is not None]
    # Polars has no PyArrow lowering for `str.starts_with`, so a provider that
    # only reads `pyarrow_predicate` sees nothing at all here.
    assert call["pyarrow_predicate"] is None
    assert call["filter_columns"] == ["b"]

    got = pl.Expr.deserialize(call["serialized_predicate"])
    assert str(got) == str(predicate)


def test_serialized_predicate_carries_the_whole_predicate() -> None:
    """Including the parts that do lower to PyArrow, and their structure."""
    provider = _PredicateProvider()
    predicate = pl.col("b").str.starts_with("be") & (pl.col("a") > 1)

    _lf(provider).filter(predicate).collect()

    (call,) = [c for c in provider.calls if c["filter_columns"] is not None]
    # Only one conjunct lowers to PyArrow; the serialized form has both. It is
    # the optimizer's predicate, so conjuncts may be reordered and literals may
    # have picked up a dtype -- what it must not do is lose a conjunct.
    assert call["pyarrow_predicate"] is not None
    assert "starts_with" not in call["pyarrow_predicate"]

    got = str(pl.Expr.deserialize(call["serialized_predicate"]))
    assert 'col("b").str.starts_with(["be"])' in got
    assert 'col("a")) > (' in got


def test_provider_without_the_parameter_is_not_passed_one() -> None:
    """The argument list is a private protocol; adding to it must not break it."""
    provider = _NarrowProvider()
    predicate = pl.col("b").str.starts_with("be")

    out = _lf(provider).filter(predicate).collect()

    assert_frame_equal(out, pl.DataFrame(TABLE).filter(predicate))
    assert provider.calls


def test_kwargs_provider_receives_the_predicate() -> None:
    provider = _KwargsProvider()

    _lf(provider).filter(pl.col("a") > 2).collect()

    (call,) = [c for c in provider.calls if c.get("filter_columns") is not None]
    assert "serialized_predicate" in call


def test_no_predicate_means_no_serialized_predicate() -> None:
    provider = _PredicateProvider()

    _lf(provider).select("a").collect()

    assert all(call["serialized_predicate"] is None for call in provider.calls)


@pytest.mark.parametrize("with_row_index", [True, False])
def test_row_index_suppresses_the_predicate(with_row_index: bool) -> None:
    """A source acting on the predicate would number the wrong rows."""
    provider = _PredicateProvider()
    lf = _lf(provider)

    if with_row_index:
        lf = lf.with_row_index()

    lf.filter(pl.col("a") > 2).collect()

    passed = [c["serialized_predicate"] for c in provider.calls]
    assert (not any(p is not None for p in passed)) == with_row_index


def test_the_resolved_scan_is_cached_per_predicate() -> None:
    """Two predicates over the same columns must not share a resolved scan."""
    provider = _PredicateProvider()
    lf = _lf(provider)

    assert lf.filter(pl.col("b").str.starts_with("be")).collect().height == 2
    assert lf.filter(pl.col("b").str.starts_with("d")).collect().height == 1

    serialized = [
        call["serialized_predicate"]
        for call in provider.calls
        if call["serialized_predicate"] is not None
    ]
    assert len(serialized) == 2
    assert serialized[0] != serialized[1]

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_schema_equal


def test_assert_schema_equal_matching() -> None:
    schema1 = {"a": pl.Int64, "b": pl.String}
    schema2 = {"a": pl.Int64, "b": pl.String}
    assert_schema_equal(schema1, schema2)
    assert_schema_equal(pl.Schema(schema1), pl.Schema(schema2))


def test_assert_schema_equal_column_order() -> None:
    schema1 = {"a": pl.Int64, "b": pl.String}
    schema2 = {"b": pl.String, "a": pl.Int64}

    with pytest.raises(AssertionError, match="columns are not in the same order"):
        assert_schema_equal(schema1, schema2, check_column_order=True)

    assert_schema_equal(schema1, schema2, check_column_order=False)


def test_assert_schema_equal_dtypes() -> None:
    schema1 = {"a": pl.Int64, "b": pl.Float64}
    schema2 = {"a": pl.Int32, "b": pl.Float64}

    with pytest.raises(AssertionError, match="dtypes do not match"):
        assert_schema_equal(schema1, schema2, check_dtypes=True)

    assert_schema_equal(schema1, schema2, check_dtypes=False)

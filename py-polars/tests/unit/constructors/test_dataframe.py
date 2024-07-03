from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Any, Iterator, Mapping

import pytest

import polars as pl
from polars.exceptions import DataOrientationWarning, InvalidOperationError


def test_df_mixed_dtypes_string() -> None:
    data = {"x": [["abc", 12, 34.5]], "y": [1]}

    with pytest.raises(TypeError, match="unexpected value"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {"x": pl.List(pl.String), "y": pl.Int64}
    assert df.rows() == [(["abc", "12", "34.5"], 1)]


def test_df_mixed_dtypes_object() -> None:
    data = {"x": [[b"abc", 12, 34.5]], "y": [1]}

    with pytest.raises(TypeError, match="failed to determine supertype"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {"x": pl.Object, "y": pl.Int64}
    assert df.rows() == [([b"abc", 12, 34.5], 1)]


def test_df_object() -> None:
    class Foo:
        def __init__(self, value: int) -> None:
            self._value = value

        def __eq__(self, other: Any) -> bool:
            return issubclass(other.__class__, self.__class__) and (
                self._value == other._value
            )

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self._value})"

    df = pl.DataFrame({"a": [Foo(1), Foo(2)]})
    assert df["a"].dtype == pl.Object
    assert df.rows() == [(Foo(1),), (Foo(2),)]


def test_df_init_from_generator_dict_view() -> None:
    d = {0: "x", 1: "y", 2: "z"}
    data = {
        "keys": d.keys(),
        "vals": d.values(),
        "itms": d.items(),
    }
    with pytest.raises(TypeError, match="unexpected value"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {
        "keys": pl.Int64,
        "vals": pl.String,
        "itms": pl.List(pl.String),
    }
    assert df.to_dict(as_series=False) == {
        "keys": [0, 1, 2],
        "vals": ["x", "y", "z"],
        "itms": [["0", "x"], ["1", "y"], ["2", "z"]],
    }


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="reversed dict views not supported before Python 3.11",
)
def test_df_init_from_generator_reversed_dict_view() -> None:
    d = {0: "x", 1: "y", 2: "z"}
    data = {
        "rev_keys": reversed(d.keys()),
        "rev_vals": reversed(d.values()),
        "rev_itms": reversed(d.items()),
    }
    df = pl.DataFrame(data, schema_overrides={"rev_itms": pl.Object})

    assert df.schema == {
        "rev_keys": pl.Int64,
        "rev_vals": pl.String,
        "rev_itms": pl.Object,
    }
    assert df.to_dict(as_series=False) == {
        "rev_keys": [2, 1, 0],
        "rev_vals": ["z", "y", "x"],
        "rev_itms": [(2, "z"), (1, "y"), (0, "x")],
    }


def test_df_init_strict() -> None:
    data = {"a": [1, 2, 3.0]}
    schema = {"a": pl.Int8}
    with pytest.raises(TypeError):
        pl.DataFrame(data, schema=schema, strict=True)

    df = pl.DataFrame(data, schema=schema, strict=False)

    assert df["a"].to_list() == [1, 2, 3]
    assert df["a"].dtype == pl.Int8


def test_df_init_from_series_strict() -> None:
    s = pl.Series("a", [-1, 0, 1])
    schema = {"a": pl.UInt8}
    with pytest.raises(InvalidOperationError):
        pl.DataFrame(s, schema=schema, strict=True)

    df = pl.DataFrame(s, schema=schema, strict=False)

    assert df["a"].to_list() == [None, 0, 1]
    assert df["a"].dtype == pl.UInt8


# https://github.com/pola-rs/polars/issues/15471
def test_df_init_rows_overrides_non_existing() -> None:
    df = pl.DataFrame([{"a": 1}], schema_overrides={"a": pl.Int8(), "b": pl.Boolean()})
    assert df.schema == OrderedDict({"a": pl.Int8})

    df = pl.DataFrame(
        [{"a": 3, "b": 1.0}],
        schema_overrides={"a": pl.Int8, "c": pl.Utf8},
    )
    assert df.schema == OrderedDict({"a": pl.Int8, "b": pl.Float64})


# https://github.com/pola-rs/polars/issues/15245
def test_df_init_nested_mixed_types() -> None:
    data = [{"key": [{"value": 1}, {"value": 1.0}]}]

    with pytest.raises(TypeError, match="unexpected value"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)

    assert df.schema == {"key": pl.List(pl.Struct({"value": pl.Float64}))}
    assert df.to_dicts() == [{"key": [{"value": 1.0}, {"value": 1.0}]}]


def test_unit_and_empty_construction_15896() -> None:
    # This is still incorrect.
    # We should raise, but currently for len 1 dfs,
    # we cannot tell if they come from a literal or expression.
    assert "shape: (0, 2)" in str(
        pl.DataFrame({"A": [0]}).select(
            C="A",
            A=pl.int_range("A"),  # creates empty series
        )
    )


class CustomSchema(Mapping[str, Any]):
    """Dummy schema object for testing compatibility with Mapping."""

    _entries: dict[str, Any]

    def __init__(self, **named_entries: Any) -> None:
        self._items = OrderedDict(named_entries.items())

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        yield from self._items


def test_custom_schema() -> None:
    df = pl.DataFrame(schema=CustomSchema(bool=pl.Boolean, misc=pl.UInt8))
    assert df.schema == OrderedDict([("bool", pl.Boolean), ("misc", pl.UInt8)])

    with pytest.raises(TypeError):
        pl.DataFrame(schema=CustomSchema(bool="boolean", misc="unsigned int"))


def test_list_null_constructor_schema() -> None:
    expected = pl.List(pl.Null)
    assert pl.DataFrame({"a": [[]]}).dtypes[0] == expected
    assert pl.DataFrame(schema={"a": pl.List}).dtypes[0] == expected


def test_df_init_schema_object() -> None:
    schema = pl.Schema({"a": pl.Int8(), "b": pl.String()})
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, schema=schema)

    assert df.columns == schema.names()
    assert df.dtypes == schema.dtypes()


def test_df_init_data_orientation_inference_warning() -> None:
    with pytest.warns(DataOrientationWarning):
        pl.from_records([[1, 2, 3], [4, 5, 6]], schema=["a", "b", "c"])

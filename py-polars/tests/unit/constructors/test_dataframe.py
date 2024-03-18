from __future__ import annotations

import sys
from typing import Any

import pytest

import polars as pl


def test_df_mixed_dtypes_string() -> None:
    data = {"x": [["abc", 12, 34.5]], "y": [1]}

    with pytest.raises(pl.SchemaError, match="unexpected value"):
        pl.DataFrame(data, strict=True)

    df = pl.DataFrame(data, strict=False)
    assert df.schema == {"x": pl.List(pl.String), "y": pl.Int64}
    assert df.rows() == [(["abc", "12", "34.5"], 1)]


def test_df_mixed_dtypes_object() -> None:
    data = {"x": [[b"abc", 12, 34.5]], "y": [1]}
    # with pytest.raises(pl.SchemaError, match="unexpected value"):
    with pytest.raises(pl.ComputeError, match="failed to determine supertype"):
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
    with pytest.raises(pl.SchemaError, match="unexpected value"):
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

    # TODO: This should result in a Float Series without nulls
    # https://github.com/pola-rs/polars/issues/14427
    assert df["a"].to_list() == [1, 2, None]

    assert df["a"].dtype == pl.Int8


def test_df_init_from_series_strict() -> None:
    s = pl.Series("a", [-1, 0, 1])
    schema = {"a": pl.UInt8}
    with pytest.raises(pl.ComputeError):
        pl.DataFrame(s, schema=schema, strict=True)

    df = pl.DataFrame(s, schema=schema, strict=False)

    assert df["a"].to_list() == [None, 0, 1]
    assert df["a"].dtype == pl.UInt8

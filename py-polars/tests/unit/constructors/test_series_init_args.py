from __future__ import annotations

import pytest

import polars as pl


@pytest.fixture(scope="module")
def name() -> str:
    return "a"


@pytest.fixture(scope="module")
def values() -> list[int]:
    return [1, 2]


@pytest.fixture(scope="module")
def values_str() -> str:
    return "xyz"


@pytest.fixture(scope="module")
def dtype() -> pl.PolarsDataType:
    return pl.Int8


def test_series_init_args(
    name: str, values: list[int], values_str: str, dtype: pl.PolarsDataType
) -> None:
    s = pl.Series(values, name)
    assert s.name == name
    assert s.to_list() == values

    s = pl.Series(values, name=name)
    assert s.name == name
    assert s.to_list() == values

    s = pl.Series(values_str, name=name)
    assert s.name == name
    assert s.to_list() == list(values_str)

    s = pl.Series(None, name=name)
    assert s.name == name
    assert s.to_list() == []

    s = pl.Series(values=values)
    assert s.name == ""
    assert s.to_list() == values

    s = pl.Series(values, name, dtype)
    assert s.name == name
    assert s.to_list() == values
    assert s.dtype == dtype

    s = pl.Series(None)
    assert s.name == ""
    assert s.to_list() == []

    s = pl.Series(None, None)
    assert s.name == ""
    assert s.to_list() == []

    s = pl.Series(name, name)
    assert s.name == name
    assert s.to_list() == [name]


def test_series_init_args_deprecated(
    name: str, values: list[int], values_str: str, dtype: pl.PolarsDataType
) -> None:
    s = pl.Series(name)
    assert s.name == name
    assert s.to_list() == []

    s = pl.Series(name, values)
    assert s.name == name
    assert s.to_list() == values

    s = pl.Series(name, values=values)
    assert s.name == name
    assert s.to_list() == values

    s = pl.Series(name, values_str)
    assert s.name == name
    assert s.to_list() == list(values_str)

    s = pl.Series(None, values)
    assert s.name == ""
    assert s.to_list() == values

    s = pl.Series(None, values=values)
    assert s.name == ""
    assert s.to_list() == values

    s = pl.Series(name, values=name)
    assert s.name == name
    assert s.to_list() == [name]


def test_series_init_args_raises(
    name: str, values: list[int], values_str: str, dtype: pl.PolarsDataType
) -> None:
    with pytest.raises(TypeError):
        pl.Series(name, values, dtype, False)
    with pytest.raises(TypeError):
        pl.Series(values, values=values)
    with pytest.raises(TypeError):
        pl.Series(values, name, values=values)
    with pytest.raises(TypeError):
        pl.Series(values, name, name=name)
    with pytest.raises(TypeError):
        pl.Series(name, values, dtype, dtype=dtype)

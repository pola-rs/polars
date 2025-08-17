from __future__ import annotations

import enum
from datetime import date, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    NamedTuple,
    Optional,
    Union,
)

import pytest

import polars as pl
from polars.datatypes._parse import (
    _parse_forward_ref_into_dtype,
    _parse_generic_into_dtype,
    _parse_union_type_into_dtype,
    parse_into_dtype,
    parse_py_type_into_dtype,
)

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def assert_dtype_equal(left: PolarsDataType, right: PolarsDataType) -> None:
    assert left == right
    assert type(left) is type(right)
    assert hash(left) == hash(right)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (pl.Int8(), pl.Int8()),
        (list, pl.List),
    ],
)
def test_parse_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = parse_into_dtype(input)
    assert_dtype_equal(result, expected)


def test_parse_into_dtype_enum_19724() -> None:
    class PythonEnum(str, enum.Enum):
        CAT1 = "A"
        CAT2 = "B"
        CAT3 = "C"

    result = parse_into_dtype(PythonEnum)
    expected = pl.Enum(["A", "B", "C"])
    assert_dtype_equal(result, expected)


def test_parse_into_dtype_enum_ints_19724() -> None:
    class PythonEnum(int, enum.Enum):
        CAT1 = 1
        CAT2 = 2
        CAT3 = 3

    with pytest.raises(TypeError, match="Enum categories must be strings"):
        parse_into_dtype(PythonEnum)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (datetime, pl.Datetime("us")),
        (date, pl.Date()),
        (type(None), pl.Null()),
        (object, pl.Object()),
    ],
)
def test_parse_py_type_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = parse_py_type_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (list[int], pl.List(pl.Int64())),
        (tuple[str, ...], pl.List(pl.String())),
        (tuple[datetime, datetime], pl.List(pl.Datetime("us"))),
    ],
)
def test_parse_generic_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = _parse_generic_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize(
    "input",
    [
        dict[str, float],
        tuple[int, str],
        tuple[int, float, float],
    ],
)
def test_parse_generic_into_dtype_invalid(input: Any) -> None:
    with pytest.raises(TypeError):
        _parse_generic_into_dtype(input)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (ForwardRef("date"), pl.Date()),
        (ForwardRef("int | None"), pl.Int64()),
        (ForwardRef("None | float"), pl.Float64()),
    ],
)
def test_parse_forward_ref_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = _parse_forward_ref_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (Optional[int], pl.Int64()),
        (Optional[pl.String], pl.String),
        (Union[float, None], pl.Float64()),
    ],
)
def test_parse_union_type_into_dtype(input: Any, expected: PolarsDataType) -> None:
    result = _parse_union_type_into_dtype(input)
    assert_dtype_equal(result, expected)


@pytest.mark.parametrize(
    "input",
    [
        Union[int, float],
        Optional[Union[int, str]],
    ],
)
def test_parse_union_type_into_dtype_invalid(input: Any) -> None:
    with pytest.raises(TypeError):
        _parse_union_type_into_dtype(input)


def test_parse_dtype_namedtuple_fields() -> None:
    # Utilizes ForwardRef parsing

    class MyTuple(NamedTuple):
        a: str
        b: int
        c: str | None = None

    schema = {c: parse_into_dtype(a) for c, a in MyTuple.__annotations__.items()}

    expected = pl.Schema({"a": pl.String(), "b": pl.Int64(), "c": pl.String()})
    assert schema == expected

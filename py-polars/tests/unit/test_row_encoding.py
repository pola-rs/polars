from __future__ import annotations

from decimal import Decimal as D
from typing import TYPE_CHECKING

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import dataframes, series
from polars.testing.parametric.strategies.dtype import dtypes
from tests.unit.conftest import FLOAT_DTYPES, INTEGER_DTYPES

if TYPE_CHECKING:
    from typing import Any

    from polars._typing import PolarsDataType

FIELD_COMBS = [
    (descending, nulls_last, False)
    for descending in [False, True]
    for nulls_last in [False, True]
] + [(False, False, True)]


def roundtrip_re(
    df: pl.DataFrame, fields: list[tuple[bool, bool, bool]] | None = None
) -> None:
    if fields is None:
        fields = [(False, False, False)] * df.width

    row_encoded = df._row_encode(fields)
    if any(f[2] for f in fields):
        return

    dtypes = [(c, df.get_column(c).dtype) for c in df.columns]
    result = row_encoded._row_decode(dtypes, fields)

    assert_frame_equal(df, result)


def roundtrip_series_re(
    values: pl.series.series.ArrayLike,
    dtype: PolarsDataType,
    field: tuple[bool, bool, bool],
) -> None:
    roundtrip_re(pl.Series("series", values, dtype).to_frame(), [field])


@given(
    df=dataframes(
        excluded_dtypes=[
            pl.Categorical,
            pl.Decimal,  # Bug: see https://github.com/pola-rs/polars/issues/20308
        ]
    )
)
@pytest.mark.parametrize("field", FIELD_COMBS)
def test_row_encoding_parametric(
    df: pl.DataFrame, field: tuple[bool, bool, bool]
) -> None:
    roundtrip_re(df, [field] * df.width)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_nulls(field: tuple[bool, bool, bool]) -> None:
    roundtrip_series_re([], pl.Null, field)
    roundtrip_series_re([None], pl.Null, field)
    roundtrip_series_re([None] * 2, pl.Null, field)
    roundtrip_series_re([None] * 13, pl.Null, field)
    roundtrip_series_re([None] * 42, pl.Null, field)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_bool(field: tuple[bool, bool, bool]) -> None:
    roundtrip_series_re([], pl.Boolean, field)
    roundtrip_series_re([False], pl.Boolean, field)
    roundtrip_series_re([True], pl.Boolean, field)
    roundtrip_series_re([False, True], pl.Boolean, field)
    roundtrip_series_re([True, False], pl.Boolean, field)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("field", FIELD_COMBS)
def test_int(dtype: pl.DataType, field: tuple[bool, bool, bool]) -> None:
    min = pl.select(x=dtype.min()).item()  # type: ignore[attr-defined]
    max = pl.select(x=dtype.max()).item()  # type: ignore[attr-defined]

    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([0], dtype, field)
    roundtrip_series_re([min], dtype, field)
    roundtrip_series_re([max], dtype, field)

    roundtrip_series_re([1, 2, 3], dtype, field)
    roundtrip_series_re([0, 1, 2, 3], dtype, field)
    roundtrip_series_re([min, 0, max], dtype, field)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("field", FIELD_COMBS)
def test_float(dtype: pl.DataType, field: tuple[bool, bool, bool]) -> None:
    inf = float("inf")
    inf_b = float("-inf")

    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([0.0], dtype, field)
    roundtrip_series_re([inf], dtype, field)
    roundtrip_series_re([-inf_b], dtype, field)

    roundtrip_series_re([1.0, 2.0, 3.0], dtype, field)
    roundtrip_series_re([0.0, 1.0, 2.0, 3.0], dtype, field)
    roundtrip_series_re([inf, 0, -inf_b], dtype, field)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_str(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.String
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([""], dtype, field)

    roundtrip_series_re(["a", "b", "c"], dtype, field)
    roundtrip_series_re(["", "a", "b", "c"], dtype, field)

    roundtrip_series_re(
        ["different", "length", "strings"],
        dtype,
        field,
    )
    roundtrip_series_re(
        ["different", "", "length", "", "strings"],
        dtype,
        field,
    )


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_struct(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.Struct({})
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re([{}], dtype, field)
    roundtrip_series_re([{}, {}, {}], dtype, field)
    roundtrip_series_re([{}, None, {}], dtype, field)

    dtype = pl.Struct({"x": pl.Int32})
    roundtrip_series_re([{"x": 1}], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re([{"x": 1}] * 3, dtype, field)
    roundtrip_series_re([{"x": 1}, {"x": None}, None], dtype, field)

    dtype = pl.Struct({"x": pl.Int32, "y": pl.Int32})
    roundtrip_series_re(
        [{"x": 1}, {"y": 2}],
        dtype,
        field,
    )
    roundtrip_series_re([None], dtype, field)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_list(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.List(pl.Int32)
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([[]], dtype, field)
    roundtrip_series_re([[1], [2]], dtype, field)
    roundtrip_series_re([[1, 2], [3]], dtype, field)
    roundtrip_series_re([[1, 2], [], [3]], dtype, field)
    roundtrip_series_re([None, [1, 2], None, [], [3]], dtype, field)

    dtype = pl.List(pl.String)
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([[]], dtype, field)
    roundtrip_series_re([[""], [""]], dtype, field)
    roundtrip_series_re([["abc"], ["xyzw"]], dtype, field)
    roundtrip_series_re([["x", "yx"], ["abc"]], dtype, field)
    roundtrip_series_re([["wow", "this is"], [], ["cool"]], dtype, field)
    roundtrip_series_re(
        [None, ["very", "very"], None, [], ["cool"]],
        dtype,
        field,
    )


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_array(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.Array(pl.Int32, 0)
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([[]], dtype, field)
    roundtrip_series_re([None, [], None], dtype, field)
    roundtrip_series_re([None], dtype, field)

    dtype = pl.Array(pl.Int32, 2)
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([[5, 6]], dtype, field)
    roundtrip_series_re([[1, 2], [2, 3]], dtype, field)
    roundtrip_series_re([[1, 2], [3, 7]], dtype, field)
    roundtrip_series_re([[1, 2], [13, 11], [3, 7]], dtype, field)
    roundtrip_series_re(
        [None, [1, 2], None, [13, 11], [5, 7]],
        dtype,
        field,
    )

    dtype = pl.Array(pl.String, 2)
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([["a", "b"]], dtype, field)
    roundtrip_series_re([["", ""], ["", "a"]], dtype, field)
    roundtrip_series_re([["abc", "def"], ["ghi", "xyzw"]], dtype, field)
    roundtrip_series_re([["x", "yx"], ["abc", "xxx"]], dtype, field)
    roundtrip_series_re(
        [["wow", "this is"], ["soo", "so"], ["veryyy", "cool"]],
        dtype,
        field,
    )
    roundtrip_series_re(
        [None, ["very", "very"], None, [None, None], ["verryy", "cool"]],
        dtype,
        field,
    )


@pytest.mark.parametrize("field", FIELD_COMBS)
@pytest.mark.parametrize("precision", range(1, 38))
def test_decimal(field: tuple[bool, bool, bool], precision: int) -> None:
    dtype = pl.Decimal(precision=precision, scale=0)
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re([D("1")], dtype, field)
    roundtrip_series_re([D("-1")], dtype, field)
    roundtrip_series_re([D("9" * precision)], dtype, field)
    roundtrip_series_re([D("-" + "9" * precision)], dtype, field)
    roundtrip_series_re([None, D("-1"), None], dtype, field)
    roundtrip_series_re([D("-1"), D("0"), D("1")], dtype, field)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_enum(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.Enum([])

    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re([None, None], dtype, field)

    dtype = pl.Enum(["a", "x", "b"])

    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re(["a"], dtype, field)
    roundtrip_series_re(["x"], dtype, field)
    roundtrip_series_re(["b"], dtype, field)
    roundtrip_series_re(["b", "x", "a"], dtype, field)
    roundtrip_series_re([None, "b", None], dtype, field)
    roundtrip_series_re([None, "a", None], dtype, field)


@pytest.mark.parametrize("size", [127, 128, 255, 256, 2**15, 2**15 + 1])
@pytest.mark.parametrize("field", FIELD_COMBS)
@pytest.mark.slow
def test_large_enum(size: int, field: tuple[bool, bool, bool]) -> None:
    dtype = pl.Enum([str(i) for i in range(size)])
    roundtrip_series_re([None, "1"], dtype, field)
    roundtrip_series_re(["1", None], dtype, field)

    roundtrip_series_re(
        [str(i) for i in range(3, size, int(7 * size / (2**8)))], dtype, field
    )


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_list_arr(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.List(pl.Array(pl.String, 2))
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re([[None]], dtype, field)
    roundtrip_series_re([[[None, None]]], dtype, field)
    roundtrip_series_re([[["a", "b"]]], dtype, field)
    roundtrip_series_re([[["a", "b"], ["xyz", "wowie"]]], dtype, field)
    roundtrip_series_re([[["a", "b"]], None, [None, None]], dtype, field)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_list_struct_arr(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.List(
        pl.Struct({"x": pl.Array(pl.String, 2), "y": pl.Array(pl.Int64, 3)})
    )
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re([[None]], dtype, field)
    roundtrip_series_re([[{"x": None, "y": None}]], dtype, field)
    roundtrip_series_re([[{"x": ["a", None], "y": [1, None, 3]}]], dtype, field)
    roundtrip_series_re([[{"x": ["a", "xyz"], "y": [1, 7, 3]}]], dtype, field)
    roundtrip_series_re([[{"x": ["a", "xyz"], "y": [1, 7, 3]}], []], dtype, field)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_list_nulls(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.List(pl.Null)
    roundtrip_series_re([], dtype, field)
    roundtrip_series_re([[]], dtype, field)
    roundtrip_series_re([None], dtype, field)
    roundtrip_series_re([[None]], dtype, field)
    roundtrip_series_re([[None, None, None]], dtype, field)
    roundtrip_series_re([[None], [None, None], [None, None, None]], dtype, field)


@pytest.mark.parametrize("field", FIELD_COMBS)
def test_masked_out_list_20151(field: tuple[bool, bool, bool]) -> None:
    dtype = pl.List(pl.Int64())

    values = [[1, 2], None, [4, 5], [None, 3]]

    array_series = pl.Series(values, dtype=pl.Array(pl.Int64(), 2))
    list_from_array_series = array_series.cast(dtype)

    roundtrip_series_re(list_from_array_series, dtype, field)


def test_int_after_null() -> None:
    roundtrip_re(
        pl.DataFrame(
            [
                pl.Series("a", [None], pl.Null),
                pl.Series("b", [None], pl.Int8),
            ]
        ),
        [(False, True, False), (False, True, False)],
    )


@pytest.mark.parametrize("field", FIELD_COMBS)
@given(s=series(allow_null=False, allow_chunks=False, excluded_dtypes=[pl.Categorical]))
def test_optional_eq_non_optional_20320(
    field: tuple[bool, bool, bool], s: pl.Series
) -> None:
    with_null = s.extend(pl.Series([None], dtype=s.dtype))

    re_without_null = s.to_frame()._row_encode([field])
    re_with_null = with_null.to_frame()._row_encode([field])

    re_without_null = re_without_null.cast(pl.Binary)
    re_with_null = re_with_null.cast(pl.Binary)

    assert_series_equal(re_with_null.head(s.len()), re_without_null)


@pytest.mark.parametrize("field", FIELD_COMBS)
@given(dtype=dtypes(excluded_dtypes=[pl.Categorical]))
def test_null(
    field: tuple[bool, bool, bool],
    dtype: pl.DataType,
) -> None:
    s = pl.Series("a", [None], dtype)

    assert_series_equal(
        s.to_frame()
        ._row_encode([field])
        ._row_decode([("a", dtype)], [field])
        .to_series(),
        s,
    )


@pytest.mark.parametrize(
    ("dtype", "vs"),
    [
        (pl.List(pl.String), [[None], ["A"], ["B"]]),
        (pl.Array(pl.String, 1), [[None], ["A"], ["B"]]),
        (pl.Struct({"x": pl.String}), [{"x": None}, {"x": "A"}, {"x": "B"}]),
        (pl.Array(pl.String, 2), [[None, "Z"], ["A", "C"], ["B", "B"]]),
    ],
)
def test_nested_sorting_22557(dtype: pl.DataType, vs: list[Any]) -> None:
    s = pl.Series("a", [vs[1], None, vs[0], vs[2]], dtype)

    assert_series_equal(
        s.sort(descending=False, nulls_last=False), pl.Series("a", [None] + vs, dtype)
    )
    assert_series_equal(
        s.sort(descending=False, nulls_last=True), pl.Series("a", vs + [None], dtype)
    )
    assert_series_equal(
        s.sort(descending=True, nulls_last=False),
        pl.Series("a", [None] + vs[::-1], dtype),
    )
    assert_series_equal(
        s.sort(descending=True, nulls_last=True),
        pl.Series("a", vs[::-1] + [None], dtype),
    )

    roundtrip_series_re(vs, dtype, (False, False, False))
    roundtrip_series_re(vs, dtype, (False, True, False))
    roundtrip_series_re(vs, dtype, (True, False, False))
    roundtrip_series_re(vs, dtype, (True, True, False))

    assert_series_equal(
        s.to_frame()._row_encode([(False, False, False)]).arg_sort(),
        pl.Series("a", [1, 2, 0, 3], pl.get_index_type()),
        check_names=False,
    )
    assert_series_equal(
        s.to_frame()._row_encode([(False, True, False)]).arg_sort(),
        pl.Series("a", [2, 0, 3, 1], pl.get_index_type()),
        check_names=False,
    )
    assert_series_equal(
        s.to_frame()._row_encode([(True, False, False)]).arg_sort(),
        pl.Series("a", [1, 3, 0, 2], pl.get_index_type()),
        check_names=False,
    )
    assert_series_equal(
        s.to_frame()._row_encode([(True, True, False)]).arg_sort(),
        pl.Series("a", [3, 0, 2, 1], pl.get_index_type()),
        check_names=False,
    )

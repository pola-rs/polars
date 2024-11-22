from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import column, dataframes

if TYPE_CHECKING:
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
            pl.Enum,
            pl.Decimal,
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


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ],
)
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


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Float32,
        pl.Float64,
    ],
)
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


def assert_order_dataframe(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    order: list[Literal["lt", "eq", "gt"]],
    *,
    descending: bool = False,
    nulls_last: bool = False,
) -> None:
    field = (descending, nulls_last, False)
    l_re = lhs._row_encode([field] * lhs.width).cast(pl.Binary)
    r_re = rhs._row_encode([field] * rhs.width).cast(pl.Binary)

    l_lt_r_s = "gt" if descending else "lt"
    l_gt_r_s = "lt" if descending else "gt"

    assert_series_equal(
        l_re < r_re, pl.Series([o == l_lt_r_s for o in order]), check_names=False
    )
    assert_series_equal(
        l_re == r_re, pl.Series([o == "eq" for o in order]), check_names=False
    )
    assert_series_equal(
        l_re > r_re, pl.Series([o == l_gt_r_s for o in order]), check_names=False
    )


def assert_order_series(
    lhs: pl.series.series.ArrayLike,
    rhs: pl.series.series.ArrayLike,
    dtype: pl._typing.PolarsDataType,
    order: list[Literal["lt", "eq", "gt"]],
    *,
    descending: bool = False,
    nulls_last: bool = False,
) -> None:
    lhs = pl.Series("lhs", lhs, dtype).to_frame()
    rhs = pl.Series("rhs", rhs, dtype).to_frame()
    assert_order_dataframe(
        lhs, rhs, order, descending=descending, nulls_last=nulls_last
    )


def parametric_order_base(df: pl.DataFrame) -> None:
    lhs = df.get_columns()[0]
    rhs = df.get_columns()[1]

    field = (False, False, False)
    lhs_re = lhs.to_frame()._row_encode([field]).cast(pl.Binary)
    rhs_re = rhs.to_frame()._row_encode([field]).cast(pl.Binary)

    assert_series_equal(lhs < rhs, lhs_re < rhs_re, check_names=False)
    assert_series_equal(lhs == rhs, lhs_re == rhs_re, check_names=False)
    assert_series_equal(lhs > rhs, lhs_re > rhs_re, check_names=False)

    field = (True, False, False)
    lhs_re = lhs.to_frame()._row_encode([field]).cast(pl.Binary)
    rhs_re = rhs.to_frame()._row_encode([field]).cast(pl.Binary)

    assert_series_equal(lhs > rhs, lhs_re < rhs_re, check_names=False)
    assert_series_equal(lhs == rhs, lhs_re == rhs_re, check_names=False)
    assert_series_equal(lhs < rhs, lhs_re > rhs_re, check_names=False)


@given(
    df=dataframes([column(dtype=pl.Int32), column(dtype=pl.Int32)], allow_null=False)
)
def test_parametric_int_order(df: pl.DataFrame) -> None:
    parametric_order_base(df)


@given(
    df=dataframes([column(dtype=pl.UInt32), column(dtype=pl.UInt32)], allow_null=False)
)
def test_parametric_uint_order(df: pl.DataFrame) -> None:
    parametric_order_base(df)


@given(
    df=dataframes([column(dtype=pl.String), column(dtype=pl.String)], allow_null=False)
)
def test_parametric_string_order(df: pl.DataFrame) -> None:
    parametric_order_base(df)


@given(
    df=dataframes([column(dtype=pl.Binary), column(dtype=pl.Binary)], allow_null=False)
)
def test_parametric_binary_order(df: pl.DataFrame) -> None:
    parametric_order_base(df)


def test_order_bool() -> None:
    dtype = pl.Boolean
    assert_order_series(
        [None, False, True], [True, False, None], dtype, ["lt", "eq", "gt"]
    )
    assert_order_series(
        [None, False, True],
        [True, False, None],
        dtype,
        ["gt", "eq", "lt"],
        nulls_last=True,
    )

    assert_order_series(
        [False, False, True, True],
        [True, False, True, False],
        dtype,
        ["lt", "eq", "eq", "gt"],
    )
    assert_order_series(
        [False, False, True, True],
        [True, False, True, False],
        dtype,
        ["lt", "eq", "eq", "gt"],
        descending=True,
    )


def test_order_int() -> None:
    dtype = pl.Int32
    assert_order_series([1, 2, 3], [3, 2, 1], dtype, ["lt", "eq", "gt"])
    assert_order_series([-1, 0, 1], [1, 0, -1], dtype, ["lt", "eq", "gt"])
    assert_order_series([None], [None], dtype, ["eq"])
    assert_order_series([None], [1], dtype, ["lt"])
    assert_order_series([None], [1], dtype, ["gt"], nulls_last=True)


def test_order_uint() -> None:
    dtype = pl.UInt32
    assert_order_series([1, 2, 3], [3, 2, 1], dtype, ["lt", "eq", "gt"])
    assert_order_series([None], [None], dtype, ["eq"])
    assert_order_series([None], [1], dtype, ["lt"])
    assert_order_series([None], [1], dtype, ["gt"], nulls_last=True)


def test_order_list() -> None:
    dtype = pl.List(pl.Int32)
    assert_order_series([[1, 2, 3]], [[3, 2, 1]], dtype, ["lt"])
    assert_order_series([[-1, 0, 1]], [[1, 0, -1]], dtype, ["lt"])
    assert_order_series([None], [None], dtype, ["eq"])
    assert_order_series([None], [[1, 2, 3]], dtype, ["lt"])
    assert_order_series([None], [[1, 2, 3]], dtype, ["gt"], nulls_last=True)
    assert_order_series([[None, 2, 3]], [[None, 2, 1]], dtype, ["gt"])


def test_order_array() -> None:
    dtype = pl.Array(pl.Int32, 3)
    assert_order_series([[1, 2, 3]], [[3, 2, 1]], dtype, ["lt"])
    assert_order_series([[-1, 0, 1]], [[1, 0, -1]], dtype, ["lt"])
    assert_order_series([None], [None], dtype, ["eq"])
    assert_order_series([None], [[1, 2, 3]], dtype, ["lt"])
    assert_order_series([None], [[1, 2, 3]], dtype, ["gt"], nulls_last=True)
    assert_order_series([[None, 2, 3]], [[None, 2, 1]], dtype, ["gt"])


def test_order_masked_array() -> None:
    dtype = pl.Array(pl.Int32, 3)
    lhs = pl.Series("l", [1, 2, 3], pl.Int32).replace(1, None).reshape((1, 3))
    rhs = pl.Series("r", [3, 2, 1], pl.Int32).replace(3, None).reshape((1, 3))
    assert_order_series(lhs, rhs, dtype, ["gt"])


def test_order_masked_struct() -> None:
    dtype = pl.Array(pl.Int32, 3)
    lhs = pl.Series("l", [1, 2, 3], pl.Int32).replace(1, None).reshape((1, 3))
    rhs = pl.Series("r", [3, 2, 1], pl.Int32).replace(3, None).reshape((1, 3))
    assert_order_series(
        lhs.to_frame().to_struct(), rhs.to_frame().to_struct(), dtype, ["gt"]
    )

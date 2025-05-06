# mypy: disable-error-code="valid-type"

from __future__ import annotations

import datetime
import decimal
import functools
from typing import Any, Literal, Optional, Union, cast

import pytest
from hypothesis import example, given

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import column, dataframes, series

Element = Optional[
    Union[
        bool,
        int,
        float,
        str,
        decimal.Decimal,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        list[Any],
        dict[Any, Any],
    ]
]
OrderSign = Literal[-1, 0, 1]


def elem_order_sign(
    lhs: Element, rhs: Element, *, descending: bool, nulls_last: bool
) -> OrderSign:
    if isinstance(lhs, pl.Series) and isinstance(rhs, pl.Series):
        assert lhs.dtype == rhs.dtype

        if isinstance(lhs.dtype, pl.Enum) or lhs.dtype == pl.Categorical(
            ordering="physical"
        ):
            lhs = cast(Element, lhs.to_physical())
            rhs = cast(Element, rhs.to_physical())
            assert isinstance(lhs, pl.Series)
            assert isinstance(rhs, pl.Series)

        if lhs.dtype == pl.Categorical(ordering="lexical"):
            lhs = cast(Element, lhs.cast(pl.String))
            rhs = cast(Element, rhs.cast(pl.String))
            assert isinstance(lhs, pl.Series)
            assert isinstance(rhs, pl.Series)

        if lhs.is_null().equals(rhs.is_null()) and lhs.equals(rhs):
            return 0

        lhs = lhs.to_list()
        rhs = rhs.to_list()

    if lhs is None and rhs is None:
        return 0
    elif lhs is None:
        return 1 if nulls_last else -1
    elif rhs is None:
        return -1 if nulls_last else 1
    elif lhs == rhs:
        return 0
    elif isinstance(lhs, bool) and isinstance(rhs, bool):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, datetime.date) and isinstance(rhs, datetime.date):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, datetime.datetime) and isinstance(rhs, datetime.datetime):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, datetime.time) and isinstance(rhs, datetime.time):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, datetime.timedelta) and isinstance(rhs, datetime.timedelta):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, decimal.Decimal) and isinstance(rhs, decimal.Decimal):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, int) and isinstance(rhs, int):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, float) and isinstance(rhs, float):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, bytes) and isinstance(rhs, bytes):
        lhs_b: bytes = lhs
        rhs_b: bytes = rhs

        for lh, rh in zip(lhs_b, rhs_b):
            o = elem_order_sign(lh, rh, descending=descending, nulls_last=nulls_last)
            if o != 0:
                return o

        if len(lhs_b) == len(rhs_b):
            return 0
        else:
            return -1 if (len(lhs_b) < len(rhs_b)) ^ descending else 1
    elif isinstance(lhs, str) and isinstance(rhs, str):
        return -1 if (lhs < rhs) ^ descending else 1
    elif isinstance(lhs, list) and isinstance(rhs, list):
        for lh, rh in zip(lhs, rhs):
            # Nulls lasts is set to descending for nested values. See #22557.
            o = elem_order_sign(lh, rh, descending=descending, nulls_last=descending)
            if o != 0:
                return o

        if len(lhs) == len(rhs):
            return 0
        else:
            return -1 if (len(lhs) < len(rhs)) ^ descending else 1
    elif isinstance(lhs, dict) and isinstance(rhs, dict):
        assert len(lhs) == len(rhs)

        for lh, rh in zip(lhs.values(), rhs.values()):
            # Nulls lasts is set to descending for nested values. See #22557.
            o = elem_order_sign(lh, rh, descending=descending, nulls_last=descending)
            if o != 0:
                return o

        return 0
    else:
        pytest.fail("type mismatch")


def tuple_order(
    lhs: tuple[Element, ...],
    rhs: tuple[Element, ...],
    *,
    descending: list[bool],
    nulls_last: list[bool],
) -> OrderSign:
    assert len(lhs) == len(rhs)

    for lh, rh, dsc, nl in zip(lhs, rhs, descending, nulls_last):
        o = elem_order_sign(lh, rh, descending=dsc, nulls_last=nl)
        if o != 0:
            return o

    return 0


@given(
    s=series(
        excluded_dtypes=[
            pl.Float32,  # We cannot really deal with totalOrder
            pl.Float64,  # We cannot really deal with totalOrder
            pl.Decimal,  # Bug: see https://github.com/pola-rs/polars/issues/20308
            pl.Categorical,
        ],
        max_size=5,
    )
)
@example(s=pl.Series("col0", [None, [None]], pl.List(pl.Int64)))
@example(s=pl.Series("col0", [[None], [0]], pl.List(pl.Int64)))
def test_series_sort_parametric(s: pl.Series) -> None:
    for descending in [False, True]:
        for nulls_last in [False, True]:
            fields = [(descending, nulls_last, False)]

            def cmp(
                lhs: Element,
                rhs: Element,
                descending: bool = descending,
                nulls_last: bool = nulls_last,
            ) -> OrderSign:
                return elem_order_sign(
                    lhs, rhs, descending=descending, nulls_last=nulls_last
                )

            rows = list(s)
            rows.sort(key=functools.cmp_to_key(cmp))  # type: ignore[arg-type, unused-ignore]

            re = s.to_frame()._row_encode(fields)
            re_sorted = re.sort()
            re_decoded = re_sorted._row_decode([("s", s.dtype)], fields)

            assert_series_equal(
                pl.Series("s", rows, s.dtype), re_decoded.get_column("s")
            )


@given(
    df=dataframes(
        excluded_dtypes=[
            pl.Float32,  # We cannot really deal with totalOrder
            pl.Float64,  # We cannot really deal with totalOrder
            pl.Decimal,  # Bug: see https://github.com/pola-rs/polars/issues/20308
            pl.Enum,
            pl.Categorical,
        ],
        max_cols=3,
        max_size=5,
    )
)
@example(df=pl.DataFrame([pl.Series([{"x": 0}, {"x": None}])]))
def test_df_sort_parametric(df: pl.DataFrame) -> None:
    for i in range(4**df.width):
        descending = [((i // (4**j)) % 4) in [2, 3] for j in range(df.width)]
        nulls_last = [((i // (4**j)) % 4) in [1, 3] for j in range(df.width)]

        fields = [
            (descending, nulls_last, False)
            for (descending, nulls_last) in zip(descending, nulls_last)
        ]

        def cmp(
            lhs: tuple[Element, ...],
            rhs: tuple[Element, ...],
            descending: list[bool] = descending,
            nulls_last: list[bool] = nulls_last,
        ) -> OrderSign:
            return tuple_order(lhs, rhs, descending=descending, nulls_last=nulls_last)

        rows = df.rows()
        rows.sort(key=functools.cmp_to_key(cmp))  # type: ignore[arg-type, unused-ignore]

        re = df._row_encode(fields)
        re_sorted = re.sort()
        re_decoded = re_sorted._row_decode(df.schema.items(), fields)

        assert_frame_equal(pl.DataFrame(rows, df.schema, orient="row"), re_decoded)


def assert_order_series(
    lhs: pl.series.series.ArrayLike,
    rhs: pl.series.series.ArrayLike,
    dtype: pl._typing.PolarsDataType,
) -> None:
    lhs_df = pl.Series("lhs", lhs, dtype).to_frame()
    rhs_df = pl.Series("rhs", rhs, dtype).to_frame()

    for descending in [False, True]:
        for nulls_last in [False, True]:
            field = (descending, nulls_last, False)
            l_re = lhs_df._row_encode([field]).cast(pl.Binary)
            r_re = rhs_df._row_encode([field]).cast(pl.Binary)

            order = [
                elem_order_sign(
                    lh[0], rh[0], descending=descending, nulls_last=nulls_last
                )
                for (lh, rh) in zip(lhs_df.rows(), rhs_df.rows())
            ]

            assert_series_equal(
                l_re < r_re, pl.Series([o == -1 for o in order]), check_names=False
            )
            assert_series_equal(
                l_re == r_re, pl.Series([o == 0 for o in order]), check_names=False
            )
            assert_series_equal(
                l_re > r_re, pl.Series([o == 1 for o in order]), check_names=False
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
    assert_order_series([None, False, True], [True, False, None], dtype)
    assert_order_series(
        [None, False, True],
        [True, False, None],
        dtype,
    )

    assert_order_series(
        [False, False, True, True],
        [True, False, True, False],
        dtype,
    )
    assert_order_series(
        [False, False, True, True],
        [True, False, True, False],
        dtype,
    )


def test_order_int() -> None:
    dtype = pl.Int32
    assert_order_series([1, 2, 3], [3, 2, 1], dtype)
    assert_order_series([-1, 0, 1], [1, 0, -1], dtype)
    assert_order_series([None], [None], dtype)
    assert_order_series([None], [1], dtype)


def test_order_uint() -> None:
    dtype = pl.UInt32
    assert_order_series([1, 2, 3], [3, 2, 1], dtype)
    assert_order_series([None], [None], dtype)
    assert_order_series([None], [1], dtype)


def test_order_str() -> None:
    dtype = pl.String
    assert_order_series(["a", "b", "c"], ["c", "b", "a"], dtype)
    assert_order_series(["a", "aa", "aaa"], ["aaa", "aa", "a"], dtype)
    assert_order_series(["", "a", "aa"], ["aa", "a", ""], dtype)
    assert_order_series([None], [None], dtype)
    assert_order_series([None], ["a"], dtype)


def test_order_bin() -> None:
    dtype = pl.Binary
    assert_order_series([b"a", b"b", b"c"], [b"c", b"b", b"a"], dtype)
    assert_order_series([b"a", b"aa", b"aaa"], [b"aaa", b"aa", b"a"], dtype)
    assert_order_series([b"", b"a", b"aa"], [b"aa", b"a", b""], dtype)
    assert_order_series([None], [None], dtype)
    assert_order_series([None], [b"a"], dtype)
    assert_order_series([None], [b"a"], dtype)


def test_order_list() -> None:
    dtype = pl.List(pl.Int32)
    assert_order_series([[1, 2, 3]], [[3, 2, 1]], dtype)
    assert_order_series([[-1, 0, 1]], [[1, 0, -1]], dtype)
    assert_order_series([None], [None], dtype)
    assert_order_series([None], [[1, 2, 3]], dtype)
    assert_order_series([[None, 2, 3]], [[None, 2, 1]], dtype)

    assert_order_series([[]], [[None]], dtype)
    assert_order_series([[]], [[1]], dtype)
    assert_order_series([[1]], [[1, 2]], dtype)


def test_order_array() -> None:
    dtype = pl.Array(pl.Int32, 3)
    assert_order_series([[1, 2, 3]], [[3, 2, 1]], dtype)
    assert_order_series([[-1, 0, 1]], [[1, 0, -1]], dtype)
    assert_order_series([None], [None], dtype)
    assert_order_series([None], [[1, 2, 3]], dtype)
    assert_order_series([[None, 2, 3]], [[None, 2, 1]], dtype)


def test_order_masked_array() -> None:
    dtype = pl.Array(pl.Int32, 3)
    lhs = pl.Series("l", [1, 2, 3], pl.Int32).replace(1, None).reshape((1, 3))
    rhs = pl.Series("r", [3, 2, 1], pl.Int32).replace(3, None).reshape((1, 3))
    assert_order_series(lhs, rhs, dtype)


def test_order_masked_struct() -> None:
    dtype = pl.Array(pl.Int32, 3)
    lhs = pl.Series("l", [1, 2, 3], pl.Int32).replace(1, None).reshape((1, 3))
    rhs = pl.Series("r", [3, 2, 1], pl.Int32).replace(3, None).reshape((1, 3))
    assert_order_series(lhs.to_frame().to_struct(), rhs.to_frame().to_struct(), dtype)


def test_order_enum() -> None:
    dtype = pl.Enum(["a", "b", "c"])

    assert_order_series(["a", "b", "c"], ["c", "b", "a"], dtype)
    assert_order_series([None], [None], dtype)
    assert_order_series([None], ["a"], dtype)

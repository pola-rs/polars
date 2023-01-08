from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Iterator, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars.datatypes import (
    Date,
    Datetime,
    Field,
    Float64,
    Int32,
    Int64,
    PolarsDataType,
    Time,
    UInt32,
    UInt64,
)
from polars.exceptions import ShapeError
from polars.internals.construction import iterable_to_pyseries
from polars.internals.type_aliases import EpochTimeUnit
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing._private import verify_series_and_expr_api

if TYPE_CHECKING:
    from polars.internals.type_aliases import TimeUnit


def test_cum_agg() -> None:
    # confirm that known series give expected results
    s = pl.Series("a", [1, 2, 3, 2])
    verify_series_and_expr_api(s, pl.Series("a", [1, 3, 6, 8]), "cumsum")
    verify_series_and_expr_api(s, pl.Series("a", [1, 1, 1, 1]), "cummin")
    verify_series_and_expr_api(s, pl.Series("a", [1, 2, 3, 3]), "cummax")
    verify_series_and_expr_api(s, pl.Series("a", [1, 2, 6, 12]), "cumprod")


def test_init_inputs(monkeypatch: Any) -> None:
    nan = float("nan")
    for flag in [False, True]:
        monkeypatch.setattr(pl.internals.construction, "_PYARROW_AVAILABLE", flag)
        # Good inputs
        pl.Series("a", [1, 2])
        pl.Series("a", values=[1, 2])
        pl.Series(name="a", values=[1, 2])
        pl.Series(values=[1, 2], name="a")

        assert pl.Series([1, 2]).dtype == pl.Int64
        assert pl.Series(values=[1, 2]).dtype == pl.Int64
        assert pl.Series("a").dtype == pl.Float32  # f32 type used in case of no data
        assert pl.Series().dtype == pl.Float32
        assert pl.Series([]).dtype == pl.Float32
        assert pl.Series(dtype_if_empty=pl.Utf8).dtype == pl.Utf8
        assert pl.Series([], dtype_if_empty=pl.UInt16).dtype == pl.UInt16
        # "== []" will be cast to empty Series with Utf8 dtype.
        assert_series_equal(
            pl.Series([], dtype_if_empty=pl.Utf8) == [], pl.Series("", dtype=pl.Boolean)
        )
        assert pl.Series(values=[True, False]).dtype == pl.Boolean
        assert pl.Series(values=np.array([True, False])).dtype == pl.Boolean
        assert pl.Series(values=np.array(["foo", "bar"])).dtype == pl.Utf8
        assert pl.Series(values=["foo", "bar"]).dtype == pl.Utf8
        assert (
            pl.Series("a", [pl.Series([1, 2, 4]), pl.Series([3, 2, 1])]).dtype
            == pl.List
        )
        assert pl.Series("a", [10000, 20000, 30000], dtype=pl.Time).dtype == pl.Time

        # 2d numpy array and/or list of 1d numpy arrays
        for res in (
            pl.Series(
                name="a",
                values=np.array([[1, 2], [3, nan]], dtype=np.float32),
                nan_to_null=True,
            ),
            pl.Series(
                name="a",
                values=[
                    np.array([1, 2], dtype=np.float32),
                    np.array([3, nan], dtype=np.float32),
                ],
                nan_to_null=True,
            ),
            pl.Series(
                name="a",
                values=(
                    np.ndarray((2,), np.float32, np.array([1, 2], dtype=np.float32)),
                    np.ndarray((2,), np.float32, np.array([3, nan], dtype=np.float32)),
                ),
                nan_to_null=True,
            ),
        ):
            assert res.dtype == pl.List(pl.Float32)
            assert res[0].to_list() == [1.0, 2.0]
            assert res[1].to_list() == [3.0, None]

        assert pl.Series(
            values=np.array([["foo", "bar"], ["foo2", "bar2"]])
        ).dtype == pl.List(pl.Utf8)

        # lists
        assert pl.Series("a", [[1, 2], [3, 4]]).dtype == pl.List(pl.Int64)

    # datetime64: check timeunit (auto-detect, implicit/explicit) and NaT
    d64 = pd.date_range(date(2021, 8, 1), date(2021, 8, 3)).values
    d64[1] = None

    expected = [datetime(2021, 8, 1, 0), None, datetime(2021, 8, 3, 0)]
    for dtype in (None, Datetime, Datetime("ns")):
        s = pl.Series("dates", d64, dtype)
        assert s.to_list() == expected
        assert Datetime == s.dtype
        assert s.dtype.tu == "ns"  # type: ignore[attr-defined]

    s = pl.Series(values=d64.astype("<M8[ms]"))
    assert s.dtype.tu == "ms"  # type: ignore[attr-defined]
    assert expected == s.to_list()

    # pandas
    assert pl.Series(pd.Series([1, 2])).dtype == pl.Int64

    # Bad inputs
    with pytest.raises(ValueError):
        pl.Series([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        pl.Series({"a": [1, 2, 3]})
    with pytest.raises(OverflowError):
        pl.Series("bigint", [2**64])

    # numpy not available
    monkeypatch.setattr(pl.internals.series.series, "_check_for_numpy", lambda x: False)
    with pytest.raises(ValueError):
        pl.DataFrame(np.array([1, 2, 3]), columns=["a"])


def test_init_dataclass_namedtuple() -> None:
    from dataclasses import dataclass
    from typing import NamedTuple

    @dataclass
    class TeaShipmentDC:
        exporter: str
        importer: str
        product: str
        tonnes: int | None

    class TeaShipmentNT(NamedTuple):
        exporter: str
        importer: str
        product: str
        tonnes: None | int

    for Tea in (TeaShipmentDC, TeaShipmentNT):
        t0 = Tea(exporter="Sri Lanka", importer="USA", product="Ceylon", tonnes=10)
        t1 = Tea(exporter="India", importer="UK", product="Darjeeling", tonnes=25)

        s = pl.Series("t", [t0, t1])

        assert isinstance(s, pl.Series)
        assert s.dtype.fields == [  # type: ignore[attr-defined]
            Field("exporter", pl.Utf8),
            Field("importer", pl.Utf8),
            Field("product", pl.Utf8),
            Field("tonnes", pl.Int64),
        ]
        assert s.to_list() == [
            {
                "exporter": "Sri Lanka",
                "importer": "USA",
                "product": "Ceylon",
                "tonnes": 10,
            },
            {
                "exporter": "India",
                "importer": "UK",
                "product": "Darjeeling",
                "tonnes": 25,
            },
        ]
        assert_frame_equal(s.to_frame(), pl.DataFrame({"t": [t0, t1]}))


def test_concat() -> None:
    s = pl.Series("a", [2, 1, 3])

    assert pl.concat([s, s]).len() == 6
    # check if s remains unchanged
    assert s.len() == 3


def test_to_frame() -> None:
    s1 = pl.Series([1, 2])
    s2 = pl.Series("s", [1, 2])

    df1 = s1.to_frame()
    df2 = s2.to_frame()
    df3 = s1.to_frame("xyz")
    df4 = s2.to_frame("xyz")

    for df, name in ((df1, ""), (df2, "s"), (df3, "xyz"), (df4, "xyz")):
        assert isinstance(df, pl.DataFrame)
        assert df.rows() == [(1,), (2,)]
        assert df.columns == [name]

    # note: the empty string IS technically a valid column name
    assert s2.to_frame("").columns == [""]
    assert s2.name == "s"


def test_bitwise_ops() -> None:
    a = pl.Series([True, False, True])
    b = pl.Series([False, True, True])
    assert (a & b).series_equal(pl.Series([False, False, True]))
    assert (a | b).series_equal(pl.Series([True, True, True]))
    assert (a ^ b).series_equal(pl.Series([True, True, False]))
    assert (~a).series_equal(pl.Series([False, True, False]))

    # rand/rxor/ror we trigger by casting the left hand to a list here in the test
    # Note that the type annotations only allow Series to be passed in, but there is
    # specific code to deal with non-Series inputs.
    assert (True & a).series_equal(  # type: ignore[operator]
        pl.Series([True, False, True])
    )
    assert (True | a).series_equal(  # type: ignore[operator]
        pl.Series([True, True, True])
    )
    assert (True ^ a).series_equal(  # type: ignore[operator]
        pl.Series([False, True, False])
    )


def test_bitwise_floats_invert() -> None:
    a = pl.Series([2.0, 3.0, 0.0])
    assert ~a == NotImplemented


def test_equality() -> None:
    a = pl.Series("a", [1, 2])
    b = a

    cmp = a == b
    assert isinstance(cmp, pl.Series)
    assert cmp.sum() == 2
    assert (a != b).sum() == 0
    assert (a >= b).sum() == 2
    assert (a <= b).sum() == 2
    assert (a > b).sum() == 0
    assert (a < b).sum() == 0
    assert a.sum() == 3
    assert a.series_equal(b)

    a = pl.Series("name", ["ham", "foo", "bar"])
    assert_series_equal((a == "ham"), pl.Series("name", [True, False, False]))


def test_agg() -> None:
    series = pl.Series("a", [1, 2])
    assert series.mean() == 1.5
    assert series.min() == 1
    assert series.max() == 2


def test_date_agg() -> None:
    series = pl.Series(
        [
            date(2022, 8, 2),
            date(2096, 8, 1),
            date(9009, 9, 9),
        ],
        dtype=pl.Date,
    )
    assert series.min() == date(2022, 8, 2)
    assert series.max() == date(9009, 9, 9)


@pytest.mark.parametrize(
    "s", [pl.Series([1, 2], dtype=Int64), pl.Series([1, 2], dtype=Float64)]
)
def test_arithmetic(s: pl.Series) -> None:
    a = s
    b = s

    assert ((a * b) == [1, 4]).sum() == 2
    assert ((a / b) == [1.0, 1.0]).sum() == 2
    assert ((a + b) == [2, 4]).sum() == 2
    assert ((a - b) == [0, 0]).sum() == 2
    assert ((a + 1) == [2, 3]).sum() == 2
    assert ((a - 1) == [0, 1]).sum() == 2
    assert ((a / 1) == [1.0, 2.0]).sum() == 2
    assert ((a // 2) == [0, 1]).sum() == 2
    assert ((a * 2) == [2, 4]).sum() == 2
    assert ((2 + a) == [3, 4]).sum() == 2
    assert ((1 - a) == [0, -1]).sum() == 2
    assert ((2 * a) == [2, 4]).sum() == 2

    # integer division
    assert_series_equal(1 / a, pl.Series([1.0, 0.5]))
    if s.dtype == Int64:
        expected = pl.Series([1, 0])
    else:
        expected = pl.Series([1.0, 0.5])
    assert_series_equal(1 // a, expected)
    # modulo
    assert ((1 % a) == [0, 1]).sum() == 2
    assert ((a % 1) == [0, 0]).sum() == 2
    # negate
    assert (-a == [-1, -2]).sum() == 2
    # wrong dtypes in rhs operands
    assert ((1.0 - a) == [0.0, -1.0]).sum() == 2
    assert ((1.0 / a) == [1.0, 0.5]).sum() == 2
    assert ((1.0 * a) == [1, 2]).sum() == 2
    assert ((1.0 + a) == [2, 3]).sum() == 2
    assert ((1.0 % a) == [0, 1]).sum() == 2

    a = pl.Series("a", [datetime(2021, 1, 1)])
    with pytest.raises(ValueError):
        a // 2
    with pytest.raises(ValueError):
        a / 2
    with pytest.raises(ValueError):
        a * 2
    with pytest.raises(ValueError):
        a % 2
    with pytest.raises(ValueError):
        a**2
    with pytest.raises(ValueError):
        2 / a
    with pytest.raises(ValueError):
        2 // a
    with pytest.raises(ValueError):
        2 * a
    with pytest.raises(ValueError):
        2 % a
    with pytest.raises(ValueError):
        2**a


def test_arithmetic_empty() -> None:
    series = pl.Series("a", [])
    assert series.sum() == 0


def test_arithmetic_null() -> None:
    series = pl.Series("a", [None])
    assert series.sum() is None


def test_power() -> None:
    a = pl.Series([1, 2], dtype=Int64)
    b = pl.Series([None, 2.0], dtype=Float64)
    c = pl.Series([date(2020, 2, 28), date(2020, 3, 1)], dtype=Date)

    # pow
    assert_series_equal(a**2, pl.Series([1.0, 4.0], dtype=Float64))
    assert_series_equal(b**3, pl.Series([None, 8.0], dtype=Float64))
    assert_series_equal(a**a, pl.Series([1.0, 4.0], dtype=Float64))
    assert_series_equal(b**b, pl.Series([None, 4.0], dtype=Float64))
    assert_series_equal(a**b, pl.Series([None, 4.0], dtype=Float64))
    with pytest.raises(ValueError):
        c**2
    with pytest.raises(pl.ComputeError):
        a ** "hi"  # type: ignore[operator]

    # rpow
    assert_series_equal(2.0**a, pl.Series("literal", [2.0, 4.0], dtype=Float64))
    assert_series_equal(2**b, pl.Series("literal", [None, 4.0], dtype=Float64))
    with pytest.raises(ValueError):
        2**c
    with pytest.raises(pl.ComputeError):
        "hi" ** a


def test_add_string() -> None:
    s = pl.Series(["hello", "weird"])
    result = s + " world"
    assert_series_equal(result, pl.Series(["hello world", "weird world"]))

    result = "pfx:" + s
    assert_series_equal(result, pl.Series(["pfx:hello", "pfx:weird"]))


def test_append_extend() -> None:
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [8, 9, None])
    a.append(b, append_chunks=False)
    expected = pl.Series("a", [1, 2, 8, 9, None])
    assert a.series_equal(expected, null_equal=True)
    assert a.n_chunks() == 1


def test_various() -> None:
    a = pl.Series("a", [1, 2])
    assert a.is_null().sum() == 0
    assert a.name == "a"

    a.rename("b", in_place=True)
    assert a.name == "b"
    assert a.len() == 2
    assert len(a) == 2

    a.append(a.clone())
    assert a.series_equal(pl.Series("b", [1, 2, 1, 2]))

    a = pl.Series("a", range(20))
    assert a.head(5).len() == 5
    assert a.tail(5).len() == 5
    assert (a.head(5) != a.tail(5)).all()

    a = pl.Series("a", [2, 1, 4])
    a.sort(in_place=True)
    assert a.series_equal(pl.Series("a", [1, 2, 4]))
    a = pl.Series("a", [2, 1, 1, 4, 4, 4])
    assert_series_equal(a.arg_unique(), pl.Series("a", [0, 1, 3], dtype=UInt32))

    assert a.take([2, 3]).series_equal(pl.Series("a", [1, 4]))
    assert a.is_numeric()

    a = pl.Series("bool", [True, False])
    assert not a.is_numeric()


def test_filter_ops() -> None:
    a = pl.Series("a", range(20))
    with pytest.deprecated_call(match="passing a boolean mask to Series.__getitem__"):
        assert a[a > 1].len() == 18
        assert a[a < 1].len() == 1
        assert a[a <= 1].len() == 2
        assert a[a >= 1].len() == 19
        assert a[a == 1].len() == 1
        assert a[a != 1].len() == 19


def test_cast() -> None:
    a = pl.Series("a", range(20))

    assert a.cast(pl.Float32).dtype == pl.Float32
    assert a.cast(pl.Float64).dtype == pl.Float64
    assert a.cast(pl.Int32).dtype == pl.Int32
    assert a.cast(pl.UInt32).dtype == pl.UInt32
    assert a.cast(pl.Datetime).dtype == pl.Datetime
    assert a.cast(pl.Date).dtype == pl.Date

    # display failed values, GH#4706
    with pytest.raises(pl.ComputeError, match="foobar"):
        pl.Series(["1", "2", "3", "4", "foobar"]).cast(int)


def test_to_pandas() -> None:
    for test_data in (
        [1, None, 2],
        ["abc", None, "xyz"],
        [None, datetime.now()],
        [[1, 2], [3, 4], None],
    ):
        a = pl.Series("s", test_data)
        b = a.to_pandas()

        assert a.name == b.name
        assert b.isnull().sum() == 1

        if a.dtype == pl.List:
            vals = [(None if x is None else x.tolist()) for x in b]
        else:
            vals = b.replace({np.nan: None}).values.tolist()  # type: ignore[union-attr]

        assert vals == test_data


def test_to_python() -> None:
    a = pl.Series("a", range(20))
    b = a.to_list()
    assert isinstance(b, list)
    assert len(b) == 20

    b = a.to_list(use_pyarrow=True)
    assert isinstance(b, list)
    assert len(b) == 20

    a = pl.Series("a", [1, None, 2])
    assert a.null_count() == 1
    assert a.to_list() == [1, None, 2]


def test_sort() -> None:
    a = pl.Series("a", [2, 1, 3])
    assert_series_equal(a.sort(), pl.Series("a", [1, 2, 3]))
    assert_series_equal(a.sort(reverse=True), pl.Series("a", [3, 2, 1]))


def test_rechunk() -> None:
    a = pl.Series("a", [1, 2, 3])
    b = pl.Series("b", [4, 5, 6])
    a.append(b)
    assert a.n_chunks() == 2
    assert a.rechunk(in_place=False).n_chunks() == 1
    a.rechunk(in_place=True)
    assert a.n_chunks() == 1


def test_indexing() -> None:
    a = pl.Series("a", [1, 2, None])
    assert a[1] == 2
    assert a[2] is None
    b = pl.Series("b", [True, False])
    assert b[0]
    assert not b[1]
    a = pl.Series("a", ["a", None])
    assert a[0] == "a"
    assert a[1] is None
    a = pl.Series("a", [0.1, None])
    assert a[0] == 0.1
    assert a[1] is None


def test_arrow() -> None:
    a = pl.Series("a", [1, 2, 3, None])
    out = a.to_arrow()
    assert out == pa.array([1, 2, 3, None])

    s = cast(
        pl.Series,
        pl.from_arrow(pa.array([["foo"], ["foo", "bar"]], pa.list_(pa.utf8()))),
    )
    assert s.dtype == pl.List

    # categorical dtype tests (including various forms of empty pyarrow array)
    with pl.StringCache():
        arr0 = pa.array(["foo", "bar"], pa.dictionary(pa.int32(), pa.utf8()))
        assert_series_equal(
            pl.Series("arr", ["foo", "bar"], pl.Categorical), pl.Series("arr", arr0)
        )
        arr1 = pa.array(["xxx", "xxx", None, "yyy"]).dictionary_encode()
        arr2 = pa.array([]).dictionary_encode()
        arr3 = pa.chunked_array([], arr1.type)
        arr4 = pa.array([], arr1.type)

        assert_series_equal(
            pl.Series("arr", ["xxx", "xxx", None, "yyy"], dtype=pl.Categorical),
            pl.Series("arr", arr1),
        )
        for arr in (arr2, arr3, arr4):
            assert_series_equal(
                pl.Series("arr", [], dtype=pl.Categorical), pl.Series("arr", arr)
            )


def test_view() -> None:
    a = pl.Series("a", [1.0, 2.0, 3.0])
    assert isinstance(a.view(), np.ndarray)
    assert np.all(a.view() == np.array([1, 2, 3]))


def test_ufunc() -> None:
    # test if output dtype is calculated correctly.
    s_float32 = pl.Series("a", [1.0, 2.0, 3.0, 4.0], dtype=pl.Float32)
    assert_series_equal(
        cast(pl.Series, np.multiply(s_float32, 4)),
        pl.Series("a", [4.0, 8.0, 12.0, 16.0], dtype=pl.Float32),
    )

    s_float64 = pl.Series("a", [1.0, 2.0, 3.0, 4.0], dtype=pl.Float64)
    assert_series_equal(
        cast(pl.Series, np.multiply(s_float64, 4)),
        pl.Series("a", [4.0, 8.0, 12.0, 16.0], dtype=pl.Float64),
    )

    s_uint8 = pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt8)
    assert_series_equal(
        cast(pl.Series, np.power(s_uint8, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt8),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint8, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint8, 2, dtype=np.uint16)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt16),
    )

    s_int8 = pl.Series("a", [1, -2, 3, -4], dtype=pl.Int8)
    assert_series_equal(
        cast(pl.Series, np.power(s_int8, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int8),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int8, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int8, 2, dtype=np.int16)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int16),
    )

    s_uint32 = pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt32)
    assert_series_equal(
        cast(pl.Series, np.power(s_uint32, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt32),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint32, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    s_int32 = pl.Series("a", [1, -2, 3, -4], dtype=pl.Int32)
    assert_series_equal(
        cast(pl.Series, np.power(s_int32, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int32),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int32, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    s_uint64 = pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt64)
    assert_series_equal(
        cast(pl.Series, np.power(s_uint64, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.UInt64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_uint64, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    s_int64 = pl.Series("a", [1, -2, 3, -4], dtype=pl.Int64)
    assert_series_equal(
        cast(pl.Series, np.power(s_int64, 2)),
        pl.Series("a", [1, 4, 9, 16], dtype=pl.Int64),
    )
    assert_series_equal(
        cast(pl.Series, np.power(s_int64, 2.0)),
        pl.Series("a", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
    )

    # test if null bitmask is preserved
    a1 = pl.Series("a", [1.0, None, 3.0])
    b1 = cast(pl.Series, np.exp(a1))
    assert b1.null_count() == 1

    # test if it works with chunked series.
    a2 = pl.Series("a", [1.0, None, 3.0])
    b2 = pl.Series("b", [4.0, 5.0, None])
    a2.append(b2)
    assert a2.n_chunks() == 2
    c2 = np.multiply(a2, 3)
    assert_series_equal(
        cast(pl.Series, c2),
        pl.Series("a", [3.0, None, 9.0, 12.0, 15.0, None]),
    )


def test_get() -> None:
    a = pl.Series("a", [1, 2, 3])
    pos_idxs = pl.Series("idxs", [2, 0, 1, 0], dtype=pl.Int8)
    neg_and_pos_idxs = pl.Series(
        "neg_and_pos_idxs", [-2, 1, 0, -1, 2, -3], dtype=pl.Int8
    )
    assert a[0] == 1
    assert a[:2].to_list() == [1, 2]
    assert a[range(1)].to_list() == [1]
    assert a[range(0, 4, 2)].to_list() == [1, 3]
    for dtype in (
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
    ):
        assert a[pos_idxs.cast(dtype)].to_list() == [3, 1, 2, 1]
        assert a[pos_idxs.cast(dtype).to_numpy()].to_list() == [3, 1, 2, 1]

    for dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        nps = a[neg_and_pos_idxs.cast(dtype).to_numpy()]
        assert nps.to_list() == [2, 2, 1, 3, 3, 1]


def test_set() -> None:
    a = pl.Series("a", [True, False, True])
    mask = pl.Series("msk", [True, False, True])
    a[mask] = False
    assert_series_equal(a, pl.Series("", [False] * 3))


def test_set_value_as_list_fail() -> None:
    # only allowed for numerical physical types
    s = pl.Series("a", [1, 2, 3])
    s[[0, 2]] = [4, 5]
    assert s.to_list() == [4, 2, 5]

    # for other types it is not allowed
    s = pl.Series("a", ["a", "b", "c"])
    with pytest.raises(ValueError):
        s[[0, 1]] = ["d", "e"]

    s = pl.Series("a", [True, False, False])
    with pytest.raises(ValueError):
        s[[0, 1]] = [True, False]


@pytest.mark.parametrize("key", [True, False, 1.0])
def test_set_invalid_key(key: Any) -> None:
    s = pl.Series("a", [1, 2, 3])
    with pytest.raises(ValueError):
        s[key] = 1


@pytest.mark.parametrize(
    "key",
    [
        pl.Series([False, True, True]),
        pl.Series([1, 2], dtype=UInt32),
        pl.Series([1, 2], dtype=UInt64),
    ],
)
def test_set_key_series(key: pl.Series) -> None:
    """Only UInt32/UInt64/bool are allowed."""
    s = pl.Series("a", [1, 2, 3])
    s[key] = 4
    assert_series_equal(s, pl.Series("a", [1, 4, 4]))


def test_set_np_array_boolean_mask() -> None:
    a = pl.Series("a", [1, 2, 3])
    mask = np.array([True, False, True])
    a[mask] = 4
    assert_series_equal(a, pl.Series("a", [4, 2, 4]))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_set_np_array(dtype: Any) -> None:
    a = pl.Series("a", [1, 2, 3])
    idx = np.array([0, 2], dtype=dtype)
    a[idx] = 4
    assert_series_equal(a, pl.Series("a", [4, 2, 4]))


@pytest.mark.parametrize("idx", [[0, 2], (0, 2)])  # noqa: PT007
def test_set_list_and_tuple(idx: list[int] | tuple[int]) -> None:
    a = pl.Series("a", [1, 2, 3])
    a[idx] = 4
    assert_series_equal(a, pl.Series("a", [4, 2, 4]))


def test_fill_null() -> None:
    a = pl.Series("a", [1, 2, None])
    verify_series_and_expr_api(
        a, pl.Series("a", [1, 2, 2]), "fill_null", strategy="forward"
    )

    verify_series_and_expr_api(
        a, pl.Series("a", [1, 2, 14], dtype=Int64), "fill_null", 14
    )

    a = pl.Series("a", [0.0, 1.0, None, 2.0, None, 3.0])

    assert a.fill_null(0).to_list() == [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
    assert a.fill_null(strategy="zero").to_list() == [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
    assert a.fill_null(strategy="max").to_list() == [0.0, 1.0, 3.0, 2.0, 3.0, 3.0]
    assert a.fill_null(strategy="min").to_list() == [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
    assert a.fill_null(strategy="one").to_list() == [0.0, 1.0, 1.0, 2.0, 1.0, 3.0]
    assert a.fill_null(strategy="forward").to_list() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    assert a.fill_null(strategy="backward").to_list() == [0.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert a.fill_null(strategy="mean").to_list() == [0.0, 1.0, 1.5, 2.0, 1.5, 3.0]

    df = pl.DataFrame(
        [
            pl.Series("i32", [1, 2, None], dtype=pl.Int32),
            pl.Series("i64", [1, 2, None], dtype=pl.Int64),
            pl.Series("f32", [1, 2, None], dtype=pl.Float32),
            pl.Series("cat", ["a", "b", None], dtype=pl.Categorical),
            pl.Series("str", ["a", "b", None], dtype=pl.Utf8),
            pl.Series("bool", [True, True, None], dtype=pl.Boolean),
        ]
    )

    assert df.fill_null(0, matches_supertype=False).fill_null("bar").fill_null(
        False
    ).to_dict(False) == {
        "i32": [1, 2, None],
        "i64": [1, 2, 0],
        "f32": [1.0, 2.0, None],
        "cat": ["a", "b", "bar"],
        "str": ["a", "b", "bar"],
        "bool": [True, True, False],
    }

    assert df.fill_null(0, matches_supertype=True).fill_null("bar").fill_null(
        False
    ).to_dict(False) == {
        "i32": [1, 2, 0],
        "i64": [1, 2, 0],
        "f32": [1.0, 2.0, 0.0],
        "cat": ["a", "b", "bar"],
        "str": ["a", "b", "bar"],
        "bool": [True, True, False],
    }
    df = pl.DataFrame({"a": [1, None, 2, None]})

    out = df.with_columns(
        [
            pl.col("a").cast(pl.UInt8).alias("u8"),
            pl.col("a").cast(pl.UInt16).alias("u16"),
            pl.col("a").cast(pl.UInt32).alias("u32"),
            pl.col("a").cast(pl.UInt64).alias("u64"),
        ]
    ).fill_null(3)

    assert out.to_dict(False) == {
        "a": [1, 3, 2, 3],
        "u8": [1, 3, 2, 3],
        "u16": [1, 3, 2, 3],
        "u32": [1, 3, 2, 3],
        "u64": [1, 3, 2, 3],
    }
    assert out.dtypes == [pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]


def test_fill_nan() -> None:
    nan = float("nan")
    a = pl.Series("a", [1.0, nan, 2.0, nan, 3.0])
    assert a.fill_nan(None).series_equal(
        pl.Series("a", [1.0, None, 2.0, None, 3.0]), null_equal=True
    )
    assert a.fill_nan(0).series_equal(
        pl.Series("a", [1.0, 0.0, 2.0, 0.0, 3.0]), null_equal=True
    )


def test_apply() -> None:
    a = pl.Series("a", [1, 2, None])
    b = a.apply(lambda x: x**2)
    assert list(b) == [1, 4, None]

    a = pl.Series("a", ["foo", "bar", None])
    b = a.apply(lambda x: x + "py")
    assert list(b) == ["foopy", "barpy", None]

    b = a.apply(lambda x: len(x), return_dtype=pl.Int32)
    assert list(b) == [3, 3, None]

    b = a.apply(lambda x: len(x))
    assert list(b) == [3, 3, None]

    # just check that it runs (somehow problem with conditional compilation)
    a = pl.Series("a", [2, 2, 3]).cast(pl.Datetime)
    a.apply(lambda x: x)
    a = pl.Series("a", [2, 2, 3]).cast(pl.Date)
    a.apply(lambda x: x)


def test_shift() -> None:
    a = pl.Series("a", [1, 2, 3])
    assert_series_equal(a.shift(1), pl.Series("a", [None, 1, 2]))
    assert_series_equal(a.shift(-1), pl.Series("a", [2, 3, None]))
    assert_series_equal(a.shift(-2), pl.Series("a", [3, None, None]))
    assert_series_equal(a.shift_and_fill(-1, 10), pl.Series("a", [2, 3, 10]))


def test_rolling() -> None:
    a = pl.Series("a", [1, 2, 3, 2, 1])
    assert_series_equal(a.rolling_min(2), pl.Series("a", [None, 1, 2, 2, 1]))
    assert_series_equal(a.rolling_max(2), pl.Series("a", [None, 2, 3, 3, 2]))
    assert_series_equal(a.rolling_sum(2), pl.Series("a", [None, 3, 5, 5, 3]))
    assert_series_equal(a.rolling_mean(2), pl.Series("a", [None, 1.5, 2.5, 2.5, 1.5]))

    assert a.rolling_std(2).to_list()[1] == pytest.approx(0.7071067811865476)
    assert a.rolling_var(2).to_list()[1] == pytest.approx(0.5)
    assert_series_equal(
        a.rolling_median(4), pl.Series("a", [None, None, None, 2, 2], dtype=Float64)
    )
    assert_series_equal(
        a.rolling_quantile(0, "nearest", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    assert_series_equal(
        a.rolling_quantile(0, "lower", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    assert_series_equal(
        a.rolling_quantile(0, "higher", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    assert a.rolling_skew(4).null_count() == 3

    # 3099
    # test if we maintain proper dtype
    for dt in [pl.Float32, pl.Float64]:
        assert (
            pl.Series([1, 2, 3], dtype=dt)
            .rolling_min(2, weights=[0.1, 0.2])
            .series_equal(pl.Series([None, 0.1, 0.2], dtype=dt), True)
        )

    df = pl.DataFrame({"val": [1.0, 2.0, 3.0, np.NaN, 5.0, 6.0, 7.0]})

    for e in [
        pl.col("val").rolling_min(window_size=3),
        pl.col("val").rolling_max(window_size=3),
    ]:
        out = df.with_column(e).to_series()
        assert out.null_count() == 2
        assert np.isnan(out.to_numpy()).sum() == 5

    expected = [None, None, 2.0, 3.0, 5.0, 6.0, 6.0]
    assert (
        df.with_column(pl.col("val").rolling_median(window_size=3))
        .to_series()
        .to_list()
        == expected
    )
    assert (
        df.with_column(pl.col("val").rolling_quantile(0.5, window_size=3))
        .to_series()
        .to_list()
        == expected
    )

    nan = float("nan")
    a = pl.Series("a", [11.0, 2.0, 9.0, nan, 8.0])
    assert_series_equal(
        a.rolling_sum(3),
        pl.Series("a", [None, None, 22.0, nan, nan]),
    )
    assert_series_equal(
        a.rolling_apply(np.nansum, 3),
        pl.Series("a", [None, None, 22.0, 11.0, 17.0]),
    )


def test_object() -> None:
    vals = [[12], "foo", 9]
    a = pl.Series("a", vals)
    assert a.dtype == pl.Object
    assert a.to_list() == vals
    assert a[1] == "foo"


def test_repeat() -> None:
    s = pl.repeat(1, 10, eager=True)
    assert s.dtype == pl.Int64
    assert s.len() == 10
    s = pl.repeat("foo", 10, eager=True)
    assert s.dtype == pl.Utf8
    assert s.len() == 10
    s = pl.repeat(1.0, 5, eager=True)
    assert s.dtype == pl.Float64
    assert s.len() == 5
    assert s.to_list() == [1.0, 1.0, 1.0, 1.0, 1.0]
    s = pl.repeat(True, 5, eager=True)
    assert s.dtype == pl.Boolean
    assert s.len() == 5


def test_median() -> None:
    s = pl.Series([1, 2, 3])
    assert s.median() == 2


def test_quantile() -> None:
    s = pl.Series([1, 2, 3])
    assert s.quantile(0.5, "nearest") == 2
    assert s.quantile(0.5, "lower") == 2
    assert s.quantile(0.5, "higher") == 2


def test_shape() -> None:
    s = pl.Series([1, 2, 3])
    assert s.shape == (3,)


@pytest.mark.parametrize("arrow_available", [True, False])
def test_create_list_series(arrow_available: bool, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        pl.internals.series.series, "_PYARROW_AVAILABLE", arrow_available
    )
    a = [[1, 2], None, [None, 3]]
    s = pl.Series("", a)
    assert s.to_list() == a


def test_iter() -> None:
    s = pl.Series("", [1, 2, 3])

    itr = s.__iter__()
    assert itr.__next__() == 1
    assert itr.__next__() == 2
    assert itr.__next__() == 3
    assert sum(s) == 6


def test_empty() -> None:
    a = pl.Series(dtype=pl.Int8)
    assert a.dtype == pl.Int8
    assert a.is_empty()

    a = pl.Series()
    assert a.dtype == pl.Float32
    assert a.is_empty()

    a = pl.Series("name", [])
    assert a.dtype == pl.Float32
    assert a.is_empty()

    a = pl.Series(values=(), dtype=pl.Int8)
    assert a.dtype == pl.Int8
    assert a.is_empty()

    assert_series_equal(pl.Series(), pl.Series())
    assert_series_equal(
        pl.Series(dtype=pl.Int32), pl.Series(dtype=pl.Int64), check_dtype=False
    )

    a = pl.Series(name="a", values=[1, 2, 3], dtype=pl.Int16)
    empty_a = a.cleared()
    assert a.dtype == empty_a.dtype
    assert len(empty_a) == 0

    with pytest.raises(ValueError, match="ambiguous"):
        not empty_a


def test_describe() -> None:
    num_s = pl.Series([1, 2, 3])
    float_s = pl.Series([1.3, 4.6, 8.9])
    str_s = pl.Series(["abc", "pqr", "xyz"])
    bool_s = pl.Series([True, False, None, True, True])
    date_s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
    empty_s = pl.Series(np.empty(0))

    assert {k: v for k, v in num_s.describe().rows()} == {
        "count": 3.0,
        "max": 3.0,
        "mean": 2.0,
        "min": 1.0,
        "null_count": 0.0,
        "std": 1.0,
    }
    assert {k: v for k, v in float_s.describe().rows()} == {
        "count": 3.0,
        "max": 8.9,
        "mean": 4.933333333333334,
        "min": 1.3,
        "null_count": 0.0,
        "std": 3.8109491381194442,
    }
    assert {k: v for k, v in str_s.describe().rows()} == {
        "count": 3,
        "null_count": 0,
        "unique": 3,
    }
    assert {k: v for k, v in bool_s.describe().rows()} == {
        "count": 5,
        "null_count": 1,
        "sum": 3,
    }
    assert {k: v for k, v in date_s.describe().rows()} == {
        "count": "3",
        "max": "2021-01-03",
        "min": "2021-01-01",
        "null_count": "0",
    }

    with pytest.raises(ValueError):
        assert empty_s.describe()


def test_is_in() -> None:
    s = pl.Series(["a", "b", "c"])

    out = s.is_in(["a", "b"])
    assert out.to_list() == [True, True, False]

    # Check if empty list is converted to pl.Utf8.
    out = s.is_in([])
    assert out.to_list() == [False]  # one element?

    out = s.is_in(["x", "y", "z"])
    assert out.to_list() == [False, False, False]

    df = pl.DataFrame({"a": [1.0, 2.0], "b": [1, 4], "c": ["e", "d"]})
    assert df.select(pl.col("a").is_in(pl.col("b"))).to_series().to_list() == [
        True,
        False,
    ]
    assert df.select(pl.col("b").is_in([])).to_series().to_list() == [False]
    assert df.select(pl.col("b").is_in(["x", "x"])).to_series().to_list() == [
        False,
        False,
    ]


def test_slice() -> None:
    s = pl.Series(name="a", values=[0, 1, 2, 3, 4, 5], dtype=pl.UInt8)
    for srs_slice, expected in (
        [s.slice(2, 3), [2, 3, 4]],
        [s.slice(4, 1), [4]],
        [s.slice(4, None), [4, 5]],
        [s.slice(3), [3, 4, 5]],
        [s.slice(-2), [4, 5]],
    ):
        assert srs_slice.to_list() == expected  # type: ignore[attr-defined]

    for py_slice in (
        slice(1, 2),
        slice(0, 2, 2),
        slice(3, -3, -1),
        slice(1, None, -2),
        slice(-1, -3, -1),
        slice(-3, None, -3),
    ):
        # confirm series slice matches python slice
        assert s[py_slice].to_list() == s.to_list()[py_slice]


def test_str_slice() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    assert df["a"].str.slice(-3).to_list() == ["bar", "foo"]
    assert df.select([pl.col("a").str.slice(2, 4)])["a"].to_list() == ["obar", "rfoo"]


def test_arange_expr() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    out = df.select([pl.arange(0, pl.col("a").count() * 10)])
    assert out.shape == (20, 1)
    assert out.to_series(0)[-1] == 19

    # eager arange
    out2 = pl.arange(0, 10, 2, eager=True)
    assert out2.to_list() == [0, 2, 4, 6, 8]

    out3 = pl.arange(pl.Series([0, 19]), pl.Series([3, 39]), step=2, eager=True)
    assert out3.dtype == pl.List
    assert out3[0].to_list() == [0, 2]

    df = pl.DataFrame({"start": [1, 2, 3, 5, 5, 5], "stop": [8, 3, 12, 8, 8, 8]})

    assert df.select(pl.arange(pl.lit(1), pl.col("stop") + 1).alias("test")).to_dict(
        False
    ) == {
        "test": [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    }


def test_round() -> None:
    a = pl.Series("f", [1.003, 2.003])
    b = a.round(2)
    assert b.to_list() == [1.00, 2.00]


def test_apply_list_out() -> None:
    s = pl.Series("count", [3, 2, 2])
    out = s.apply(lambda val: pl.repeat(val, val, eager=True))
    assert out[0].to_list() == [3, 3, 3]
    assert out[1].to_list() == [2, 2]
    assert out[2].to_list() == [2, 2]


def test_is_first() -> None:
    s = pl.Series("", [1, 1, 2])
    assert s.is_first().to_list() == [True, False, True]


def test_reinterpret() -> None:
    s = pl.Series("a", [1, 1, 2], dtype=pl.UInt64)
    assert s.reinterpret(signed=True).dtype == pl.Int64
    df = pl.DataFrame([s])
    assert df.select([pl.col("a").reinterpret(signed=True)])["a"].dtype == pl.Int64


def test_mode() -> None:
    s = pl.Series("a", [1, 1, 2])
    assert s.mode().to_list() == [1]

    df = pl.DataFrame([s])
    assert df.select([pl.col("a").mode()])["a"].to_list() == [1]


def test_jsonpath_single() -> None:
    s = pl.Series(['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}'])
    expected = pl.Series(
        [
            "1",
            None,
            "2",
            "2.1",
            "true",
        ]
    )
    verify_series_and_expr_api(s, expected, "str.json_path_match", "$.a")


def test_extract_regex() -> None:
    s = pl.Series(
        [
            "http://vote.com/ballon_dor?candidate=messi&ref=polars",
            "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
            "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ]
    )
    expected = pl.Series(
        [
            "messi",
            None,
            "ronaldo",
        ]
    )
    verify_series_and_expr_api(s, expected, "str.extract", r"candidate=(\w+)", 1)


def test_rank() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert_series_equal(
        s.rank("dense"), pl.Series("a", [2, 3, 4, 3, 3, 4, 1], dtype=UInt32)
    )

    df = pl.DataFrame([s])
    assert df.select(pl.col("a").rank("dense"))["a"].to_list() == [2, 3, 4, 3, 3, 4, 1]

    assert_series_equal(
        s.rank("dense", reverse=True),
        pl.Series("a", [3, 2, 1, 2, 2, 1, 4], dtype=UInt32),
    )


def test_diff() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    expected = pl.Series("a", [1, 1, -1, 0, 1, -3])

    assert_series_equal(s.diff(null_behavior="drop"), expected)

    df = pl.DataFrame([s])
    assert_series_equal(
        df.select(pl.col("a").diff())["a"], pl.Series("a", [None, 1, 1, -1, 0, 1, -3])
    )


def test_pct_change() -> None:
    s = pl.Series("a", [1, 2, 4, 8, 16, 32, 64])
    expected = pl.Series("a", [None, None, float("inf"), 3.0, 3.0, 3.0, 3.0])
    verify_series_and_expr_api(s, expected, "pct_change", 2)


def test_skew() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert s.skew(True) == pytest.approx(-0.5953924651018018)
    assert s.skew(False) == pytest.approx(-0.7717168360221258)

    df = pl.DataFrame([s])
    assert np.isclose(df.select(pl.col("a").skew(False))["a"][0], -0.7717168360221258)


def test_kurtosis() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    expected = -0.6406250000000004

    assert s.kurtosis() == pytest.approx(expected)
    df = pl.DataFrame([s])
    assert np.isclose(df.select(pl.col("a").kurtosis())["a"][0], expected)


def test_arr_lengths() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    assert_series_equal(s.arr.lengths(), pl.Series("a", [2, 3], dtype=UInt32))
    df = pl.DataFrame([s])
    assert_series_equal(
        df.select(pl.col("a").arr.lengths())["a"], pl.Series("a", [2, 3], dtype=UInt32)
    )


def test_arr_arithmetic() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    assert_series_equal(s.arr.sum(), pl.Series("a", [3, 6]))
    assert_series_equal(s.arr.mean(), pl.Series("a", [1.5, 2.0]))
    assert_series_equal(s.arr.max(), pl.Series("a", [2, 3]))
    assert_series_equal(s.arr.min(), pl.Series("a", [1, 1]))


def test_arr_ordering() -> None:
    s = pl.Series("a", [[2, 1], [1, 3, 2]])
    assert_series_equal(s.arr.sort(), pl.Series("a", [[1, 2], [1, 2, 3]]))
    assert_series_equal(s.arr.reverse(), pl.Series("a", [[1, 2], [2, 3, 1]]))


def test_arr_unique() -> None:
    s = pl.Series("a", [[2, 1], [1, 2, 2]])
    result = s.arr.unique()
    assert len(result) == 2
    assert sorted(result[0]) == [1, 2]
    assert sorted(result[1]) == [1, 2]


def test_sqrt() -> None:
    s = pl.Series("a", [1, 2])
    assert_series_equal(s.sqrt(), pl.Series("a", [1.0, np.sqrt(2)]))
    df = pl.DataFrame([s])
    assert_series_equal(
        df.select(pl.col("a").sqrt())["a"], pl.Series("a", [1.0, np.sqrt(2)])
    )


def test_range() -> None:
    s1 = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    assert s1[2:5].series_equal(s1[range(2, 5)])

    ranges = [range(-2, 1), range(3), range(2, 8, 2)]

    s2 = pl.Series("b", ranges, dtype=pl.List(pl.Int8))
    assert s2.to_list() == [[-2, -1, 0], [0, 1, 2], [2, 4, 6]]
    assert s2.dtype == pl.List(pl.Int8)
    assert s2.name == "b"

    s3 = pl.Series("c", (ranges for _ in range(3)))
    assert s3.to_list() == [
        [[-2, -1, 0], [0, 1, 2], [2, 4, 6]],
        [[-2, -1, 0], [0, 1, 2], [2, 4, 6]],
        [[-2, -1, 0], [0, 1, 2], [2, 4, 6]],
    ]
    assert s3.dtype == pl.List(pl.List(pl.Int64))

    df = pl.DataFrame([s1])
    assert df[2:5].frame_equal(df[range(2, 5)])


def test_strict_cast() -> None:
    with pytest.raises(pl.ComputeError):
        pl.Series("a", [2**16]).cast(dtype=pl.Int16, strict=True)
    with pytest.raises(pl.ComputeError):
        pl.DataFrame({"a": [2**16]}).select([pl.col("a").cast(pl.Int16, strict=True)])


def test_list_concat() -> None:
    s0 = pl.Series("a", [[1, 2]])
    s1 = pl.Series("b", [[3, 4, 5]])
    expected = pl.Series("a", [[1, 2, 3, 4, 5]])

    out = s0.arr.concat([s1])
    assert out.series_equal(expected)

    out = s0.arr.concat(s1)
    assert out.series_equal(expected)

    df = pl.DataFrame([s0, s1])
    assert df.select(pl.concat_list(["a", "b"]).alias("a"))["a"].series_equal(expected)
    assert df.select(pl.col("a").arr.concat("b").alias("a"))["a"].series_equal(expected)
    assert df.select(pl.col("a").arr.concat(["b"]).alias("a"))["a"].series_equal(
        expected
    )


def test_floor_divide() -> None:
    s = pl.Series("a", [1, 2, 3])
    assert_series_equal(s // 2, pl.Series("a", [0, 1, 1]))
    assert_series_equal(
        pl.DataFrame([s]).select(pl.col("a") // 2)["a"], pl.Series("a", [0, 1, 1])
    )


def test_true_divide() -> None:
    s = pl.Series("a", [1, 2])
    assert_series_equal(s / 2, pl.Series("a", [0.5, 1.0]))
    assert_series_equal(
        pl.DataFrame([s]).select(pl.col("a") / 2)["a"], pl.Series("a", [0.5, 1.0])
    )

    # rtruediv
    assert_series_equal(
        pl.DataFrame([s]).select(2 / pl.col("a"))["literal"],
        pl.Series("literal", [2.0, 1.0]),
    )

    # https://github.com/pola-rs/polars/issues/1369
    vals = [3000000000, 2, 3]
    foo = pl.Series(vals)
    assert_series_equal(foo / 1, pl.Series(vals, dtype=Float64))
    assert_series_equal(
        pl.DataFrame({"a": vals}).select([pl.col("a") / 1])["a"],
        pl.Series("a", vals, dtype=Float64),
    )


def test_invalid_categorical() -> None:
    s = pl.Series("cat_series", ["a", "b", "b", "c", "a"]).cast(pl.Categorical)
    assert s.std() is None
    assert s.var() is None
    assert s.median() is None
    assert s.quantile(0.5) is None
    assert s.mode().to_list() == [None]


def test_bitwise() -> None:
    a = pl.Series("a", [1, 2, 3])
    b = pl.Series("b", [3, 4, 5])
    assert_series_equal(a & b, pl.Series("a", [1, 0, 1]))
    assert_series_equal(a | b, pl.Series("a", [3, 6, 7]))
    assert_series_equal(a ^ b, pl.Series("a", [2, 6, 6]))

    df = pl.DataFrame([a, b])
    out = df.select(
        [
            (pl.col("a") & pl.col("b")).alias("and"),
            (pl.col("a") | pl.col("b")).alias("or"),
            (pl.col("a") ^ pl.col("b")).alias("xor"),
        ]
    )
    assert_series_equal(out["and"], pl.Series("and", [1, 0, 1]))
    assert_series_equal(out["or"], pl.Series("or", [3, 6, 7]))
    assert_series_equal(out["xor"], pl.Series("xor", [2, 6, 6]))

    # ensure mistaken use of logical 'and'/'or' raises an exception
    with pytest.raises(ValueError, match="ambiguous"):
        a and b

    with pytest.raises(ValueError, match="ambiguous"):
        a or b


def test_to_numpy(monkeypatch: Any) -> None:
    for writable in [False, True]:
        for flag in [False, True]:
            monkeypatch.setattr(pl.internals.series.series, "_PYARROW_AVAILABLE", flag)

            np_array = pl.Series("a", [1, 2, 3], pl.UInt8).to_numpy(writable=writable)

            np.testing.assert_array_equal(np_array, np.array([1, 2, 3], dtype=np.uint8))
            # Test if numpy array is readonly or writable.
            assert np_array.flags.writeable == writable

            if writable:
                np_array[1] += 10
                np.testing.assert_array_equal(
                    np_array, np.array([1, 12, 3], dtype=np.uint8)
                )

            np_array_with_missing_values = pl.Series(
                "a", [None, 2, 3], pl.UInt8
            ).to_numpy(writable=writable)

            np.testing.assert_array_equal(
                np_array_with_missing_values,
                np.array(
                    [np.NaN, 2.0, 3.0],
                    dtype=(np.float64 if flag is True else np.float32),
                ),
            )

            if writable:
                # As Null values can't be encoded natively in a numpy array,
                # this array will never be a view.
                assert np_array_with_missing_values.flags.writeable == writable


def test_from_generator_or_iterable() -> None:
    # generator function
    def gen(n: int) -> Iterator[int]:
        yield from range(n)

    # iterable object
    class Data:
        def __init__(self, n: int):
            self._n = n

        def __iter__(self) -> Iterator[int]:
            yield from gen(self._n)

    expected = pl.Series("s", range(10))
    assert expected.dtype == pl.Int64

    for generated_series in (
        pl.Series("s", values=gen(10)),
        pl.Series("s", values=Data(10)),
        pl.Series("s", values=(x for x in gen(10))),
    ):
        assert_series_equal(expected, generated_series)

    # test 'iterable_to_pyseries' directly to validate 'chunk_size' behaviour
    ps1 = iterable_to_pyseries("s", gen(10), dtype=pl.UInt8)
    ps2 = iterable_to_pyseries("s", gen(10), dtype=pl.UInt8, chunk_size=3)
    ps3 = iterable_to_pyseries("s", Data(10), dtype=pl.UInt8, chunk_size=6)

    expected = pl.Series("s", range(10), dtype=pl.UInt8)
    assert expected.dtype == pl.UInt8

    for ps in (ps1, ps2, ps3):
        generated_series = pl.Series("s")
        generated_series._s = ps
        assert_series_equal(expected, generated_series)

    # empty generator
    assert_series_equal(pl.Series("s", []), pl.Series("s", values=gen(0)))


def test_from_sequences(monkeypatch: Any) -> None:
    # test int, str, bool, flt
    values = [
        [[1], [None, 3]],
        [["foo"], [None, "bar"]],
        [[True], [None, False]],
        [[1.0], [None, 3.0]],
    ]

    for vals in values:
        monkeypatch.setattr(pl.internals.series.series, "_PYARROW_AVAILABLE", False)
        a = pl.Series("a", vals)
        monkeypatch.setattr(pl.internals.series.series, "_PYARROW_AVAILABLE", True)
        b = pl.Series("a", vals)
        assert a.series_equal(b, null_equal=True)
        assert a.to_list() == vals


def test_comparisons_int_series_to_float() -> None:
    srs_int = pl.Series([1, 2, 3, 4])

    assert_series_equal(srs_int - 1.0, pl.Series([0.0, 1.0, 2.0, 3.0]))
    assert_series_equal(srs_int + 1.0, pl.Series([2.0, 3.0, 4.0, 5.0]))
    assert_series_equal(srs_int * 2.0, pl.Series([2.0, 4.0, 6.0, 8.0]))
    assert_series_equal(srs_int / 2.0, pl.Series([0.5, 1.0, 1.5, 2.0]))
    assert_series_equal(srs_int % 2.0, pl.Series([1.0, 0.0, 1.0, 0.0]))
    assert_series_equal(4.0 % srs_int, pl.Series([0.0, 0.0, 1.0, 0.0]))

    assert_series_equal(srs_int - pl.lit(1.0), pl.Series([0.0, 1.0, 2.0, 3.0]))
    assert_series_equal(srs_int + pl.lit(1.0), pl.Series([2.0, 3.0, 4.0, 5.0]))

    assert_series_equal(srs_int // 2.0, pl.Series([0.0, 1.0, 1.0, 2.0]))
    assert_series_equal(srs_int < 3.0, pl.Series([True, True, False, False]))
    assert_series_equal(srs_int <= 3.0, pl.Series([True, True, True, False]))
    assert_series_equal(srs_int > 3.0, pl.Series([False, False, False, True]))
    assert_series_equal(srs_int >= 3.0, pl.Series([False, False, True, True]))
    assert_series_equal(srs_int == 3.0, pl.Series([False, False, True, False]))
    assert_series_equal(srs_int - True, pl.Series([0, 1, 2, 3]))


def test_comparisons_float_series_to_int() -> None:
    srs_float = pl.Series([1.0, 2.0, 3.0, 4.0])

    assert_series_equal(srs_float - 1, pl.Series([0.0, 1.0, 2.0, 3.0]))
    assert_series_equal(srs_float + 1, pl.Series([2.0, 3.0, 4.0, 5.0]))
    assert_series_equal(srs_float * 2, pl.Series([2.0, 4.0, 6.0, 8.0]))
    assert_series_equal(srs_float / 2, pl.Series([0.5, 1.0, 1.5, 2.0]))
    assert_series_equal(srs_float % 2, pl.Series([1.0, 0.0, 1.0, 0.0]))
    assert_series_equal(4 % srs_float, pl.Series([0.0, 0.0, 1.0, 0.0]))

    assert_series_equal(srs_float - pl.lit(1), pl.Series([0.0, 1.0, 2.0, 3.0]))
    assert_series_equal(srs_float + pl.lit(1), pl.Series([2.0, 3.0, 4.0, 5.0]))

    assert_series_equal(srs_float // 2, pl.Series([0.0, 1.0, 1.0, 2.0]))
    assert_series_equal(srs_float < 3, pl.Series([True, True, False, False]))
    assert_series_equal(srs_float <= 3, pl.Series([True, True, True, False]))
    assert_series_equal(srs_float > 3, pl.Series([False, False, False, True]))
    assert_series_equal(srs_float >= 3, pl.Series([False, False, True, True]))
    assert_series_equal(srs_float == 3, pl.Series([False, False, True, False]))
    assert_series_equal(srs_float - True, pl.Series([0.0, 1.0, 2.0, 3.0]))


def test_comparisons_bool_series_to_int() -> None:
    srs_bool = pl.Series([True, False])
    # todo: do we want this to work?
    assert_series_equal(srs_bool / 1, pl.Series([True, False], dtype=Float64))
    match = (
        r"cannot do arithmetic with series of dtype: Boolean"
        r" and argument of type: <class 'bool'>"
    )
    with pytest.raises(ValueError, match=match):
        srs_bool - 1
    with pytest.raises(ValueError, match=match):
        srs_bool + 1
    match = (
        r"cannot do arithmetic with series of dtype: Boolean"
        r" and argument of type: <class 'bool'>"
    )
    with pytest.raises(ValueError, match=match):
        srs_bool % 2
    with pytest.raises(ValueError, match=match):
        srs_bool * 1
    with pytest.raises(
        TypeError, match=r"'<' not supported between instances of 'Series' and 'int'"
    ):
        srs_bool < 2  # noqa: B015
    with pytest.raises(
        TypeError, match=r"'>' not supported between instances of 'Series' and 'int'"
    ):
        srs_bool > 2  # noqa: B015


def test_abs() -> None:
    # ints
    s = pl.Series([1, -2, 3, -4])
    assert_series_equal(s.abs(), pl.Series([1, 2, 3, 4]))
    assert_series_equal(cast(pl.Series, np.abs(s)), pl.Series([1, 2, 3, 4]))

    # floats
    s = pl.Series([1.0, -2.0, 3, -4.0])
    assert_series_equal(s.abs(), pl.Series([1.0, 2.0, 3.0, 4.0]))
    assert_series_equal(cast(pl.Series, np.abs(s)), pl.Series([1.0, 2.0, 3.0, 4.0]))
    assert_series_equal(
        pl.select(pl.lit(s).abs()).to_series(), pl.Series([1.0, 2.0, 3.0, 4.0])
    )


def test_to_dummies() -> None:
    s = pl.Series("a", [1, 2, 3])
    result = s.to_dummies()
    expected = pl.DataFrame({"a_1": [1, 0, 0], "a_2": [0, 1, 0], "a_3": [0, 0, 1]})
    assert result.frame_equal(expected)


def test_value_counts() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    result = s.value_counts()
    expected = pl.DataFrame({"a": [1, 2, 3], "counts": [1, 2, 1]})
    result_sorted: pl.DataFrame = result.sort("a")
    assert result_sorted.frame_equal(expected)


def test_chunk_lengths() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    # this is a Series with one chunk, of length 4
    assert s.n_chunks() == 1
    assert s.chunk_lengths() == [4]


def test_limit() -> None:
    s = pl.Series("a", [1, 2, 3])
    assert s.limit(2).series_equal(pl.Series("a", [1, 2]))


def test_filter() -> None:
    s = pl.Series("a", [1, 2, 3])
    mask = pl.Series("", [True, False, True])
    assert s.filter(mask).series_equal(pl.Series("a", [1, 3]))

    assert s.filter([True, False, True]).series_equal(pl.Series("a", [1, 3]))


def test_take_every() -> None:
    s = pl.Series("a", [1, 2, 3, 4])
    assert s.take_every(2).series_equal(pl.Series("a", [1, 3]))


def test_argsort() -> None:
    s = pl.Series("a", [5, 3, 4, 1, 2])
    expected = pl.Series("a", [3, 4, 1, 2, 0], dtype=UInt32)

    assert s.argsort().series_equal(expected)

    expected_reverse = pl.Series("a", [0, 2, 1, 4, 3], dtype=UInt32)
    assert s.argsort(True).series_equal(expected_reverse)


def test_arg_min_and_arg_max() -> None:
    s = pl.Series("a", [5, 3, 4, 1, 2])
    assert s.arg_min() == 3
    assert s.arg_max() == 0

    s = pl.Series([None, True, False, True])
    assert s.arg_min() == 2
    assert s.arg_max() == 1
    s = pl.Series([None, None], dtype=pl.Boolean)
    assert s.arg_min() is None
    assert s.arg_max() is None
    s = pl.Series([True, True])
    assert s.arg_min() == 0
    assert s.arg_max() == 0
    s = pl.Series([False, False])
    assert s.arg_min() == 0
    assert s.arg_max() == 0


def test_is_null_is_not_null() -> None:
    s = pl.Series("a", [1.0, 2.0, 3.0, None])
    assert s.is_null().series_equal(pl.Series("a", [False, False, False, True]))
    assert s.is_not_null().series_equal(pl.Series("a", [True, True, True, False]))


def test_is_finite_is_infinite() -> None:
    s = pl.Series("a", [1.0, 2.0, np.inf])

    s.is_finite().series_equal(pl.Series("a", [True, True, False]))
    s.is_infinite().series_equal(pl.Series("a", [False, False, True]))


def test_is_nan_is_not_nan() -> None:
    s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
    assert s.is_nan().series_equal(pl.Series("a", [False, False, False, True]))
    assert s.is_not_nan().series_equal(pl.Series("a", [True, True, True, False]))


def test_is_unique() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    assert s.is_unique().series_equal(pl.Series("a", [True, False, False, True]))


def test_is_duplicated() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    assert s.is_duplicated().series_equal(pl.Series("a", [False, True, True, False]))


def test_dot() -> None:
    s1 = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("b", [4.0, 5.0, 6.0])

    assert np.array([1, 2, 3]) @ np.array([4, 5, 6]) == 32

    for dot_result in (
        s1.dot(s2),
        s1 @ s2,
        [1, 2, 3] @ s2,
        s1 @ np.array([4, 5, 6]),
    ):
        assert dot_result == 32

    with pytest.raises(ShapeError, match="length mismatch"):
        s1 @ [4, 5, 6, 7, 8]


def test_sample() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5])

    assert len(s.sample(n=2, seed=0)) == 2
    assert len(s.sample(frac=0.4, seed=0)) == 2

    assert len(s.sample(n=2, with_replacement=True, seed=0)) == 2

    # on a series of length 5, you cannot sample more than 5 items
    with pytest.raises(Exception):
        s.sample(n=10, with_replacement=False, seed=0)
    # unless you use with_replacement=True
    assert len(s.sample(n=10, with_replacement=True, seed=0)) == 10


def test_peak_max_peak_min() -> None:
    s = pl.Series("a", [4, 1, 3, 2, 5])
    result = s.peak_min()
    expected = pl.Series([False, True, False, True, False])
    assert_series_equal(result, expected)

    result = s.peak_max()
    expected = pl.Series([True, False, True, False, True])
    assert_series_equal(result, expected)


def test_shrink_to_fit() -> None:
    s = pl.Series("a", [4, 1, 3, 2, 5])
    sf = s.shrink_to_fit(in_place=True)
    assert sf is s

    s = pl.Series("a", [4, 1, 3, 2, 5])
    sf = s.shrink_to_fit(in_place=False)
    assert s is not sf


def test_str_concat() -> None:
    s = pl.Series(["1", None, "2"])
    result = s.str.concat()
    expected = pl.Series(["1-null-2"])
    assert_series_equal(result, expected)


def test_str_lengths() -> None:
    s = pl.Series(["Café", None, "345", "東京"])
    expected = pl.Series([5, None, 3, 6], dtype=UInt32)
    verify_series_and_expr_api(s, expected, "str.lengths")


def test_str_n_chars() -> None:
    s = pl.Series(["Café", None, "345", "東京"])
    expected = pl.Series([4, None, 3, 2], dtype=UInt32)
    verify_series_and_expr_api(s, expected, "str.n_chars")


def test_str_contains() -> None:
    s = pl.Series(["messi", "ronaldo", "ibrahimovic"])
    expected = pl.Series([True, False, False])
    verify_series_and_expr_api(s, expected, "str.contains", "mes")


def test_str_encode() -> None:
    s = pl.Series(["foo", "bar", None])
    hex_encoded = pl.Series(["666f6f", "626172", None])
    base64_encoded = pl.Series(["Zm9v", "YmFy", None])
    verify_series_and_expr_api(s, hex_encoded, "str.encode", "hex")
    verify_series_and_expr_api(s, base64_encoded, "str.encode", "base64")
    with pytest.raises(ValueError):
        s.str.encode("utf8")  # type: ignore[arg-type]


def test_str_decode() -> None:
    hex_encoded = pl.Series(["666f6f", "626172", None])
    base64_encoded = pl.Series(["Zm9v", "YmFy", None])
    expected = pl.Series([b"foo", b"bar", None])

    verify_series_and_expr_api(hex_encoded, expected, "str.decode", "hex")
    verify_series_and_expr_api(base64_encoded, expected, "str.decode", "base64")


def test_str_decode_exception() -> None:
    s = pl.Series(["not a valid", "626172", None])
    with pytest.raises(Exception):
        s.str.decode(encoding="hex")
    with pytest.raises(Exception):
        s.str.decode(encoding="base64")
    with pytest.raises(ValueError):
        s.str.decode("utf8")  # type: ignore[arg-type]


def test_str_replace_str_replace_all() -> None:
    s = pl.Series(["hello", "world", "test"])
    expected = pl.Series(["hell0", "w0rld", "test"])
    verify_series_and_expr_api(s, expected, "str.replace", "o", "0")

    s = pl.Series(["hello", "world", "test"])
    expected = pl.Series(["hell0", "w0rld", "test"])
    verify_series_and_expr_api(s, expected, "str.replace_all", "o", "0")


def test_str_to_lowercase() -> None:
    s = pl.Series(["Hello", "WORLD"])
    expected = pl.Series(["hello", "world"])
    verify_series_and_expr_api(s, expected, "str.to_lowercase")


def test_str_to_uppercase() -> None:
    s = pl.Series(["Hello", "WORLD"])
    expected = pl.Series(["HELLO", "WORLD"])
    verify_series_and_expr_api(s, expected, "str.to_uppercase")


def test_str_strip() -> None:
    s = pl.Series([" hello ", "world\t "])
    expected = pl.Series(["hello", "world"])
    assert_series_equal(s.str.strip(), expected)

    expected = pl.Series(["hello", "worl"])
    assert_series_equal(s.str.strip().str.strip("d"), expected)

    expected = pl.Series(["ell", "rld\t"])
    assert_series_equal(s.str.strip(" hwo"), expected)


def test_str_lstrip() -> None:
    s = pl.Series([" hello ", "\t world"])
    expected = pl.Series(["hello ", "world"])
    assert_series_equal(s.str.lstrip(), expected)

    expected = pl.Series(["ello ", "world"])
    assert_series_equal(s.str.lstrip().str.lstrip("h"), expected)

    expected = pl.Series(["ello ", "\t world"])
    assert_series_equal(s.str.lstrip("hw "), expected)


def test_str_rstrip() -> None:
    s = pl.Series([" hello ", "world\t "])
    expected = pl.Series([" hello", "world"])
    assert_series_equal(s.str.rstrip(), expected)

    expected = pl.Series([" hell", "world"])
    assert_series_equal(s.str.rstrip().str.rstrip("o"), expected)

    expected = pl.Series([" he", "wor"])
    assert_series_equal(s.str.rstrip("odl \t"), expected)


def test_str_strip_whitespace() -> None:
    a = pl.Series("a", ["trailing  ", "  leading", "  both  "])

    expected = pl.Series("a", ["trailing", "  leading", "  both"])
    verify_series_and_expr_api(a, expected, "str.rstrip")

    expected = pl.Series("a", ["trailing  ", "leading", "both  "])
    verify_series_and_expr_api(a, expected, "str.lstrip")

    expected = pl.Series("a", ["trailing", "leading", "both"])
    verify_series_and_expr_api(a, expected, "str.strip")


def test_str_strptime() -> None:
    s = pl.Series(["2020-01-01", "2020-02-02"])
    expected = pl.Series([date(2020, 1, 1), date(2020, 2, 2)])
    verify_series_and_expr_api(s, expected, "str.strptime", pl.Date, "%Y-%m-%d")

    s = pl.Series(["2020-01-01 00:00:00", "2020-02-02 03:20:10"])
    expected = pl.Series(
        [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 2, 2, 3, 20, 10)]
    )
    verify_series_and_expr_api(
        s, expected, "str.strptime", pl.Datetime, "%Y-%m-%d %H:%M:%S"
    )
    s = pl.Series(["00:00:00", "03:20:10"])
    expected = pl.Series([0, 12010000000000], dtype=pl.Time)
    verify_series_and_expr_api(s, expected, "str.strptime", pl.Time, "%H:%M:%S")


def test_dt_strftime() -> None:
    a = pl.Series("a", [10000, 20000, 30000], dtype=pl.Date)
    assert a.dtype == pl.Date
    expected = pl.Series("a", ["1997-05-19", "2024-10-04", "2052-02-20"])
    verify_series_and_expr_api(a, expected, "dt.strftime", "%F")


def test_dt_year_month_week_day_ordinal_day() -> None:
    a = pl.Series("a", [10000, 20000, 30000], dtype=pl.Date)

    exp = pl.Series("a", [1997, 2024, 2052], dtype=Int32)
    verify_series_and_expr_api(a, exp, "dt.year")

    verify_series_and_expr_api(a, pl.Series("a", [5, 10, 2], dtype=UInt32), "dt.month")
    verify_series_and_expr_api(a, pl.Series("a", [1, 5, 2], dtype=UInt32), "dt.weekday")
    verify_series_and_expr_api(a, pl.Series("a", [21, 40, 8], dtype=UInt32), "dt.week")
    verify_series_and_expr_api(a, pl.Series("a", [19, 4, 20], dtype=UInt32), "dt.day")
    verify_series_and_expr_api(
        a, pl.Series("a", [139, 278, 51], dtype=UInt32), "dt.ordinal_day"
    )

    assert a.dt.median() == date(2024, 10, 4)
    assert a.dt.mean() == date(2024, 10, 4)


def test_dt_datetimes() -> None:
    s = pl.Series(["2020-01-01 00:00:00.000000000", "2020-02-02 03:20:10.987654321"])
    s = s.str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S.%9f")

    # hours, minutes, seconds, milliseconds, microseconds, and nanoseconds
    verify_series_and_expr_api(s, pl.Series("", [0, 3], dtype=UInt32), "dt.hour")
    verify_series_and_expr_api(s, pl.Series("", [0, 20], dtype=UInt32), "dt.minute")
    verify_series_and_expr_api(s, pl.Series("", [0, 10], dtype=UInt32), "dt.second")
    verify_series_and_expr_api(
        s, pl.Series("", [0, 987], dtype=UInt32), "dt.millisecond"
    )
    verify_series_and_expr_api(
        s, pl.Series("", [0, 987654], dtype=UInt32), "dt.microsecond"
    )
    verify_series_and_expr_api(
        s, pl.Series("", [0, 987654321], dtype=UInt32), "dt.nanosecond"
    )

    # epoch methods
    verify_series_and_expr_api(
        s, pl.Series("", [18262, 18294], dtype=Int32), "dt.epoch", tu="d"
    )
    verify_series_and_expr_api(
        s,
        pl.Series("", [1_577_836_800, 1_580_613_610], dtype=Int64),
        "dt.epoch",
        tu="s",
    )
    verify_series_and_expr_api(
        s,
        pl.Series("", [1_577_836_800_000, 1_580_613_610_000], dtype=Int64),
        "dt.epoch",
        tu="ms",
    )
    # fractional seconds
    assert_series_equal(
        pl.Series("", [0.0, 10.987654321], dtype=Float64), s.dt.second(fractional=True)
    )


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_cast_datetime_to_time(unit: TimeUnit) -> None:
    a = pl.Series(
        "a",
        [
            datetime(2022, 9, 7, 0, 0),
            datetime(2022, 9, 6, 12, 0),
            datetime(2022, 9, 7, 23, 59, 59),
            datetime(2022, 9, 7, 23, 59, 59, 201),
        ],
        dtype=Datetime(unit),
    )
    if unit == "ms":
        # NOTE: microseconds are lost for `unit=ms`
        expected_values = [time(0, 0), time(12, 0), time(23, 59, 59), time(23, 59, 59)]
    else:
        expected_values = [
            time(0, 0),
            time(12, 0),
            time(23, 59, 59),
            time(23, 59, 59, 201),
        ]
    expected = pl.Series("a", expected_values)
    assert_series_equal(a.cast(Time), expected)


def test_reshape() -> None:
    s = pl.Series("a", [1, 2, 3, 4])
    out = s.reshape((-1, 2))
    expected = pl.Series("a", [[1, 2], [3, 4]])
    assert out.series_equal(expected)
    out = s.reshape((2, 2))
    assert out.series_equal(expected)
    out = s.reshape((2, -1))
    assert out.series_equal(expected)

    out = s.reshape((-1, 1))
    expected = pl.Series("a", [[1], [2], [3], [4]])
    assert out.series_equal(expected)

    # test lazy_dispatch
    out = pl.select(pl.lit(s).reshape((-1, 1))).to_series()
    assert out.series_equal(expected)


def test_init_categorical() -> None:
    with pl.StringCache():
        for values in [[None], ["foo", "bar"], [None, "foo", "bar"]]:
            expected = pl.Series("a", values, dtype=pl.Utf8).cast(pl.Categorical)
            a = pl.Series("a", values, dtype=pl.Categorical)
            assert_series_equal(a, expected)


def test_nested_list_types_preserved() -> None:
    expected_dtype = pl.UInt32
    srs1 = pl.Series([pl.Series([3, 4, 5, 6], dtype=expected_dtype) for _ in range(5)])
    for srs2 in srs1:
        assert srs2.dtype == expected_dtype


def test_log_exp() -> None:
    a = pl.Series("a", [1, 100, 1000])
    b = pl.Series("a", [0.0, 2.0, 3.0])
    verify_series_and_expr_api(a, b, "log10")

    expected = pl.Series("a", np.log(a.to_numpy()))
    verify_series_and_expr_api(a, expected, "log")

    expected = pl.Series("a", np.exp(b.to_numpy()))
    verify_series_and_expr_api(b, expected, "exp")


def test_shuffle() -> None:
    a = pl.Series("a", [1, 2, 3])
    out = a.shuffle(2)
    expected = pl.Series("a", [2, 1, 3])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).shuffle(2)).to_series()
    assert_series_equal(out, expected)


def test_to_physical() -> None:
    # casting an int result in an int
    a = pl.Series("a", [1, 2, 3])
    verify_series_and_expr_api(a, a, "to_physical")

    # casting a date results in an Int32
    a = pl.Series("a", [date(2020, 1, 1)] * 3)
    expected = pl.Series("a", [18262] * 3, dtype=Int32)
    verify_series_and_expr_api(a, expected, "to_physical")


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_is_between_datetime() -> None:
    s = pl.Series("a", [datetime(2020, 1, 1, 10, 0, 0), datetime(2020, 1, 1, 20, 0, 0)])
    start = datetime(2020, 1, 1, 12, 0, 0)
    end = datetime(2020, 1, 1, 23, 0, 0)
    expected = pl.Series("a", [False, True])

    # only on the expression api
    result = s.to_frame().with_column(pl.col("*").is_between(start, end))["is_between"]
    assert_series_equal(result.rename("a"), expected)


@pytest.mark.parametrize(
    "f",
    [
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
    ],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_trigonometric(f: str) -> None:
    s = pl.Series("a", [0.0, math.pi])
    expected = pl.Series("a", getattr(np, f)(s.to_numpy()))
    verify_series_and_expr_api(s, expected, f)


def test_trigonometric_invalid_input() -> None:
    # String
    s = pl.Series("a", ["1", "2", "3"])
    with pytest.raises(pl.ComputeError):
        s.sin()

    # Date
    s = pl.Series("a", [date(1990, 2, 28), date(2022, 7, 26)])
    with pytest.raises(pl.ComputeError):
        s.cosh()


def test_ewm_mean() -> None:
    a = pl.Series("a", [2, 5, 3])
    expected = pl.Series(
        "a",
        [
            2.0,
            4.0,
            3.4285714285714284,
        ],
    )
    verify_series_and_expr_api(a, expected, "ewm_mean", alpha=0.5, adjust=True)
    expected = pl.Series("a", [2.0, 3.8, 3.421053])
    verify_series_and_expr_api(a, expected, "ewm_mean", com=2.0, adjust=True)
    expected = pl.Series("a", [2.0, 3.5, 3.25])
    verify_series_and_expr_api(a, expected, "ewm_mean", alpha=0.5, adjust=False)
    a = pl.Series("a", [2, 3, 5, 7, 4])
    expected = pl.Series("a", [None, 2.666667, 4.0, 5.6, 4.774194])
    verify_series_and_expr_api(
        a, expected, "ewm_mean", alpha=0.5, adjust=True, min_periods=2
    )
    expected = pl.Series("a", [None, None, 4.0, 5.6, 4.774194])
    verify_series_and_expr_api(
        a, expected, "ewm_mean", alpha=0.5, adjust=True, min_periods=3
    )

    a = pl.Series("a", [None, 1.0, 5.0, 7.0, None, 2.0, 5.0, 4])
    expected = pl.Series(
        "a",
        [
            None,
            1.0,
            3.6666666666666665,
            5.571428571428571,
            5.571428571428571,
            3.6666666666666665,
            4.354838709677419,
            4.174603174603175,
        ],
    )
    verify_series_and_expr_api(
        a, expected, "ewm_mean", alpha=0.5, adjust=True, min_periods=1
    )
    expected = pl.Series("a", [None, 1.0, 3.0, 5.0, 5.0, 3.5, 4.25, 4.125])
    verify_series_and_expr_api(
        a, expected, "ewm_mean", alpha=0.5, adjust=False, min_periods=1
    )


def test_ewm_mean_leading_nulls() -> None:
    for min_periods in [1, 2, 3]:
        assert (
            pl.Series([1, 2, 3, 4]).ewm_mean(3, min_periods=min_periods).null_count()
            == min_periods - 1
        )
    assert pl.Series([None, 1.0, 1.0, 1.0]).ewm_mean(
        alpha=0.5, min_periods=1
    ).to_list() == [None, 1.0, 1.0, 1.0]
    assert pl.Series([None, 1.0, 1.0, 1.0]).ewm_mean(
        alpha=0.5, min_periods=2
    ).to_list() == [None, None, 1.0, 1.0]


def test_ewm_mean_min_periods() -> None:
    series = pl.Series([1.0, None, None, None])

    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=1)
    assert ewm_mean.to_list() == [1.0, 1.0, 1.0, 1.0]
    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=2)
    assert ewm_mean.to_list() == [None, None, None, None]

    series = pl.Series([1.0, None, 2.0, None, 3.0])

    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=1)
    assert ewm_mean.to_list() == [
        1.0,
        1.0,
        1.6666666666666665,
        1.6666666666666665,
        2.4285714285714284,
    ]
    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=2)
    assert ewm_mean.to_list() == [
        None,
        None,
        1.6666666666666665,
        1.6666666666666665,
        2.4285714285714284,
    ]


def test_ewm_std_var() -> None:
    series = pl.Series("a", [2, 5, 3])

    var = series.ewm_var(alpha=0.5)
    std = series.ewm_std(alpha=0.5)

    assert np.allclose(var, std**2, rtol=1e-16)


def test_ewm_param_validation() -> None:
    s = pl.Series("values", range(10))

    with pytest.raises(ValueError, match="mutually exclusive"):
        s.ewm_std(com=0.5, alpha=0.5)

    with pytest.raises(ValueError, match="mutually exclusive"):
        s.ewm_mean(span=1.5, half_life=0.75)

    with pytest.raises(ValueError, match="mutually exclusive"):
        s.ewm_var(alpha=0.5, span=1.5)

    with pytest.raises(ValueError, match="Require 'com' >= 0"):
        s.ewm_std(com=-0.5)

    with pytest.raises(ValueError, match="Require 'span' >= 1"):
        s.ewm_mean(span=0.5)

    with pytest.raises(ValueError, match="Require 'half_life' > 0"):
        s.ewm_var(half_life=0)

    for alpha in (-0.5, -0.0000001, 0.0, 1.0000001, 1.5):
        with pytest.raises(ValueError, match="Require 0 < 'alpha' <= 1"):
            s.ewm_std(alpha=alpha)


def test_extend_constant() -> None:
    today = date.today()

    for const, dtype in (
        (1, pl.Int8),
        (today, pl.Date),
        ("xyz", pl.Utf8),
        (None, pl.Float64),
    ):
        s = pl.Series("s", [None], dtype=dtype)
        expected = pl.Series("s", [None, const, const, const], dtype=dtype)
        verify_series_and_expr_api(s, expected, "extend_constant", const, 3)


def test_any_all() -> None:
    a = pl.Series("a", [True, False, True])
    assert a.any() is True
    assert a.all() is False

    a = pl.Series("a", [True, True, True])
    assert a.any() is True
    assert a.all() is True

    a = pl.Series("a", [False, False, False])
    assert a.any() is False
    assert a.all() is False


def test_product() -> None:
    a = pl.Series("a", [1, 2, 3])
    out = a.product()
    assert out == 6
    a = pl.Series("a", [1, 2, None])
    out = a.product()
    assert out is None
    a = pl.Series("a", [None, 2, 3])
    out = a.product()
    assert out is None


def test_ceil() -> None:
    a = pl.Series("a", [1.8, 1.2, 3.0])
    expected = pl.Series("a", [2.0, 2.0, 3.0])
    verify_series_and_expr_api(a, expected, "ceil")


def test_duration_arithmetic() -> None:
    # apply some basic duration math to series
    s = pl.Series([datetime(2022, 1, 1, 10, 20, 30), datetime(2022, 1, 2, 20, 40, 50)])
    d1 = pl.duration(days=5, microseconds=123456)
    d2 = timedelta(days=5, microseconds=123456)

    expected_values = [
        datetime(2022, 1, 6, 10, 20, 30, 123456),
        datetime(2022, 1, 7, 20, 40, 50, 123456),
    ]
    for d in (d1, d2):
        df1 = pl.select((s + d).alias("d_offset"))
        df2 = pl.select((d + s).alias("d_offset"))
        assert df1["d_offset"].to_list() == expected_values
        assert df1["d_offset"].series_equal(df2["d_offset"])


def test_duration_extract_times() -> None:
    a = pl.Series("a", [datetime(2021, 1, 1)])
    b = pl.Series("b", [datetime(2021, 1, 2)])

    duration = b - a
    expected = pl.Series("b", [1])
    verify_series_and_expr_api(duration, expected, "dt.days")

    expected = pl.Series("b", [24])
    verify_series_and_expr_api(duration, expected, "dt.hours")

    expected = pl.Series("b", [3600 * 24])
    verify_series_and_expr_api(duration, expected, "dt.seconds")

    expected = pl.Series("b", [3600 * 24 * int(1e3)])
    verify_series_and_expr_api(duration, expected, "dt.milliseconds")

    expected = pl.Series("b", [3600 * 24 * int(1e6)])
    verify_series_and_expr_api(duration, expected, "dt.microseconds")

    expected = pl.Series("b", [3600 * 24 * int(1e9)])
    verify_series_and_expr_api(duration, expected, "dt.nanoseconds")


def test_mean_overflow() -> None:
    arr = np.array([255] * (1 << 17), dtype="int16")
    assert arr.mean() == 255.0


def test_str_split() -> None:
    a = pl.Series("a", ["a, b", "a", "ab,c,de"])
    for out in [a.str.split(","), pl.select(pl.lit(a).str.split(",")).to_series()]:
        assert out[0].to_list() == ["a", " b"]
        assert out[1].to_list() == ["a"]
        assert out[2].to_list() == ["ab", "c", "de"]

    for out in [
        a.str.split(",", inclusive=True),
        pl.select(pl.lit(a).str.split(",", inclusive=True)).to_series(),
    ]:
        assert out[0].to_list() == ["a,", " b"]
        assert out[1].to_list() == ["a"]
        assert out[2].to_list() == ["ab,", "c,", "de"]


def test_sign() -> None:
    # Integers
    a = pl.Series("a", [-9, -0, 0, 4, None])
    expected = pl.Series("a", [-1, 0, 0, 1, None])
    verify_series_and_expr_api(a, expected, "sign")

    # Floats
    a = pl.Series("a", [-9.0, -0.0, 0.0, 4.0, None])
    expected = pl.Series("a", [-1, 0, 0, 1, None])
    verify_series_and_expr_api(a, expected, "sign")

    # Invalid input
    a = pl.Series("a", [date(1950, 2, 1), date(1970, 1, 1), date(2022, 12, 12), None])
    with pytest.raises(pl.ComputeError):
        a.sign()


def test_exp() -> None:
    a = pl.Series("a", [0.1, 0.01, None])
    expected = pl.Series("a", [1.1051709180756477, 1.010050167084168, None])
    verify_series_and_expr_api(a, expected, "exp")
    # test if we can run on empty series as well.
    assert a[:0].exp().to_list() == []


def test_cumulative_eval() -> None:
    s = pl.Series("values", [1, 2, 3, 4, 5])

    # evaluate expressions individually
    expr1 = pl.element().first()
    expr2 = pl.element().last() ** 2

    expected1 = pl.Series("values", [1, 1, 1, 1, 1])
    expected2 = pl.Series("values", [1.0, 4.0, 9.0, 16.0, 25.0])
    verify_series_and_expr_api(s, expected1, "cumulative_eval", expr1)
    verify_series_and_expr_api(s, expected2, "cumulative_eval", expr2)

    # evaluate combined expressions and validate
    expr3 = expr1 - expr2
    expected3 = pl.Series("values", [0.0, -3.0, -8.0, -15.0, -24.0])
    verify_series_and_expr_api(s, expected3, "cumulative_eval", expr3)


def test_drop_nan_ignore_null_3525() -> None:
    df = pl.DataFrame({"a": [1.0, float("NaN"), 2.0, None, 3.0, 4.0]})
    assert df.select(pl.col("a").drop_nans()).to_series().to_list() == [
        1.0,
        2.0,
        None,
        3.0,
        4.0,
    ]


def test_reverse() -> None:
    s = pl.Series("values", [1, 2, 3, 4, 5])
    assert s.reverse().to_list() == [5, 4, 3, 2, 1]

    s = pl.Series("values", ["a", "b", None, "y", "x"])
    assert s.reverse().to_list() == ["x", "y", None, "b", "a"]


def test_n_unique() -> None:
    s = pl.Series("s", [11, 11, 11, 22, 22, 33, None, None, None])
    assert s.n_unique() == 4


def test_clip() -> None:
    s = pl.Series("foo", [-50, 5, None, 50])
    assert s.clip(1, 10).to_list() == [1, 5, None, 10]


def test_mutable_borrowed_append_3915() -> None:
    s = pl.Series("s", [1, 2, 3])
    assert s.append(s).to_list() == [1, 2, 3, 1, 2, 3]


def test_set_at_idx() -> None:
    s = pl.Series("s", [1, 2, 3])

    # no-op (empty sequences)
    for x in (
        (),
        [],
        pl.Series(),
        pl.Series(dtype=pl.Int8),
        np.array([]),
        np.ndarray(shape=(0, 0)),
    ):
        s.set_at_idx(x, 8)  # type: ignore[arg-type]
        assert s.to_list() == [1, 2, 3]

    # set new values, one index at a time
    s.set_at_idx(0, 8)
    s.set_at_idx([1], None)
    assert s.to_list() == [8, None, 3]

    # set new value at multiple indexes in one go
    s.set_at_idx([0, 2], None)
    assert s.to_list() == [None, None, None]

    # try with different series dtype
    s = pl.Series("s", ["a", "b", "c"])
    s.set_at_idx((1, 2), "x")
    assert s.to_list() == ["a", "x", "x"]
    assert s.set_at_idx([0, 2], 0.12345).to_list() == ["0.12345", "x", "0.12345"]

    # set multiple values values
    s = pl.Series(["z", "z", "z"])
    assert s.set_at_idx([0, 1], ["a", "b"]).to_list() == ["a", "b", "z"]
    s = pl.Series([True, False, True])
    assert s.set_at_idx([0, 1], [False, True]).to_list() == [False, True, True]


def test_repr() -> None:
    s = pl.Series("ints", [1001, 2002, 3003])
    s_repr = repr(s)

    assert "shape: (3,)" in s_repr
    assert "Series: 'ints' [i64]" in s_repr
    for n in s.to_list():
        assert str(n) in s_repr

    class XSeries(pl.Series):
        """Custom Series class."""

    # check custom class name reflected in repr ouput
    x = XSeries("ints", [1001, 2002, 3003])
    x_repr = repr(x)

    assert "shape: (3,)" in x_repr
    assert "XSeries: 'ints' [i64]" in x_repr
    assert "1001" in x_repr
    for n in x.to_list():
        assert str(n) in x_repr


def test_repr_html(df: pl.DataFrame) -> None:
    # check it does not panic/error, and appears to contain a table
    html = pl.Series("misc", [123, 456, 789])._repr_html_()
    assert "<table" in html


def test_builtin_abs() -> None:
    s = pl.Series("s", [-1, 0, 1, None])
    assert abs(s).to_list() == [1, 0, 1, None]


@pytest.mark.parametrize(
    ("value", "unit", "exp", "exp_type"),
    [
        (13285, "d", date(2006, 5, 17), pl.Date),
        (1147880044, "s", datetime(2006, 5, 17, 15, 34, 4), pl.Datetime),
        (1147880044 * 1_000, "ms", datetime(2006, 5, 17, 15, 34, 4), pl.Datetime("ms")),
        (
            1147880044 * 1_000_000,
            "us",
            datetime(2006, 5, 17, 15, 34, 4),
            pl.Datetime("us"),
        ),
        (
            1147880044 * 1_000_000_000,
            "ns",
            datetime(2006, 5, 17, 15, 34, 4),
            pl.Datetime("ns"),
        ),
    ],
)
def test_from_epoch_expr(
    value: int, unit: EpochTimeUnit, exp: date | datetime, exp_type: PolarsDataType
) -> None:
    s = pl.Series("timestamp", [value, None])
    result = pl.from_epoch(s, unit=unit)

    expected = pl.Series("timestamp", [exp, None]).cast(exp_type)
    assert_series_equal(result, expected)


def test_get_chunks() -> None:
    a = pl.Series("a", [1, 2])
    b = pl.Series("a", [3, 4])
    chunks = pl.concat([a, b], rechunk=False).get_chunks()
    assert chunks[0].series_equal(a)
    assert chunks[1].series_equal(b)


def test_item() -> None:
    s = pl.Series("a", [1])
    assert s.item() == 1

    s = pl.Series("a", [1, 2])
    with pytest.raises(ValueError):
        s.item()

    s = pl.Series("a", [])
    with pytest.raises(ValueError):
        s.item()

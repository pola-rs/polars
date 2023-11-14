from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Iterator, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

import polars
import polars as pl
from polars.datatypes import (
    Date,
    Datetime,
    Field,
    Float64,
    Int32,
    Int64,
    Time,
    UInt32,
    UInt64,
    Unknown,
)
from polars.exceptions import ComputeError, PolarsInefficientMapWarning, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal
from polars.utils._construction import iterable_to_pyseries
from polars.utils._wrap import wrap_s

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars.type_aliases import EpochTimeUnit, PolarsDataType, TimeUnit
else:
    from polars.utils.convert import get_zoneinfo as ZoneInfo


def test_cum_agg() -> None:
    # confirm that known series give expected results
    s = pl.Series("a", [1, 2, 3, 2])
    assert_series_equal(s.cumsum(), pl.Series("a", [1, 3, 6, 8]))
    assert_series_equal(s.cummin(), pl.Series("a", [1, 1, 1, 1]))
    assert_series_equal(s.cummax(), pl.Series("a", [1, 2, 3, 3]))
    assert_series_equal(s.cumprod(), pl.Series("a", [1, 2, 6, 12]))


def test_init_inputs(monkeypatch: Any) -> None:
    nan = float("nan")
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
    assert (
        pl.Series([None, None, None]).dtype == pl.Float32
    )  # f32 type used for list with only None
    assert pl.Series([None, None, None], dtype_if_empty=pl.Int8).dtype == pl.Int8
    # note: "== []" will be cast to empty Series with Utf8 dtype.
    assert_series_equal(
        pl.Series([], dtype_if_empty=pl.Utf8) == [], pl.Series("", dtype=pl.Boolean)
    )
    assert pl.Series(values=[True, False]).dtype == pl.Boolean
    assert pl.Series(values=np.array([True, False])).dtype == pl.Boolean
    assert pl.Series(values=np.array(["foo", "bar"])).dtype == pl.Utf8
    assert pl.Series(values=["foo", "bar"]).dtype == pl.Utf8
    assert pl.Series("a", [pl.Series([1, 2, 4]), pl.Series([3, 2, 1])]).dtype == pl.List
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

    # numpy from arange, with/without dtype
    two_ints = np.arange(2, dtype=np.int64)
    three_ints = np.arange(3, dtype=np.int64)
    for res in (
        pl.Series("a", [two_ints, three_ints]),
        pl.Series("a", [two_ints, three_ints], dtype=pl.List(pl.Int64)),
    ):
        assert res.dtype == pl.List(pl.Int64)
        assert res.to_list() == [[0, 1], [0, 1, 2]]

    assert pl.Series(
        values=np.array([["foo", "bar"], ["foo2", "bar2"]])
    ).dtype == pl.List(pl.Utf8)

    # lists
    assert pl.Series("a", [[1, 2], [3, 4]]).dtype == pl.List(pl.Int64)

    # conversion of Date to Datetime
    s = pl.Series([date(2023, 1, 1), date(2023, 1, 2)], dtype=pl.Datetime)
    assert s.to_list() == [datetime(2023, 1, 1), datetime(2023, 1, 2)]
    assert Datetime == s.dtype
    assert s.dtype.time_unit == "us"  # type: ignore[union-attr]
    assert s.dtype.time_zone is None  # type: ignore[union-attr]

    # conversion of Date to Datetime with specified timezone and units
    tu: TimeUnit = "ms"
    tz = "America/Argentina/Rio_Gallegos"
    s = pl.Series([date(2023, 1, 1), date(2023, 1, 2)], dtype=pl.Datetime(tu, tz))
    d1 = datetime(2023, 1, 1, 0, 0, 0, 0, ZoneInfo(tz))
    d2 = datetime(2023, 1, 2, 0, 0, 0, 0, ZoneInfo(tz))
    assert s.to_list() == [d1, d2]
    assert Datetime == s.dtype
    assert s.dtype.time_unit == tu  # type: ignore[union-attr]
    assert s.dtype.time_zone == tz  # type: ignore[union-attr]

    # datetime64: check timeunit (auto-detect, implicit/explicit) and NaT
    d64 = pd.date_range(date(2021, 8, 1), date(2021, 8, 3)).values
    d64[1] = None

    expected = [datetime(2021, 8, 1, 0), None, datetime(2021, 8, 3, 0)]
    for dtype in (None, Datetime, Datetime("ns")):
        s = pl.Series("dates", d64, dtype)
        assert s.to_list() == expected
        assert Datetime == s.dtype
        assert s.dtype.time_unit == "ns"  # type: ignore[union-attr]

    s = pl.Series(values=d64.astype("<M8[ms]"))
    assert s.dtype.time_unit == "ms"  # type: ignore[union-attr]
    assert expected == s.to_list()

    # pandas
    assert pl.Series(pd.Series([1, 2])).dtype == pl.Int64

    # Bad inputs
    with pytest.raises(TypeError):
        pl.Series([1, 2, 3], [1, 2, 3])
    with pytest.raises(TypeError):
        pl.Series({"a": [1, 2, 3]})
    with pytest.raises(OverflowError):
        pl.Series("bigint", [2**64])

    # numpy not available
    monkeypatch.setattr(pl.series.series, "_check_for_numpy", lambda x: False)
    with pytest.raises(TypeError):
        pl.DataFrame(np.array([1, 2, 3]), schema=["a"])


def test_init_structured_objects() -> None:
    # validate init from dataclass, namedtuple, and pydantic model objects
    from typing import NamedTuple

    from polars.dependencies import dataclasses, pydantic

    @dataclasses.dataclass
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

    class TeaShipmentPD(pydantic.BaseModel):
        exporter: str
        importer: str
        product: str
        tonnes: int

    for Tea in (TeaShipmentDC, TeaShipmentNT, TeaShipmentPD):
        t0 = Tea(exporter="Sri Lanka", importer="USA", product="Ceylon", tonnes=10)
        t1 = Tea(exporter="India", importer="UK", product="Darjeeling", tonnes=25)
        t2 = Tea(exporter="China", importer="UK", product="Keemum", tonnes=40)

        s = pl.Series("t", [t0, t1, t2])

        assert isinstance(s, pl.Series)
        assert s.dtype.fields == [  # type: ignore[union-attr]
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
            {
                "exporter": "China",
                "importer": "UK",
                "product": "Keemum",
                "tonnes": 40,
            },
        ]
        assert_frame_equal(s.to_frame(), pl.DataFrame({"t": [t0, t1, t2]}))


def test_concat() -> None:
    s = pl.Series("a", [2, 1, 3])

    assert pl.concat([s, s]).len() == 6
    # check if s remains unchanged
    assert s.len() == 3


def test_equal() -> None:
    s1 = pl.Series("a", [1.0, 2.0, None], Float64)
    s2 = pl.Series("a", [1, 2, None], Int64)

    assert s1.series_equal(s2) is True
    assert s1.series_equal(s2, strict=True) is False
    assert s1.series_equal(s2, null_equal=False) is False

    df = pl.DataFrame(
        {"dtm": [datetime(2222, 2, 22, 22, 22, 22)]},
        schema_overrides={"dtm": Datetime(time_zone="UTC")},
    ).with_columns(
        s3=pl.col("dtm").dt.convert_time_zone("Europe/London"),
        s4=pl.col("dtm").dt.convert_time_zone("Asia/Tokyo"),
    )
    s3 = df["s3"].rename("b")
    s4 = df["s4"].rename("b")

    assert s3.series_equal(s4) is False
    assert s3.series_equal(s4, strict=True) is False
    assert s3.series_equal(s4, null_equal=False) is False
    assert s3.dt.convert_time_zone("Asia/Tokyo").series_equal(s4) is True


@pytest.mark.parametrize(
    "dtype",
    [pl.Int64, pl.Float64, pl.Utf8, pl.Boolean],
)
def test_eq_missing_list_and_primitive(dtype: PolarsDataType) -> None:
    s1 = pl.Series([None, None], dtype=dtype)
    s2 = pl.Series([None, None], dtype=pl.List(dtype))

    expected = pl.Series([True, True])
    assert_series_equal(s1.eq_missing(s2), expected)


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
    assert_series_equal((a & b), pl.Series([False, False, True]))
    assert_series_equal((a | b), pl.Series([True, True, True]))
    assert_series_equal((a ^ b), pl.Series([True, True, False]))
    assert_series_equal((~a), pl.Series([False, True, False]))

    # rand/rxor/ror we trigger by casting the left hand to a list here in the test
    # Note that the type annotations only allow Series to be passed in, but there is
    # specific code to deal with non-Series inputs.
    assert_series_equal(
        (True & a),  # type: ignore[operator]
        pl.Series([True, False, True]),
    )
    assert_series_equal(
        (True | a),  # type: ignore[operator]
        pl.Series([True, True, True]),
    )
    assert_series_equal(
        (True ^ a),  # type: ignore[operator]
        pl.Series([False, True, False]),
    )


def test_bitwise_floats_invert() -> None:
    s = pl.Series([2.0, 3.0, 0.0])

    with pytest.raises(pl.SchemaError):
        ~s


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
    assert_series_equal(a, b)

    a = pl.Series("name", ["ham", "foo", "bar"])
    assert_series_equal((a == "ham"), pl.Series("name", [True, False, False]))

    a = pl.Series("name", [[1], [1, 2], [2, 3]])
    assert_series_equal((a == [1]), pl.Series("name", [True, False, False]))


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
    expected = pl.Series([1, 0]) if s.dtype == Int64 else pl.Series([1.0, 0.5])
    assert_series_equal(1 // a, expected)
    # modulo
    assert ((1 % a) == [0, 1]).sum() == 2
    assert ((a % 1) == [0, 0]).sum() == 2
    # negate
    assert (-a == [-1, -2]).sum() == 2
    # unary plus
    assert (+a == a).all()
    # wrong dtypes in rhs operands
    assert ((1.0 - a) == [0.0, -1.0]).sum() == 2
    assert ((1.0 / a) == [1.0, 0.5]).sum() == 2
    assert ((1.0 * a) == [1, 2]).sum() == 2
    assert ((1.0 + a) == [2, 3]).sum() == 2
    assert ((1.0 % a) == [0, 1]).sum() == 2


def test_arithmetic_datetime() -> None:
    a = pl.Series("a", [datetime(2021, 1, 1)])
    with pytest.raises(TypeError):
        a // 2
    with pytest.raises(TypeError):
        a / 2
    with pytest.raises(TypeError):
        a * 2
    with pytest.raises(TypeError):
        a % 2
    with pytest.raises(TypeError):
        a**2
    with pytest.raises(TypeError):
        2 / a
    with pytest.raises(TypeError):
        2 // a
    with pytest.raises(TypeError):
        2 * a
    with pytest.raises(TypeError):
        2 % a
    with pytest.raises(TypeError):
        2**a

    with pytest.raises(TypeError):
        +a


def test_arithmetic_string() -> None:
    a = pl.Series("a", [""])
    with pytest.raises(TypeError):
        +a


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
    assert_series_equal(a**None, pl.Series([None] * len(a), dtype=Float64))
    with pytest.raises(TypeError):
        c**2
    with pytest.raises(pl.ColumnNotFoundError):
        a ** "hi"  # type: ignore[operator]

    # rpow
    assert_series_equal(2.0**a, pl.Series("literal", [2.0, 4.0], dtype=Float64))
    assert_series_equal(2**b, pl.Series("literal", [None, 4.0], dtype=Float64))
    with pytest.raises(TypeError):
        2**c
    with pytest.raises(pl.ColumnNotFoundError):
        "hi" ** a

    # Series.pow() method
    assert_series_equal(a.pow(2), pl.Series([1.0, 4.0], dtype=Float64))


def test_add_string() -> None:
    s = pl.Series(["hello", "weird"])
    result = s + " world"
    assert_series_equal(result, pl.Series(["hello world", "weird world"]))

    result = "pfx:" + s
    assert_series_equal(result, pl.Series(["pfx:hello", "pfx:weird"]))


@pytest.mark.parametrize(
    ("data", "expected_dtype"),
    [
        (100, pl.Int64),
        (8.5, pl.Float64),
        ("서울특별시", pl.Utf8),
        (date.today(), pl.Date),
        (datetime.now(), pl.Datetime("us")),
        (time(23, 59, 59), pl.Time),
        (timedelta(hours=7, seconds=123), pl.Duration("us")),
    ],
)
def test_unknown_dtype(data: Any, expected_dtype: PolarsDataType) -> None:
    # if given 'Unknown', should be able to infer the correct dtype
    s = pl.Series([data], dtype=Unknown)
    assert s.dtype == expected_dtype
    assert s.to_list() == [data]


def test_various() -> None:
    a = pl.Series("a", [1, 2])
    assert a.is_null().sum() == 0
    assert a.name == "a"

    a = a.rename("b")
    assert a.name == "b"
    assert a.len() == 2
    assert len(a) == 2

    a.append(a.clone())
    assert_series_equal(a, pl.Series("b", [1, 2, 1, 2]))

    a = pl.Series("a", range(20))
    assert a.head(5).len() == 5
    assert a.tail(5).len() == 5
    assert (a.head(5) != a.tail(5)).all()

    a = pl.Series("a", [2, 1, 4])
    a.sort(in_place=True)
    assert_series_equal(a, pl.Series("a", [1, 2, 4]))
    a = pl.Series("a", [2, 1, 1, 4, 4, 4])
    assert_series_equal(a.arg_unique(), pl.Series("a", [0, 1, 3], dtype=UInt32))

    assert_series_equal(a.take([2, 3]), pl.Series("a", [1, 4]))


def test_series_dtype_is() -> None:
    s = pl.Series("s", [1, 2, 3])

    assert s.dtype.is_numeric()
    assert s.dtype.is_integer()
    assert s.dtype.is_signed_integer()
    assert not s.dtype.is_unsigned_integer()
    assert (s * 0.99).dtype.is_float()

    s = pl.Series("s", [1, 2, 3], dtype=pl.UInt8)
    assert s.dtype.is_numeric()
    assert s.dtype.is_integer()
    assert not s.dtype.is_signed_integer()
    assert s.dtype.is_unsigned_integer()

    s = pl.Series("bool", [True, None, False])
    assert not s.dtype.is_numeric()

    s = pl.Series("s", ["testing..."])
    assert s.is_utf8()

    s = pl.Series("s", [], dtype=pl.Decimal(precision=20, scale=15))
    assert not s.dtype.is_float()
    assert s.dtype.is_numeric()
    assert s.is_empty()

    s = pl.Series("s", [], dtype=pl.Datetime("ms", time_zone="UTC"))
    assert s.dtype.is_temporal()


def test_series_is_dtype_deprecated() -> None:
    s = pl.Series([1.0, 2.0])
    with pytest.deprecated_call():
        assert s.is_float() is True
    with pytest.deprecated_call():
        assert s.is_numeric() is True
    with pytest.deprecated_call():
        assert s.is_integer() is False
    with pytest.deprecated_call():
        assert s.is_temporal() is False


def test_series_head_tail_limit() -> None:
    s = pl.Series(range(10))

    assert_series_equal(s.head(5), pl.Series(range(5)))
    assert_series_equal(s.limit(5), s.head(5))
    assert_series_equal(s.tail(5), pl.Series(range(5, 10)))

    # check if it doesn't fail when out of bounds
    assert s.head(100).len() == 10
    assert s.limit(100).len() == 10
    assert s.tail(100).len() == 10

    # negative values
    assert_series_equal(s.head(-7), pl.Series(range(3)))
    assert s.head(-2).len() == 8
    assert_series_equal(s.tail(-8), pl.Series(range(8, 10)))
    assert s.head(-6).len() == 4

    # negative values out of bounds
    assert s.head(-12).len() == 0
    assert s.limit(-12).len() == 0
    assert s.tail(-12).len() == 0


def test_filter_ops() -> None:
    a = pl.Series("a", range(20))
    assert a.filter(a > 1).len() == 18
    assert a.filter(a < 1).len() == 1
    assert a.filter(a <= 1).len() == 2
    assert a.filter(a >= 1).len() == 19
    assert a.filter(a == 1).len() == 1
    assert a.filter(a != 1).len() == 19


def test_cast() -> None:
    a = pl.Series("a", range(20))

    assert a.cast(pl.Float32).dtype == pl.Float32
    assert a.cast(pl.Float64).dtype == pl.Float64
    assert a.cast(pl.Int32).dtype == pl.Int32
    assert a.cast(pl.UInt32).dtype == pl.UInt32
    assert a.cast(pl.Datetime).dtype == pl.Datetime
    assert a.cast(pl.Date).dtype == pl.Date

    # display failed values, GH#4706
    with pytest.raises(ComputeError, match="foobar"):
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
            vals_b = [(None if x is None else x.tolist()) for x in b]
        else:
            vals_b = b.replace({np.nan: None}).values.tolist()

        assert vals_b == test_data

        try:
            c = a.to_pandas(use_pyarrow_extension_array=True)
            assert a.name == c.name
            assert c.isnull().sum() == 1
            vals_c = [None if x is pd.NA else x for x in c.tolist()]
            assert vals_c == test_data
        except ModuleNotFoundError:
            # Skip test if pandas>=1.5.0 or Pyarrow>=8.0.0 is not installed.
            pass


def test_series_to_list() -> None:
    s = pl.Series("a", range(20))
    result = s.to_list()
    assert isinstance(result, list)
    assert len(result) == 20

    a = pl.Series("a", [1, None, 2])
    assert a.null_count() == 1
    assert a.to_list() == [1, None, 2]


def test_series_to_list_use_pyarrow_deprecated() -> None:
    s = pl.Series("a", range(20))
    with pytest.deprecated_call():
        result = s.to_list(use_pyarrow=True)
    assert isinstance(result, list)
    assert len(result) == 20


def test_to_struct() -> None:
    s = pl.Series("nums", ["12 34", "56 78", "90 00"]).str.extract_all(r"\d+")

    assert s.list.to_struct().struct.fields == ["field_0", "field_1"]
    assert s.list.to_struct(fields=lambda idx: f"n{idx:02}").struct.fields == [
        "n00",
        "n01",
    ]
    assert_frame_equal(
        s.list.to_struct(fields=["one", "two"]).struct.unnest(),
        pl.DataFrame({"one": ["12", "56", "90"], "two": ["34", "78", "00"]}),
    )


def test_sort() -> None:
    a = pl.Series("a", [2, 1, 3])
    assert_series_equal(a.sort(), pl.Series("a", [1, 2, 3]))
    assert_series_equal(a.sort(descending=True), pl.Series("a", [3, 2, 1]))


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

    b = pl.Series("b", [1.0, 2.0, 3.0, None])
    out = b.to_arrow()
    assert out == pa.array([1.0, 2.0, 3.0, None])

    c = pl.Series("c", ["A", "BB", "CCC", None])
    out = c.to_arrow()
    assert out == pa.array(["A", "BB", "CCC", None], type=pa.large_string())
    assert_series_equal(pl.from_arrow(out), c.rename(""))  # type: ignore[arg-type]

    out = c.to_frame().to_arrow()["c"]
    assert isinstance(out, (pa.Array, pa.ChunkedArray))
    assert_series_equal(pl.from_arrow(out), c)  # type: ignore[arg-type]
    assert_series_equal(pl.from_arrow(out, schema=["x"]), c.rename("x"))  # type: ignore[arg-type]

    d = pl.Series("d", [None, None, None], pl.Null)
    out = d.to_arrow()
    assert out == pa.nulls(3)

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
    a = pl.Series("a", [1.0, 2.5, 3.0])
    assert isinstance(a.view(), np.ndarray)
    assert np.all(a.view() == np.array([1.0, 2.5, 3.0]))

    b = pl.Series("b", [1, 2, None])
    assert b.has_validity()
    with pytest.raises(AssertionError):
        b.view()

    assert np.all(b[:2].view() == np.array([1, 2]))
    assert not b[:2].has_validity()


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

    # Test if nulls propagate through ufuncs
    a3 = pl.Series("a", [None, None, 3, 3])
    b3 = pl.Series("b", [None, 3, None, 3])
    assert_series_equal(
        cast(pl.Series, np.maximum(a3, b3)), pl.Series("a", [None, None, None, 3])
    )


def test_numpy_string_array() -> None:
    s_utf8 = pl.Series("a", ["aa", "bb", "cc", "dd"], dtype=pl.Utf8)
    assert_array_equal(
        np.char.capitalize(s_utf8),
        np.array(["Aa", "Bb", "Cc", "Dd"], dtype="<U2"),
    )


def test_get() -> None:
    a = pl.Series("a", [1, 2, 3])
    pos_idxs = pl.Series("idxs", [2, 0, 1, 0], dtype=pl.Int8)
    neg_and_pos_idxs = pl.Series(
        "neg_and_pos_idxs", [-2, 1, 0, -1, 2, -3], dtype=pl.Int8
    )
    empty_idxs = pl.Series("idxs", [], dtype=pl.Int8)
    empty_ints: list[int] = []
    assert a[0] == 1
    assert a[:2].to_list() == [1, 2]
    assert a[range(1)].to_list() == [1]
    assert a[range(0, 4, 2)].to_list() == [1, 3]
    assert a[:0].to_list() == []
    assert a[empty_ints].to_list() == []
    assert a[neg_and_pos_idxs.to_list()].to_list() == [2, 2, 1, 3, 3, 1]
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
        assert a[empty_idxs.cast(dtype)].to_list() == []
        assert a[empty_idxs.cast(dtype).to_numpy()].to_list() == []

    for dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        nps = a[neg_and_pos_idxs.cast(dtype).to_numpy()]
        assert nps.to_list() == [2, 2, 1, 3, 3, 1]


def test_set() -> None:
    a = pl.Series("a", [True, False, True])
    mask = pl.Series("msk", [True, False, True])
    a[mask] = False
    assert_series_equal(a, pl.Series("a", [False] * 3))


def test_set_value_as_list_fail() -> None:
    # only allowed for numerical physical types
    s = pl.Series("a", [1, 2, 3])
    s[[0, 2]] = [4, 5]
    assert s.to_list() == [4, 2, 5]

    # for other types it is not allowed
    s = pl.Series("a", ["a", "b", "c"])
    with pytest.raises(TypeError):
        s[[0, 1]] = ["d", "e"]

    s = pl.Series("a", [True, False, False])
    with pytest.raises(TypeError):
        s[[0, 1]] = [True, False]


@pytest.mark.parametrize("key", [True, False, 1.0])
def test_set_invalid_key(key: Any) -> None:
    s = pl.Series("a", [1, 2, 3])
    with pytest.raises(TypeError):
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


@pytest.mark.parametrize("idx", [[0, 2], (0, 2)])
def test_set_list_and_tuple(idx: list[int] | tuple[int]) -> None:
    a = pl.Series("a", [1, 2, 3])
    a[idx] = 4
    assert_series_equal(a, pl.Series("a", [4, 2, 4]))


def test_init_nested_tuple() -> None:
    s1 = pl.Series("s", (1, 2, 3))
    assert s1.to_list() == [1, 2, 3]

    s2 = pl.Series("s", ((1, 2, 3),), dtype=pl.List(pl.UInt8))
    assert s2.to_list() == [[1, 2, 3]]
    assert s2.dtype == pl.List(pl.UInt8)

    s3 = pl.Series("s", ((1, 2, 3), (1, 2, 3)), dtype=pl.List(pl.Int32))
    assert s3.to_list() == [[1, 2, 3], [1, 2, 3]]
    assert s3.dtype == pl.List(pl.Int32)


def test_fill_null() -> None:
    s = pl.Series("a", [1, 2, None])
    assert_series_equal(s.fill_null(strategy="forward"), pl.Series("a", [1, 2, 2]))
    assert_series_equal(s.fill_null(14), pl.Series("a", [1, 2, 14], dtype=Int64))

    a = pl.Series("a", [0.0, 1.0, None, 2.0, None, 3.0])

    assert a.fill_null(0).to_list() == [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
    assert a.fill_null(strategy="zero").to_list() == [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
    assert a.fill_null(strategy="max").to_list() == [0.0, 1.0, 3.0, 2.0, 3.0, 3.0]
    assert a.fill_null(strategy="min").to_list() == [0.0, 1.0, 0.0, 2.0, 0.0, 3.0]
    assert a.fill_null(strategy="one").to_list() == [0.0, 1.0, 1.0, 2.0, 1.0, 3.0]
    assert a.fill_null(strategy="forward").to_list() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    assert a.fill_null(strategy="backward").to_list() == [0.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert a.fill_null(strategy="mean").to_list() == [0.0, 1.0, 1.5, 2.0, 1.5, 3.0]

    b = pl.Series("b", ["a", None, "c", None, "e"])
    assert b.fill_null(strategy="min").to_list() == ["a", "a", "c", "a", "e"]
    assert b.fill_null(strategy="max").to_list() == ["a", "e", "c", "e", "e"]

    c = pl.Series("c", [b"a", None, b"c", None, b"e"])
    assert c.fill_null(strategy="min").to_list() == [b"a", b"a", b"c", b"a", b"e"]
    assert c.fill_null(strategy="max").to_list() == [b"a", b"e", b"c", b"e", b"e"]

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
    ).to_dict(as_series=False) == {
        "i32": [1, 2, None],
        "i64": [1, 2, 0],
        "f32": [1.0, 2.0, None],
        "cat": ["a", "b", "bar"],
        "str": ["a", "b", "bar"],
        "bool": [True, True, False],
    }

    assert df.fill_null(0, matches_supertype=True).fill_null("bar").fill_null(
        False
    ).to_dict(as_series=False) == {
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

    assert out.to_dict(as_series=False) == {
        "a": [1, 3, 2, 3],
        "u8": [1, 3, 2, 3],
        "u16": [1, 3, 2, 3],
        "u32": [1, 3, 2, 3],
        "u64": [1, 3, 2, 3],
    }
    assert out.dtypes == [pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]


def test_utf8_series_min_max_10674() -> None:
    utf8_series = pl.Series("b", ["a", None, "c", None, "e"], dtype=pl.Utf8)
    assert utf8_series.min() == "a"
    assert utf8_series.max() == "e"
    assert utf8_series.sort(descending=False).min() == "a"
    assert utf8_series.sort(descending=True).max() == "e"


def test_fill_nan() -> None:
    nan = float("nan")
    a = pl.Series("a", [1.0, nan, 2.0, nan, 3.0])
    assert_series_equal(a.fill_nan(None), pl.Series("a", [1.0, None, 2.0, None, 3.0]))
    assert_series_equal(a.fill_nan(0), pl.Series("a", [1.0, 0.0, 2.0, 0.0, 3.0]))


def test_map_elements() -> None:
    with pytest.warns(PolarsInefficientMapWarning):
        a = pl.Series("a", [1, 2, None])
        b = a.map_elements(lambda x: x**2)
        assert list(b) == [1, 4, None]

    with pytest.warns(PolarsInefficientMapWarning):
        a = pl.Series("a", ["foo", "bar", None])
        b = a.map_elements(lambda x: x + "py")
        assert list(b) == ["foopy", "barpy", None]

    b = a.map_elements(lambda x: len(x), return_dtype=pl.Int32)
    assert list(b) == [3, 3, None]

    b = a.map_elements(lambda x: len(x))
    assert list(b) == [3, 3, None]

    # just check that it runs (somehow problem with conditional compilation)
    a = pl.Series("a", [2, 2, 3]).cast(pl.Datetime)
    a.map_elements(lambda x: x)
    a = pl.Series("a", [2, 2, 3]).cast(pl.Date)
    a.map_elements(lambda x: x)


def test_object() -> None:
    vals = [[12], "foo", 9]
    a = pl.Series("a", vals)
    assert a.dtype == pl.Object
    assert a.to_list() == vals
    assert a[1] == "foo"


def test_shape() -> None:
    s = pl.Series([1, 2, 3])
    assert s.shape == (3,)


@pytest.mark.parametrize("arrow_available", [True, False])
def test_create_list_series(arrow_available: bool, monkeypatch: Any) -> None:
    monkeypatch.setattr(pl.series.series, "_PYARROW_AVAILABLE", arrow_available)
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
    for n in (0, 2, 5):
        empty_a = a.clear(n)
        assert a.dtype == empty_a.dtype
        assert a.name == empty_a.name
        assert len(empty_a) == n

    with pytest.raises(TypeError, match="ambiguous"):
        not empty_a


def test_describe() -> None:
    num_s = pl.Series([1, 2, 3])
    float_s = pl.Series([1.3, 4.6, 8.9])
    str_s = pl.Series(["abc", "pqr", "xyz"])
    bool_s = pl.Series([True, False, None, True, True])
    date_s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
    empty_s = pl.Series(np.empty(0))

    assert dict(num_s.describe().rows()) == {  # type: ignore[arg-type]
        "count": 3.0,
        "mean": 2.0,
        "null_count": 0.0,
        "std": 1.0,
        "min": 1.0,
        "25%": 1.0,
        "50%": 2.0,
        "75%": 3.0,
        "max": 3.0,
    }
    assert dict(float_s.describe().rows()) == {  # type: ignore[arg-type]
        "count": 3.0,
        "mean": 4.933333333333334,
        "null_count": 0.0,
        "std": 3.8109491381194442,
        "min": 1.3,
        "25%": 1.3,
        "50%": 4.6,
        "75%": 8.9,
        "max": 8.9,
    }
    assert dict(str_s.describe().rows()) == {  # type: ignore[arg-type]
        "count": 3,
        "null_count": 0,
        "unique": 3,
    }
    assert dict(bool_s.describe().rows()) == {  # type: ignore[arg-type]
        "count": 5,
        "null_count": 1,
        "sum": 3,
    }
    assert dict(date_s.describe().rows()) == {  # type: ignore[arg-type]
        "count": "3",
        "min": "2021-01-01",
        "50%": "2021-01-02",
        "max": "2021-01-03",
        "null_count": "0",
    }

    with pytest.raises(ValueError):
        assert empty_s.describe()


def test_round() -> None:
    a = pl.Series("f", [1.003, 2.003])
    b = a.round(2)
    assert b.to_list() == [1.00, 2.00]

    b = a.round()
    assert b.to_list() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("series", "digits", "expected_result"),
    [
        pytest.param(pl.Series([1.234, 0.1234]), 2, pl.Series([1.2, 0.12]), id="f64"),
        pytest.param(
            pl.Series([1.234, 0.1234]).cast(pl.Float32),
            2,
            pl.Series([1.2, 0.12]).cast(pl.Float32),
            id="f32",
        ),
        pytest.param(pl.Series([123400, 1234]), 2, pl.Series([120000, 1200]), id="i64"),
        pytest.param(
            pl.Series([123400, 1234]).cast(pl.Int32),
            2,
            pl.Series([120000, 1200]).cast(pl.Int32),
            id="i32",
        ),
        pytest.param(
            pl.Series([0.0]), 2, pl.Series([0.0]), id="0 should remain the same"
        ),
    ],
)
def test_round_sig_figs(
    series: pl.Series, digits: int, expected_result: pl.Series
) -> None:
    result = series.round_sig_figs(digits=digits)
    assert_series_equal(result, expected_result)


def test_round_sig_figs_raises_exc() -> None:
    with pytest.raises(polars.exceptions.InvalidOperationError):
        pl.Series([1.234, 0.1234]).round_sig_figs(digits=0)


def test_apply_list_out() -> None:
    s = pl.Series("count", [3, 2, 2])
    out = s.map_elements(lambda val: pl.repeat(val, val, eager=True))
    assert out[0].to_list() == [3, 3, 3]
    assert out[1].to_list() == [2, 2]
    assert out[2].to_list() == [2, 2]


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
    assert (
        pl.Series(["foo", "bar", "buz", "bar"], dtype=pl.Categorical).mode().item()
        == "bar"
    )
    assert pl.Series([1.0, 2.0, 3.0, 2.0]).mode().item() == 2.0

    # sorted data
    assert pl.int_range(0, 3, eager=True).mode().to_list() == [2, 1, 0]


def test_rank() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert_series_equal(
        s.rank("dense"), pl.Series("a", [2, 3, 4, 3, 3, 4, 1], dtype=UInt32)
    )

    df = pl.DataFrame([s])
    assert df.select(pl.col("a").rank("dense"))["a"].to_list() == [2, 3, 4, 3, 3, 4, 1]

    assert_series_equal(
        s.rank("dense", descending=True),
        pl.Series("a", [3, 2, 1, 2, 2, 1, 4], dtype=UInt32),
    )

    assert s.rank(method="average").dtype == pl.Float64
    assert s.rank(method="max").dtype == pl.get_index_type()


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
    assert_series_equal(s.pct_change(2), expected)
    assert_series_equal(s.pct_change(pl.Series([2])), expected)
    # negative
    assert pl.Series(range(5)).pct_change(-1).to_list() == [
        -1.0,
        -0.5,
        -0.3333333333333333,
        -0.25,
        None,
    ]


def test_skew() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert s.skew(bias=True) == pytest.approx(-0.5953924651018018)
    assert s.skew(bias=False) == pytest.approx(-0.7717168360221258)

    df = pl.DataFrame([s])
    assert np.isclose(
        df.select(pl.col("a").skew(bias=False))["a"][0], -0.7717168360221258
    )


def test_kurtosis() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    expected = -0.6406250000000004

    assert s.kurtosis() == pytest.approx(expected)
    df = pl.DataFrame([s])
    assert np.isclose(df.select(pl.col("a").kurtosis())["a"][0], expected)


def test_arr_lengths() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    assert_series_equal(s.list.len(), pl.Series("a", [2, 3], dtype=UInt32))
    df = pl.DataFrame([s])
    assert_series_equal(
        df.select(pl.col("a").list.len())["a"], pl.Series("a", [2, 3], dtype=UInt32)
    )


def test_arr_arithmetic() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    assert_series_equal(s.list.sum(), pl.Series("a", [3, 6]))
    assert_series_equal(s.list.mean(), pl.Series("a", [1.5, 2.0]))
    assert_series_equal(s.list.max(), pl.Series("a", [2, 3]))
    assert_series_equal(s.list.min(), pl.Series("a", [1, 1]))


def test_arr_ordering() -> None:
    s = pl.Series("a", [[2, 1], [1, 3, 2]])
    assert_series_equal(s.list.sort(), pl.Series("a", [[1, 2], [1, 2, 3]]))
    assert_series_equal(s.list.reverse(), pl.Series("a", [[1, 2], [2, 3, 1]]))


def test_arr_unique() -> None:
    s = pl.Series("a", [[2, 1], [1, 2, 2]])
    result = s.list.unique()
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


def test_cbrt() -> None:
    s = pl.Series("a", [1, 2])
    assert_series_equal(s.cbrt(), pl.Series("a", [1.0, np.cbrt(2)]))
    df = pl.DataFrame([s])
    assert_series_equal(
        df.select(pl.col("a").cbrt())["a"], pl.Series("a", [1.0, np.cbrt(2)])
    )


def test_range() -> None:
    s1 = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    assert_series_equal(s1[2:5], s1[range(2, 5)])

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
    assert_frame_equal(df[2:5], df[range(2, 5)])


def test_strict_cast() -> None:
    with pytest.raises(ComputeError):
        pl.Series("a", [2**16]).cast(dtype=pl.Int16, strict=True)
    with pytest.raises(ComputeError):
        pl.DataFrame({"a": [2**16]}).select([pl.col("a").cast(pl.Int16, strict=True)])


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
    with pytest.raises(TypeError, match="ambiguous"):
        a and b  # type: ignore[redundant-expr]

    with pytest.raises(TypeError, match="ambiguous"):
        a or b  # type: ignore[redundant-expr]


def test_to_numpy(monkeypatch: Any) -> None:
    for writable in [False, True]:
        for flag in [False, True]:
            monkeypatch.setattr(pl.series.series, "_PYARROW_AVAILABLE", flag)

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
        monkeypatch.setattr(pl.series.series, "_PYARROW_AVAILABLE", False)
        a = pl.Series("a", vals)
        monkeypatch.setattr(pl.series.series, "_PYARROW_AVAILABLE", True)
        b = pl.Series("a", vals)
        assert_series_equal(a, b)
        assert a.to_list() == vals


def test_comparisons_int_series_to_float() -> None:
    srs_int = pl.Series([1, 2, 3, 4])

    assert_series_equal(srs_int - 1.0, pl.Series([0.0, 1.0, 2.0, 3.0]))
    assert_series_equal(srs_int + 1.0, pl.Series([2.0, 3.0, 4.0, 5.0]))
    assert_series_equal(srs_int * 2.0, pl.Series([2.0, 4.0, 6.0, 8.0]))
    assert_series_equal(srs_int / 2.0, pl.Series([0.5, 1.0, 1.5, 2.0]))
    assert_series_equal(srs_int % 2.0, pl.Series([1.0, 0.0, 1.0, 0.0]))
    assert_series_equal(4.0 % srs_int, pl.Series([0.0, 0.0, 1.0, 0.0]))

    assert_series_equal(srs_int // 2.0, pl.Series([0.0, 1.0, 1.0, 2.0]))
    assert_series_equal(srs_int < 3.0, pl.Series([True, True, False, False]))
    assert_series_equal(srs_int <= 3.0, pl.Series([True, True, True, False]))
    assert_series_equal(srs_int > 3.0, pl.Series([False, False, False, True]))
    assert_series_equal(srs_int >= 3.0, pl.Series([False, False, True, True]))
    assert_series_equal(srs_int == 3.0, pl.Series([False, False, True, False]))
    assert_series_equal(srs_int - True, pl.Series([0, 1, 2, 3]))


def test_comparisons_int_series_to_float_scalar() -> None:
    srs_int = pl.Series([1, 2, 3, 4])

    assert_series_equal(srs_int < 1.5, pl.Series([True, False, False, False]))
    assert_series_equal(srs_int > 1.5, pl.Series([False, True, True, True]))


def test_comparisons_datetime_series_to_date_scalar() -> None:
    srs_date = pl.Series([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
    dt = datetime(2023, 1, 1, 12, 0, 0)

    assert_series_equal(srs_date < dt, pl.Series([True, False, False]))
    assert_series_equal(srs_date > dt, pl.Series([False, True, True]))


def test_comparisons_float_series_to_int() -> None:
    srs_float = pl.Series([1.0, 2.0, 3.0, 4.0])

    assert_series_equal(srs_float - 1, pl.Series([0.0, 1.0, 2.0, 3.0]))
    assert_series_equal(srs_float + 1, pl.Series([2.0, 3.0, 4.0, 5.0]))
    assert_series_equal(srs_float * 2, pl.Series([2.0, 4.0, 6.0, 8.0]))
    assert_series_equal(srs_float / 2, pl.Series([0.5, 1.0, 1.5, 2.0]))
    assert_series_equal(srs_float % 2, pl.Series([1.0, 0.0, 1.0, 0.0]))
    assert_series_equal(4 % srs_float, pl.Series([0.0, 0.0, 1.0, 0.0]))

    assert_series_equal(srs_float // 2, pl.Series([0.0, 1.0, 1.0, 2.0]))
    assert_series_equal(srs_float < 3, pl.Series([True, True, False, False]))
    assert_series_equal(srs_float <= 3, pl.Series([True, True, True, False]))
    assert_series_equal(srs_float > 3, pl.Series([False, False, False, True]))
    assert_series_equal(srs_float >= 3, pl.Series([False, False, True, True]))
    assert_series_equal(srs_float == 3, pl.Series([False, False, True, False]))
    assert_series_equal(srs_float - True, pl.Series([0.0, 1.0, 2.0, 3.0]))


def test_comparisons_bool_series_to_int() -> None:
    srs_bool = pl.Series([True, False])

    # (native bool comparison should work...)
    for t, f in ((True, False), (False, True)):
        assert list(srs_bool == t) == list(srs_bool != f) == [t, f]

    # TODO: do we want this to work?
    assert_series_equal(srs_bool / 1, pl.Series([True, False], dtype=Float64))
    match = (
        r"cannot do arithmetic with series of dtype: Boolean"
        r" and argument of type: 'bool'"
    )
    with pytest.raises(TypeError, match=match):
        srs_bool - 1
    with pytest.raises(TypeError, match=match):
        srs_bool + 1
    match = (
        r"cannot do arithmetic with series of dtype: Boolean"
        r" and argument of type: 'bool'"
    )
    with pytest.raises(TypeError, match=match):
        srs_bool % 2
    with pytest.raises(TypeError, match=match):
        srs_bool * 1

    from operator import ge, gt, le, lt

    for op in (ge, gt, le, lt):
        for scalar in (0, 1.0, True, False):
            with pytest.raises(
                TypeError,
                match=r"'\W{1,2}' not supported .* 'Series' and '(int|bool|float)'",
            ):
                op(srs_bool, scalar)


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
    expected = pl.DataFrame(
        {"a_1": [1, 0, 0], "a_2": [0, 1, 0], "a_3": [0, 0, 1]},
        schema={"a_1": pl.UInt8, "a_2": pl.UInt8, "a_3": pl.UInt8},
    )
    assert_frame_equal(result, expected)


def test_chunk_lengths() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    # this is a Series with one chunk, of length 4
    assert s.n_chunks() == 1
    assert s.chunk_lengths() == [4]


def test_limit() -> None:
    s = pl.Series("a", [1, 2, 3])
    assert_series_equal(s.limit(2), pl.Series("a", [1, 2]))


def test_filter() -> None:
    s = pl.Series("a", [1, 2, 3])
    mask = pl.Series("", [True, False, True])
    assert_series_equal(s.filter(mask), pl.Series("a", [1, 3]))

    assert_series_equal(s.filter([True, False, True]), pl.Series("a", [1, 3]))


def test_take_every() -> None:
    s = pl.Series("a", [1, 2, 3, 4])
    assert_series_equal(s.take_every(2), pl.Series("a", [1, 3]))


def test_arg_sort() -> None:
    s = pl.Series("a", [5, 3, 4, 1, 2])
    expected = pl.Series("a", [3, 4, 1, 2, 0], dtype=UInt32)

    assert_series_equal(s.arg_sort(), expected)

    expected_descending = pl.Series("a", [0, 2, 1, 4, 3], dtype=UInt32)
    assert_series_equal(s.arg_sort(descending=True), expected_descending)


def test_arg_min_and_arg_max() -> None:
    # numerical no null.
    s = pl.Series([5, 3, 4, 1, 2])
    assert s.arg_min() == 3
    assert s.arg_max() == 0

    # numerical has null.
    s = pl.Series([None, 5, 1])
    assert s.arg_min() == 2
    assert s.arg_max() == 1

    # numerical all null.
    s = pl.Series([None, None], dtype=Int32)
    assert s.arg_min() is None
    assert s.arg_max() is None

    # boolean no null.
    s = pl.Series([True, False])
    assert s.arg_min() == 1
    assert s.arg_max() == 0
    s = pl.Series([True, True])
    assert s.arg_min() == 0
    assert s.arg_max() == 0
    s = pl.Series([False, False])
    assert s.arg_min() == 0
    assert s.arg_max() == 0

    # boolean has null.
    s = pl.Series([None, True, False, True])
    assert s.arg_min() == 2
    assert s.arg_max() == 1
    s = pl.Series([None, True, True])
    assert s.arg_min() == 1
    assert s.arg_max() == 1
    s = pl.Series([None, False, False])
    assert s.arg_min() == 1
    assert s.arg_max() == 1

    # boolean all null.
    s = pl.Series([None, None], dtype=pl.Boolean)
    assert s.arg_min() is None
    assert s.arg_max() is None

    # utf8 no null
    s = pl.Series(["a", "c", "b"])
    assert s.arg_min() == 0
    assert s.arg_max() == 1

    # utf8 has null
    s = pl.Series([None, "a", None, "b"])
    assert s.arg_min() == 1
    assert s.arg_max() == 3

    # utf8 all null
    s = pl.Series([None, None], dtype=pl.Utf8)
    assert s.arg_min() is None
    assert s.arg_max() is None

    # test ascending and descending series
    s = pl.Series([None, 1, 2, 3, 4, 5])
    s.sort(in_place=True)  # set ascending sorted flag
    assert s.flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s.arg_min() == 1
    assert s.arg_max() == 5
    s = pl.Series([None, 5, 4, 3, 2, 1])
    s.sort(descending=True, in_place=True)  # set descing sorted flag
    assert s.flags == {"SORTED_ASC": False, "SORTED_DESC": True}
    assert s.arg_min() == 5
    assert s.arg_max() == 1

    # test ascending and descending numerical series
    s = pl.Series([None, 1, 2, 3, 4, 5])
    s.sort(in_place=True)  # set ascending sorted flag
    assert s.flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s.arg_min() == 1
    assert s.arg_max() == 5
    s = pl.Series([None, 5, 4, 3, 2, 1])
    s.sort(descending=True, in_place=True)  # set descing sorted flag
    assert s.flags == {"SORTED_ASC": False, "SORTED_DESC": True}
    assert s.arg_min() == 5
    assert s.arg_max() == 1

    # test ascending and descending utf8 series
    s = pl.Series([None, "a", "b", "c", "d", "e"])
    s.sort(in_place=True)  # set ascending sorted flag
    assert s.flags == {"SORTED_ASC": True, "SORTED_DESC": False}
    assert s.arg_min() == 1
    assert s.arg_max() == 5
    s = pl.Series([None, "e", "d", "c", "b", "a"])
    s.sort(descending=True, in_place=True)  # set descing sorted flag
    assert s.flags == {"SORTED_ASC": False, "SORTED_DESC": True}
    assert s.arg_min() == 5
    assert s.arg_max() == 1

    # test numerical empty series
    s = pl.Series([], dtype=pl.Int32)
    assert s.arg_min() is None
    assert s.arg_max() is None

    # test boolean empty series
    s = pl.Series([], dtype=pl.Boolean)
    assert s.arg_min() is None
    assert s.arg_max() is None

    # test utf8 empty series
    s = pl.Series([], dtype=pl.Utf8)
    assert s.arg_min() is None
    assert s.arg_max() is None


def test_is_null_is_not_null() -> None:
    s = pl.Series("a", [1.0, 2.0, 3.0, None])
    assert_series_equal(s.is_null(), pl.Series("a", [False, False, False, True]))
    assert_series_equal(s.is_not_null(), pl.Series("a", [True, True, True, False]))


def test_is_finite_is_infinite() -> None:
    s = pl.Series("a", [1.0, 2.0, np.inf])
    assert_series_equal(s.is_finite(), pl.Series("a", [True, True, False]))
    assert_series_equal(s.is_infinite(), pl.Series("a", [False, False, True]))


def test_is_nan_is_not_nan() -> None:
    s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
    assert_series_equal(s.is_nan(), pl.Series("a", [False, False, False, True]))
    assert_series_equal(s.is_not_nan(), pl.Series("a", [True, True, True, False]))


def test_is_unique() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    assert_series_equal(s.is_unique(), pl.Series("a", [True, False, False, True]))

    # utf8
    assert pl.Series(["a", "b", "c", "a"]).is_duplicated().to_list() == [
        True,
        False,
        False,
        True,
    ]
    assert pl.Series(["a", "b", "c", "a"]).is_unique().to_list() == [
        False,
        True,
        True,
        False,
    ]


def test_is_duplicated() -> None:
    s = pl.Series("a", [1, 2, 2, 3])
    assert_series_equal(s.is_duplicated(), pl.Series("a", [False, True, True, False]))


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


def test_peak_max_peak_min() -> None:
    s = pl.Series("a", [4, 1, 3, 2, 5])
    result = s.peak_min()
    expected = pl.Series("a", [False, True, False, True, False])
    assert_series_equal(result, expected)

    result = s.peak_max()
    expected = pl.Series("a", [True, False, True, False, True])
    assert_series_equal(result, expected)


def test_shrink_to_fit() -> None:
    s = pl.Series("a", [4, 1, 3, 2, 5])
    sf = s.shrink_to_fit(in_place=True)
    assert sf is s

    s = pl.Series("a", [4, 1, 3, 2, 5])
    sf = s.shrink_to_fit(in_place=False)
    assert s is not sf


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
    assert_series_equal(out, expected)
    out = s.reshape((2, 2))
    assert_series_equal(out, expected)
    out = s.reshape((2, -1))
    assert_series_equal(out, expected)

    out = s.reshape((-1, 1))
    expected = pl.Series("a", [[1], [2], [3], [4]])
    assert_series_equal(out, expected)

    # test lazy_dispatch
    out = pl.select(pl.lit(s).reshape((-1, 1))).to_series()
    assert_series_equal(out, expected)

    # invalid (empty) dimensions
    with pytest.raises(ComputeError, match="reshape `dimensions` cannot be empty"):
        s.reshape(())


def test_init_categorical() -> None:
    with pl.StringCache():
        for values in [[None], ["foo", "bar"], [None, "foo", "bar"]]:
            expected = pl.Series("a", values, dtype=pl.Utf8).cast(pl.Categorical)
            a = pl.Series("a", values, dtype=pl.Categorical)
            assert_series_equal(a, expected)


def test_iter_nested_list() -> None:
    elems = list(pl.Series("s", [[1, 2], [3, 4]]))
    assert_series_equal(elems[0], pl.Series([1, 2]))
    assert_series_equal(elems[1], pl.Series([3, 4]))

    rev_elems = list(reversed(pl.Series("s", [[1, 2], [3, 4]])))
    assert_series_equal(rev_elems[0], pl.Series([3, 4]))
    assert_series_equal(rev_elems[1], pl.Series([1, 2]))


def test_iter_nested_struct() -> None:
    # note: this feels inconsistent with the above test for nested list, but
    # let's ensure the behaviour is codified before potentially modifying...
    elems = list(pl.Series("s", [{"a": 1, "b": 2}, {"a": 3, "b": 4}]))
    assert elems[0] == {"a": 1, "b": 2}
    assert elems[1] == {"a": 3, "b": 4}

    rev_elems = list(reversed(pl.Series("s", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])))
    assert rev_elems[0] == {"a": 3, "b": 4}
    assert rev_elems[1] == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "dtype",
    [
        pl.UInt8,
        pl.Float32,
        pl.Int32,
        pl.Boolean,
        pl.List(pl.Utf8),
        pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)]),
    ],
)
def test_nested_list_types_preserved(dtype: pl.DataType) -> None:
    srs = pl.Series([pl.Series([], dtype=dtype) for _ in range(5)])
    for srs_nested in srs:
        assert srs_nested.dtype == dtype


def test_log_exp() -> None:
    a = pl.Series("a", [1, 100, 1000])
    b = pl.Series("a", [0.0, 2.0, 3.0])
    assert_series_equal(a.log10(), b)

    expected = pl.Series("a", np.log(a.to_numpy()))
    assert_series_equal(a.log(), expected)

    expected = pl.Series("a", np.exp(b.to_numpy()))
    assert_series_equal(b.exp(), expected)

    expected = pl.Series("a", np.log1p(a.to_numpy()))
    assert_series_equal(a.log1p(), expected)


def test_to_physical() -> None:
    # casting an int result in an int
    s = pl.Series("a", [1, 2, 3])
    assert_series_equal(s.to_physical(), s)

    # casting a date results in an Int32
    s = pl.Series("a", [date(2020, 1, 1)] * 3)
    expected = pl.Series("a", [18262] * 3, dtype=Int32)
    assert_series_equal(s.to_physical(), expected)

    # casting a categorical results in a UInt32
    s = pl.Series(["cat1"]).cast(pl.Categorical)
    expected = pl.Series([0], dtype=UInt32)
    assert_series_equal(s.to_physical(), expected)

    # casting a List(Categorical) results in a List(UInt32)
    s = pl.Series([["cat1"]]).cast(pl.List(pl.Categorical))
    expected = pl.Series([[0]], dtype=pl.List(UInt32))
    assert_series_equal(s.to_physical(), expected)


def test_is_between_datetime() -> None:
    s = pl.Series("a", [datetime(2020, 1, 1, 10, 0, 0), datetime(2020, 1, 1, 20, 0, 0)])
    start = datetime(2020, 1, 1, 12, 0, 0)
    end = datetime(2020, 1, 1, 23, 0, 0)
    expected = pl.Series("a", [False, True])

    # only on the expression api
    result = s.to_frame().with_columns(pl.col("*").is_between(start, end)).to_series()
    assert_series_equal(result, expected)


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
    s = pl.Series("a", [0.0, math.pi, None, math.nan])
    expected = (
        pl.Series("a", getattr(np, f)(s.to_numpy()))
        .to_frame()
        .with_columns(pl.when(s.is_null()).then(None).otherwise(pl.col("a")).alias("a"))
        .to_series()
    )
    result = getattr(s, f)()
    assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_trigonometric_cot() -> None:
    # cotangent is not available in numpy...
    s = pl.Series("a", [0.0, math.pi, None, math.nan])
    expected = pl.Series("a", [math.inf, -8.1656e15, None, math.nan])
    assert_series_equal(s.cot(), expected)


def test_trigonometric_invalid_input() -> None:
    # String
    s = pl.Series("a", ["1", "2", "3"])
    with pytest.raises(pl.InvalidOperationError):
        s.sin()

    # Date
    s = pl.Series("a", [date(1990, 2, 28), date(2022, 7, 26)])
    with pytest.raises(pl.InvalidOperationError):
        s.cosh()


def test_ewm_mean() -> None:
    s = pl.Series([2, 5, 3])

    expected = pl.Series([2.0, 4.0, 3.4285714285714284])
    assert_series_equal(s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=True), expected)
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=False), expected
    )

    expected = pl.Series([2.0, 3.8, 3.421053])
    assert_series_equal(s.ewm_mean(com=2.0, adjust=True, ignore_nulls=True), expected)
    assert_series_equal(s.ewm_mean(com=2.0, adjust=True, ignore_nulls=False), expected)

    expected = pl.Series([2.0, 3.5, 3.25])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=True), expected
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=False), expected
    )

    s = pl.Series([2, 3, 5, 7, 4])

    expected = pl.Series([None, 2.666667, 4.0, 5.6, 4.774194])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=2, ignore_nulls=True), expected
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=2, ignore_nulls=False), expected
    )

    expected = pl.Series([None, None, 4.0, 5.6, 4.774194])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=3, ignore_nulls=True), expected
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, min_periods=3, ignore_nulls=False), expected
    )

    s = pl.Series([None, 1.0, 5.0, 7.0, None, 2.0, 5.0, 4])

    expected = pl.Series(
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
    assert_series_equal(s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=True), expected)
    expected = pl.Series(
        [
            None,
            1.0,
            3.666666666666667,
            5.571428571428571,
            5.571428571428571,
            3.08695652173913,
            4.2,
            4.092436974789916,
        ]
    )
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=True, ignore_nulls=False), expected
    )

    expected = pl.Series([None, 1.0, 3.0, 5.0, 5.0, 3.5, 4.25, 4.125])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=True), expected
    )

    expected = pl.Series([None, 1.0, 3.0, 5.0, 5.0, 3.0, 4.0, 4.0])
    assert_series_equal(
        s.ewm_mean(alpha=0.5, adjust=False, ignore_nulls=False), expected
    )


def test_ewm_mean_leading_nulls() -> None:
    for min_periods in [1, 2, 3]:
        assert (
            pl.Series([1, 2, 3, 4])
            .ewm_mean(com=3, min_periods=min_periods)
            .null_count()
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
    assert_series_equal(
        ewm_mean,
        pl.Series(
            [
                1.0,
                1.0,
                1.6666666666666665,
                1.6666666666666665,
                2.4285714285714284,
            ]
        ),
    )
    ewm_mean = series.ewm_mean(alpha=0.5, min_periods=2)
    assert_series_equal(
        ewm_mean,
        pl.Series(
            [
                None,
                None,
                1.6666666666666665,
                1.6666666666666665,
                2.4285714285714284,
            ]
        ),
    )


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

    with pytest.raises(ValueError, match="require `com` >= 0"):
        s.ewm_std(com=-0.5)

    with pytest.raises(ValueError, match="require `span` >= 1"):
        s.ewm_mean(span=0.5)

    with pytest.raises(ValueError, match="require `half_life` > 0"):
        s.ewm_var(half_life=0)

    for alpha in (-0.5, -0.0000001, 0.0, 1.0000001, 1.5):
        with pytest.raises(ValueError, match="require 0 < `alpha` <= 1"):
            s.ewm_std(alpha=alpha)


@pytest.mark.parametrize(
    ("const", "dtype"),
    [
        (1, pl.Int8),
        (4, pl.UInt32),
        (4.5, pl.Float32),
        (None, pl.Float64),
        ("白鵬翔", pl.Utf8),
        (date.today(), pl.Date),
        (datetime.now(), pl.Datetime("ns")),
        (time(23, 59, 59), pl.Time),
        (timedelta(hours=7, seconds=123), pl.Duration("ms")),
    ],
)
def test_extend_constant(const: Any, dtype: pl.PolarsDataType) -> None:
    s = pl.Series("s", [None], dtype=dtype)
    expected = pl.Series("s", [None, const, const, const], dtype=dtype)
    assert_series_equal(s.extend_constant(const, 3), expected)


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
    assert out == 2
    a = pl.Series("a", [None, 2, 3])
    out = a.product()
    assert out == 6
    a = pl.Series("a", [])
    out = a.product()
    assert out == 1
    a = pl.Series("a", [None, None])
    out = a.product()
    assert out == 1
    a = pl.Series("a", [3.0, None, float("nan")])
    out = a.product()
    assert math.isnan(out)


def test_ceil() -> None:
    s = pl.Series([1.8, 1.2, 3.0])
    expected = pl.Series([2.0, 2.0, 3.0])
    assert_series_equal(s.ceil(), expected)


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
        assert_series_equal(df1["d_offset"], df2["d_offset"])


def test_mean_overflow() -> None:
    arr = np.array([255] * (1 << 17), dtype="int16")
    assert arr.mean() == 255.0


def test_sign() -> None:
    # Integers
    a = pl.Series("a", [-9, -0, 0, 4, None])
    expected = pl.Series("a", [-1, 0, 0, 1, None])
    assert_series_equal(a.sign(), expected)

    # Floats
    a = pl.Series("a", [-9.0, -0.0, 0.0, 4.0, None])
    expected = pl.Series("a", [-1, 0, 0, 1, None])
    assert_series_equal(a.sign(), expected)

    # Invalid input
    a = pl.Series("a", [date(1950, 2, 1), date(1970, 1, 1), date(2022, 12, 12), None])
    with pytest.raises(pl.InvalidOperationError):
        a.sign()


def test_exp() -> None:
    s = pl.Series("a", [0.1, 0.01, None])
    expected = pl.Series("a", [1.1051709180756477, 1.010050167084168, None])
    assert_series_equal(s.exp(), expected)
    # test if we can run on empty series as well.
    assert s[:0].exp().to_list() == []


def test_cumulative_eval() -> None:
    s = pl.Series("values", [1, 2, 3, 4, 5])

    # evaluate expressions individually
    expr1 = pl.element().first()
    expr2 = pl.element().last() ** 2

    expected1 = pl.Series("values", [1, 1, 1, 1, 1])
    expected2 = pl.Series("values", [1.0, 4.0, 9.0, 16.0, 25.0])
    assert_series_equal(s.cumulative_eval(expr1), expected1)
    assert_series_equal(s.cumulative_eval(expr2), expected2)

    # evaluate combined expressions and validate
    expr3 = expr1 - expr2
    expected3 = pl.Series("values", [0.0, -3.0, -8.0, -15.0, -24.0])
    assert_series_equal(s.cumulative_eval(expr3), expected3)


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

    # set negative indices
    a = pl.Series(range(5))
    a[-2] = None
    a[-5] = None
    assert a.to_list() == [None, 1, 2, None, 4]

    with pytest.raises(pl.OutOfBoundsError):
        a[-100] = None


def test_repr() -> None:
    s = pl.Series("ints", [1001, 2002, 3003])
    s_repr = repr(s)

    assert "shape: (3,)" in s_repr
    assert "Series: 'ints' [i64]" in s_repr
    for n in s.to_list():
        assert str(n) in s_repr

    class XSeries(pl.Series):
        """Custom Series class."""

    # check custom class name reflected in repr output
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
    ("value", "time_unit", "exp", "exp_type"),
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
    value: int,
    time_unit: EpochTimeUnit,
    exp: date | datetime,
    exp_type: pl.PolarsDataType,
) -> None:
    s = pl.Series("timestamp", [value, None])
    result = pl.from_epoch(s, time_unit=time_unit)

    expected = pl.Series("timestamp", [exp, None]).cast(exp_type)
    assert_series_equal(result, expected)


def test_get_chunks() -> None:
    a = pl.Series("a", [1, 2])
    b = pl.Series("a", [3, 4])
    chunks = pl.concat([a, b], rechunk=False).get_chunks()
    assert_series_equal(chunks[0], a)
    assert_series_equal(chunks[1], b)


def test_ptr() -> None:
    # not much to test on the ptr value itself.
    s = pl.Series([1, None, 3])

    ptr = s._get_ptr()[2]
    assert isinstance(ptr, int)
    s2 = s.append(pl.Series([1, 2]))

    ptr2 = s2.rechunk()._get_ptr()[2]
    assert ptr != ptr2

    for dtype in list(polars.datatypes.FLOAT_DTYPES) + list(
        polars.datatypes.INTEGER_DTYPES
    ):
        assert pl.Series([1, 2, 3], dtype=dtype)._s.get_ptr()[2] > 0


def test_get_buffer() -> None:
    s = pl.Series(["a", "bc", None, "éâç", ""])

    data = s._s.get_buffer(0)
    expected = pl.Series([97, 98, 99, 195, 169, 195, 162, 195, 167], dtype=pl.UInt8)
    assert_series_equal(wrap_s(data), expected)

    validity = s._s.get_buffer(1)
    expected = pl.Series([True, True, False, True, True])
    assert_series_equal(wrap_s(validity), expected)

    offsets = s._s.get_buffer(2)
    expected = pl.Series([0, 1, 3, 3, 9, 9], dtype=pl.Int64)
    assert_series_equal(wrap_s(offsets), expected)


def test_get_buffer_no_validity_or_offsets() -> None:
    s = pl.Series([1, 2, 3])

    validity = s._s.get_buffer(1)
    assert validity is None

    offsets = s._s.get_buffer(2)
    assert offsets is None


def test_get_buffer_invalid_index() -> None:
    s = pl.Series([1, None, 3])
    with pytest.raises(ValueError):
        s._s.get_buffer(3)


def test_null_comparisons() -> None:
    s = pl.Series("s", [None, "str", "a"])
    assert (s.shift() == s).null_count() == 2
    assert (s.shift() != s).null_count() == 2


def test_min_max_agg_on_str() -> None:
    strings = ["b", "a", "x"]
    s = pl.Series(strings)
    assert (s.min(), s.max()) == ("a", "x")


def test_is_between() -> None:
    s = pl.Series("num", [1, 2, None, 4, 5])
    assert s.is_between(2, 4).to_list() == [False, True, None, True, False]

    s = pl.Series("num", [1, 2, None, 4, 5])
    assert s.is_between(2, 4, closed="left").to_list() == [
        False,
        True,
        None,
        False,
        False,
    ]

    s = pl.Series("num", [1, 2, None, 4, 5])
    assert s.is_between(2, 4, closed="right").to_list() == [
        False,
        False,
        None,
        True,
        False,
    ]

    s = pl.Series("num", [1, 2, None, 4, 5])
    assert s.is_between(pl.lit(2) / 2, pl.lit(4) * 2, closed="both").to_list() == [
        True,
        True,
        None,
        True,
        True,
    ]

    s = pl.Series("s", ["a", "b", "c", "d", "e"])
    assert s.is_between("b", "d").to_list() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_map_dict() -> None:
    s = pl.Series("s", [-1, 2, None, 4, -5])
    remap = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    assert_series_equal(
        s.abs().map_dict(remap, default="?"),
        pl.Series("s", ["one", "two", "?", "four", "five"]),
    )
    assert_series_equal(
        s.map_dict(remap, default=s.cast(pl.Utf8)),
        pl.Series("s", ["-1", "two", None, "four", "-5"]),
    )

    remap_int = {1: 11, 2: 22, 3: 33, 4: 44, 5: 55}

    assert_series_equal(
        s.map_dict(remap_int, default=pl.first()),
        pl.Series("s", [-1, 22, None, 44, -5]),
    )

    assert_series_equal(
        s.cast(pl.Int16).map_dict(remap_int),
        pl.Series("s", [None, 22, None, 44, None], dtype=pl.Int16),
    )

    assert_series_equal(
        s.cast(pl.Int16).map_dict(remap_int, default=pl.first()),
        pl.Series("s", [-1, 22, None, 44, -5], dtype=pl.Int16),
    )

    assert_series_equal(
        s.cast(pl.Int16).map_dict(
            remap_int, default=pl.first(), return_dtype=pl.Float32
        ),
        pl.Series("s", [-1.0, 22.0, None, 44.0, -5.0], dtype=pl.Float32),
    )

    assert_series_equal(
        s.cast(pl.Int16).map_dict(remap_int, default=9),
        pl.Series("s", [9, 22, 9, 44, 9], dtype=pl.Int16),
    )

    assert_series_equal(
        s.cast(pl.Int16).map_dict(remap_int, default=9, return_dtype=pl.Float32),
        pl.Series("s", [9.0, 22.0, 9.0, 44.0, 9.0], dtype=pl.Float32),
    )

    assert_series_equal(
        pl.Series("boolean_to_int", [True, False]).map_dict({True: 1, False: 0}),
        pl.Series("boolean_to_int", [1, 0]),
    )

    assert_series_equal(
        pl.Series("boolean_to_str", [True, False]).map_dict({True: "1", False: "0"}),
        pl.Series("boolean_to_str", ["1", "0"]),
    )


@pytest.mark.parametrize(
    ("dtype", "lower", "upper"),
    [
        (pl.Int8, -128, 127),
        (pl.UInt8, 0, 255),
        (pl.Int16, -32768, 32767),
        (pl.UInt16, 0, 65535),
        (pl.Int32, -2147483648, 2147483647),
        (pl.UInt32, 0, 4294967295),
        (pl.Int64, -9223372036854775808, 9223372036854775807),
        (pl.UInt64, 0, 18446744073709551615),
        (pl.Float32, float("-inf"), float("inf")),
        (pl.Float64, float("-inf"), float("inf")),
    ],
)
def test_upper_lower_bounds(
    dtype: pl.PolarsDataType, upper: int | float, lower: int | float
) -> None:
    s = pl.Series("s", dtype=dtype)
    assert s.lower_bound().item() == lower
    assert s.upper_bound().item() == upper


def test_numpy_series_arithmetic() -> None:
    sx = pl.Series(values=[1, 2])
    y = np.array([3.0, 4.0])

    result_add1 = y + sx
    result_add2 = sx + y
    expected_add = pl.Series([4.0, 6.0], dtype=pl.Float64)
    assert_series_equal(result_add1, expected_add)  # type: ignore[arg-type]
    assert_series_equal(result_add2, expected_add)

    result_sub1 = cast(pl.Series, y - sx)  # py37 is different vs py311 on this one
    expected = pl.Series([2.0, 2.0], dtype=pl.Float64)
    assert_series_equal(result_sub1, expected)
    result_sub2 = sx - y
    expected = pl.Series([-2.0, -2.0], dtype=pl.Float64)
    assert_series_equal(result_sub2, expected)

    result_mul1 = y * sx
    result_mul2 = sx * y
    expected = pl.Series([3.0, 8.0], dtype=pl.Float64)
    assert_series_equal(result_mul1, expected)  # type: ignore[arg-type]
    assert_series_equal(result_mul2, expected)

    result_div1 = y / sx
    expected = pl.Series([3.0, 2.0], dtype=pl.Float64)
    assert_series_equal(result_div1, expected)  # type: ignore[arg-type]
    result_div2 = sx / y
    expected = pl.Series([1 / 3, 0.5], dtype=pl.Float64)
    assert_series_equal(result_div2, expected)

    result_pow1 = y**sx
    expected = pl.Series([3.0, 16.0], dtype=pl.Float64)
    assert_series_equal(result_pow1, expected)  # type: ignore[arg-type]
    result_pow2 = sx**y
    expected = pl.Series([1.0, 16.0], dtype=pl.Float64)
    assert_series_equal(result_pow2, expected)  # type: ignore[arg-type]


def test_from_epoch_seq_input() -> None:
    seq_input = [1147880044]
    expected = pl.Series([datetime(2006, 5, 17, 15, 34, 4)])
    result = pl.from_epoch(seq_input)
    assert_series_equal(result, expected)


def test_symmetry_for_max_in_names() -> None:
    # int
    a = pl.Series("a", [1])
    assert (a - a.max()).name == (a.max() - a).name == a.name
    # float
    a = pl.Series("a", [1.0])
    assert (a - a.max()).name == (a.max() - a).name == a.name
    # duration
    a = pl.Series("a", [1], dtype=pl.Duration("ns"))
    assert (a - a.max()).name == (a.max() - a).name == a.name
    # datetime
    a = pl.Series("a", [1], dtype=pl.Datetime("ns"))
    assert (a - a.max()).name == (a.max() - a).name == a.name

    # TODO: time arithmetic support?
    # a = pl.Series("a", [1], dtype=pl.Time)
    # assert (a - a.max()).name == (a.max() - a).name == a.name


def test_series_getitem_out_of_bounds_positive() -> None:
    s = pl.Series([1, 2])
    with pytest.raises(
        IndexError, match="index 10 is out of bounds for sequence of length 2"
    ):
        s[10]


def test_series_getitem_out_of_bounds_negative() -> None:
    s = pl.Series([1, 2])
    with pytest.raises(
        IndexError, match="index -10 is out of bounds for sequence of length 2"
    ):
        s[-10]


def test_series_cmp_fast_paths() -> None:
    assert (
        pl.Series([None], dtype=pl.Int32) != pl.Series([1, 2], dtype=pl.Int32)
    ).to_list() == [None, None]
    assert (
        pl.Series([None], dtype=pl.Int32) == pl.Series([1, 2], dtype=pl.Int32)
    ).to_list() == [None, None]

    assert (
        pl.Series([None], dtype=pl.Utf8) != pl.Series(["a", "b"], dtype=pl.Utf8)
    ).to_list() == [None, None]
    assert (
        pl.Series([None], dtype=pl.Utf8) == pl.Series(["a", "b"], dtype=pl.Utf8)
    ).to_list() == [None, None]

    assert (
        pl.Series([None], dtype=pl.Boolean)
        != pl.Series([True, False], dtype=pl.Boolean)
    ).to_list() == [None, None]
    assert (
        pl.Series([None], dtype=pl.Boolean)
        == pl.Series([False, False], dtype=pl.Boolean)
    ).to_list() == [None, None]

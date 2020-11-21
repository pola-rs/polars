import pytest
from pypolars import Series
from pypolars.datatypes import *
import numpy as np
import pytest


def create_series():
    return Series("a", [1, 2])


def test_equality():
    a = create_series()
    b = a

    cmp = a == b
    assert isinstance(cmp, Series)
    assert cmp.sum() == 2
    assert (a != b).sum() == 0
    assert (a >= b).sum() == 2
    assert (a <= b).sum() == 2
    assert (a > b).sum() == 0
    assert (a < b).sum() == 0
    assert a.sum() == 3
    assert a.series_equal(b)

    a = Series("name", ["ham", "foo", "bar"])
    assert (a == "ham").to_list() == [True, False, False]


def test_agg():
    a = create_series()
    assert a.mean() == 1.5
    assert a.min() == 1
    assert a.max() == 2


def test_arithmetic():
    a = create_series()
    b = a

    assert ((a * b) == [1, 4]).sum() == 2
    assert ((a / b) == [1.0, 1.0]).sum() == 2
    assert ((a + b) == [2, 4]).sum() == 2
    assert ((a - b) == [0, 0]).sum() == 2
    assert ((a + 1) == [2, 3]).sum() == 2
    assert ((a - 1) == [0, 1]).sum() == 2
    assert ((a / 1) == [1.0, 2.0]).sum() == 2
    assert ((a // 2) == [0, 1]).sum() == 2
    assert ((a * 2) == [2, 4]).sum() == 2
    assert ((1 + a) == [2, 3]).sum() == 2
    assert ((1 - a) == [0, -1]).sum() == 2
    assert ((1 * a) == [1, 2]).sum() == 2
    # integer division
    assert ((1 / a) == [1.0, 0.5]).sum() == 2
    assert ((1 // a) == [1, 0]).sum() == 2


def test_various():
    a = create_series()

    assert a.is_null().sum() == 0
    assert a.name == "a"
    a.rename("b")
    assert a.name == "b"
    assert a.len() == 2
    assert len(a) == 2
    b = a.slice(1, 1)
    assert b.len() == 1
    assert b.series_equal(Series("", [2]))
    a.append(b)
    assert a.series_equal(Series("", [1, 2, 2]))

    a = Series("a", range(20))
    assert a.head(5).len() == 5
    assert a.tail(5).len() == 5
    assert a.head(5) != a.tail(5)

    a = Series("a", [2, 1, 4])
    a.sort(in_place=True)
    assert a.series_equal(Series("", [1, 2, 4]))
    a = Series("a", [2, 1, 1, 4, 4, 4])
    assert list(a.arg_unique()) == [0, 1, 3]

    assert a.take([2, 3]).series_equal(Series("", [1, 4]))
    assert a.is_numeric()
    a = Series("bool", [True, False])
    assert not a.is_numeric()


def test_filter():
    a = Series("a", range(20))
    assert a[a > 1].len() == 18
    assert a[a < 1].len() == 1
    assert a[a <= 1].len() == 2
    assert a[a >= 1].len() == 19
    assert a[a == 1].len() == 1
    assert a[a != 1].len() == 19


def test_cast():
    a = Series("a", range(20))

    assert a.cast(Float32).dtype == Float32
    assert a.cast(Float64).dtype == Float64
    assert a.cast(Int32).dtype == Int32
    assert a.cast(UInt32).dtype == UInt32
    assert a.cast(Date64).dtype == Date64
    assert a.cast(Date32).dtype == Date32


def test_to_python():
    a = Series("a", range(20))
    b = a.to_list()
    assert isinstance(b, list)
    assert len(b) == 20

    a = Series("a", [1, None, 2], nullable=True)
    assert a.null_count() == 1
    assert a.to_list() == [1, None, 2]


def test_sort():
    a = Series("a", [2, 1, 3])
    assert a.sort().to_list() == [1, 2, 3]
    assert a.sort(reverse=True) == [3, 2, 1]


def test_rechunk():
    a = Series("a", [1, 2, 3])
    b = Series("b", [4, 5, 6])
    a.append(b)
    assert a.n_chunks() == 2
    assert a.rechunk(in_place=False).n_chunks() == 1
    a.rechunk(in_place=True)
    assert a.n_chunks() == 1


def test_view():
    a = Series("a", [1.0, 2.0, 3.0])
    assert isinstance(a.view(), np.ndarray)
    assert np.all(a.view() == np.array([1, 2, 3]))


def test_ufunc():
    a = Series("a", [1.0, 2.0, 3.0, 4.0])
    b = np.multiply(a, 4)
    assert isinstance(b, Series)
    assert b == [4, 8, 12, 16]

    # test if null bitmask is preserved
    a = Series("a", [1.0, None, 3.0], nullable=True)
    b = np.exp(a)
    assert b.null_count() == 1


def test_get():
    a = Series("a", [1, 2, 3])
    assert a[0] == 1
    assert a[:2] == [1, 2]


def test_fill_none():
    a = Series("a", [1, 2, None], nullable=True)
    b = a.fill_none("forward")
    assert b == [1, 2, 2]


def test_apply():
    a = Series("a", [1, 2, None], nullable=True)
    b = a.apply(lambda x: x ** 2)
    assert b == [1, 4, None]

    a = Series("a", ["foo", "bar", None], nullable=True)
    b = a.apply(lambda x: x + "py")
    assert b == ["foopy", "barpy", None]

    b = a.apply(lambda x: len(x), dtype_out=Int32)
    assert b == [3, 3, None]

    # with out dtype sniffing
    b = a.apply(lambda x: len(x))
    assert b == [3, 3, None]


def test_shift():
    a = Series("a", [1, 2, 3])
    assert a.shift(1) == [None, 1, 2]
    assert a.shift(-1) == [1, 2, None]
    assert a.shift(-2) == [1, None, None]


@pytest.mark.parametrize(
    "dtype, fmt, null_values", [(Date32, "%d-%m-%Y", 0), (Date32, "%Y-%m-%d", 3)]
)
def test_parse_date(dtype, fmt, null_values):
    dates = ["25-08-1988", "20-01-1993", "25-09-2020"]
    result = Series.parse_date("dates", dates, dtype, fmt)
    # Why results Date64 into `nan`?
    assert result.dtype == dtype
    assert result.is_null().sum() == null_values


def test_rolling():
    a = Series("a", [1, 2, 3, 2, 1])
    assert a.rolling_min(2) == [None, 1, 2, 2, 1]
    assert a.rolling_max(2) == [None, 2, 3, 3, 2]
    assert a.rolling_sum(2) == [None, 3, 5, 5, 3]


def test_object():
    vals = [[12], "foo", 9]
    a = Series("a", vals)
    assert a.dtype == Object
    assert a.to_list() == vals
    assert a[1] == "foo"

from polars import Series
from polars.ffi import aligned_array_f32
import numpy as np


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
    assert ((a / b) == [1, 1]).sum() == 2
    assert ((a + b) == [2, 4]).sum() == 2
    assert ((a - b) == [0, 0]).sum() == 2
    assert ((a + 1) == [2, 3]).sum() == 2
    assert ((a - 1) == [0, 1]).sum() == 2
    assert ((a / 1) == [1, 2]).sum() == 2
    assert ((a * 2) == [2, 4]).sum() == 2
    assert ((1 + a) == [2, 3]).sum() == 2
    assert ((1 - a) == [0, -1]).sum() == 2
    assert ((1 * a) == [1, 2]).sum() == 2
    # integer division
    assert ((1 / a) == [1, 0]).sum() == 2


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

    assert a.cast_f32().dtype == "f32"
    assert a.cast_f64().dtype == "f64"
    assert a.cast_i32().dtype == "i32"
    assert a.cast_u32().dtype == "u32"
    assert a.cast_date64().dtype == "date64"
    assert a.cast_time64ns().dtype == "time64(ns)"
    assert a.cast_date32().dtype == "date32"


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


def test_numpy_interface():
    # this isn't used anymore.
    a, ptr = aligned_array_f32(10)
    assert a.dtype == np.float32
    assert a.shape == (10,)
    pointer, read_only_flag = a.__array_interface__["data"]
    # set read only flag to False
    a.__array_interface__["data"] = (pointer, False)
    # the __array_interface is used to create a new array (pointing to the same memory)
    b = np.array(a)
    # now the memory is writeable
    b[0] = 1

    # TODO: sent pointer to Rust and take ownership of array.
    # https://stackoverflow.com/questions/37988849/safer-way-to-expose-a-c-allocated-memory-buffer-using-numpy-ctypes


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

from polars import Series


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
    assert ((1 + a) == [2, 3]).sum() == 2


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
    a.sort()
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

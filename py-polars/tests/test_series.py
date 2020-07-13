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


def test_agg():
    a = create_series()
    assert a.mean() == 1.5
    assert a.min() == 1
    assert a.max() == 2

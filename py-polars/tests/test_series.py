from datetime import date, datetime
from functools import reduce
from typing import Any, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars import testing
from polars.datatypes import Float64, Int32, Int64, UInt32, UInt64


def _getattr_multi(obj: object, op: str) -> Any:
    """ "
    Allows `op` to be multiple layers deep, i.e. op="str.lengths" will mean we first
    get the attribute "str", and then the attribute "lengths"
    """
    op_list = op.split(".")
    return reduce(lambda o, m: getattr(o, m), op_list, obj)


def verify_series_and_expr_api(
    input: pl.Series, expected: pl.Series, op: str, *args: Any, **kwargs: Any
) -> None:
    """
    Small helper function to test element-wise functions for both the series and expressions api.

    Examples
    --------
    >>> s = pl.Series([1, 3, 2])
    >>> expected = pl.Series([1, 2, 3])
    >>> verify_series_and_expr_api(s, expected, "sort")
    """
    expr = _getattr_multi(pl.col("*"), op)(*args, **kwargs)
    result_expr: pl.Series = input.to_frame().select(expr)[:, 0]  # type: ignore
    result_series = _getattr_multi(input, op)(*args, **kwargs)
    testing.assert_series_equal(result_expr, expected)
    testing.assert_series_equal(result_series, expected)


def test_cum_agg() -> None:
    s = pl.Series("a", [1, 2, 3, 2])
    verify_series_and_expr_api(s, pl.Series("a", [1, 3, 6, 8]), "cumsum")
    verify_series_and_expr_api(s, pl.Series("a", [1, 1, 1, 1]), "cummin")
    verify_series_and_expr_api(s, pl.Series("a", [1, 2, 3, 3]), "cummax")
    verify_series_and_expr_api(s, pl.Series("a", [1, 2, 6, 12]), "cumprod")


def test_init_inputs() -> None:
    # Good inputs
    pl.Series("a", [1, 2])
    pl.Series("a", values=[1, 2])
    pl.Series(name="a", values=[1, 2])
    pl.Series(values=[1, 2], name="a")

    assert pl.Series([1, 2]).dtype == pl.Int64
    assert pl.Series(values=[1, 2]).dtype == pl.Int64
    assert pl.Series("a").dtype == pl.Float32  # f32 type used in case of no data
    assert pl.Series().dtype == pl.Float32
    assert pl.Series(values=[True, False]).dtype == pl.Boolean
    assert pl.Series(values=np.array([True, False])).dtype == pl.Boolean
    assert pl.Series(values=np.array(["foo", "bar"])).dtype == pl.Utf8
    assert pl.Series(values=["foo", "bar"]).dtype == pl.Utf8
    assert pl.Series("a", [pl.Series([1, 2, 4]), pl.Series([3, 2, 1])]).dtype == pl.List
    assert pl.Series(pd.Series([1, 2])).dtype == pl.Int64
    assert pl.Series("a", [10000, 20000, 30000], dtype=pl.Time).dtype == pl.Time
    # 2d numpy array
    res = pl.Series(name="a", values=np.array([[1, 2], [3, 4]]))
    assert all(res[0] == np.array([1, 2]))
    assert all(res[1] == np.array([3, 4]))
    assert (
        pl.Series(values=np.array([["foo", "bar"], ["foo2", "bar2"]])).dtype
        == pl.Object
    )

    # Bad inputs
    with pytest.raises(ValueError):
        pl.Series([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        pl.Series({"a": [1, 2, 3]})
    with pytest.raises(OverflowError):
        pl.Series("bigint", [2 ** 64])


def test_concat() -> None:
    s = pl.Series("a", [2, 1, 3])

    assert pl.concat([s, s]).len() == 6
    # check if s remains unchanged
    assert s.len() == 3


def test_to_frame() -> None:
    s = pl.Series([1, 2])
    assert s.to_frame().shape == (2, 1)


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
    assert (True & a).series_equal(pl.Series([True, False, True]))  # type: ignore
    assert (True | a).series_equal(pl.Series([True, True, True]))  # type: ignore
    assert (True ^ a).series_equal(pl.Series([False, True, False]))  # type: ignore


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
    testing.assert_series_equal((a == "ham"), pl.Series("name", [True, False, False]))


def test_agg() -> None:
    series = pl.Series("a", [1, 2])
    assert series.mean() == 1.5
    assert series.min() == 1
    assert series.max() == 2


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
    assert ((1 + a) == [2, 3]).sum() == 2
    assert ((1 - a) == [0, -1]).sum() == 2
    assert ((1 * a) == [1, 2]).sum() == 2
    # integer division
    testing.assert_series_equal(1 / a, pl.Series([1.0, 0.5]))  # type: ignore
    if s.dtype == Int64:
        expected = pl.Series([1, 0])
    else:
        expected = pl.Series([1.0, 0.5])
    testing.assert_series_equal(1 // a, expected)
    # modulo
    assert ((1 % a) == [0, 1]).sum() == 2
    assert ((a % 1) == [0, 0]).sum() == 2
    # negate
    assert (-a == [-1, -2]).sum() == 2
    # wrong dtypes in rhs operands
    assert ((1.0 - a) == [0, -1]).sum() == 2
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
        a ** 2
    with pytest.raises(ValueError):
        2 / a
    with pytest.raises(ValueError):
        2 // a
    with pytest.raises(ValueError):
        2 * a
    with pytest.raises(ValueError):
        2 % a
    with pytest.raises(ValueError):
        2 ** a


def test_add_string() -> None:
    s = pl.Series(["hello", "weird"])
    result = s + " world"
    testing.assert_series_equal(result, pl.Series(["hello world", "weird world"]))


def test_various() -> None:
    a = pl.Series("a", [1, 2])

    assert a.is_null().sum() == 0
    assert a.name == "a"
    a.rename("b", in_place=True)
    assert a.name == "b"
    assert a.len() == 2
    assert len(a) == 2
    b = a.slice(1, 1)
    assert b.len() == 1
    assert b.series_equal(pl.Series("b", [2]))
    a.append(b)
    assert a.series_equal(pl.Series("b", [1, 2, 2]))

    a = pl.Series("a", range(20))
    assert a.head(5).len() == 5
    assert a.tail(5).len() == 5
    assert a.head(5) != a.tail(5)

    a = pl.Series("a", [2, 1, 4])
    a.sort(in_place=True)
    assert a.series_equal(pl.Series("a", [1, 2, 4]))
    a = pl.Series("a", [2, 1, 1, 4, 4, 4])
    testing.assert_series_equal(a.arg_unique(), pl.Series("a", [0, 1, 3], dtype=UInt32))

    assert a.take([2, 3]).series_equal(pl.Series("a", [1, 4]))
    assert a.is_numeric()
    a = pl.Series("bool", [True, False])
    assert not a.is_numeric()


def test_filter_ops() -> None:
    a = pl.Series("a", range(20))
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
    testing.assert_series_equal(a.sort(), pl.Series("a", [1, 2, 3]))
    testing.assert_series_equal(a.sort(reverse=True), pl.Series("a", [3, 2, 1]))


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

    a = pa.array(["foo", "bar"], pa.dictionary(pa.int32(), pa.utf8()))
    s = pl.Series("a", a)
    assert s.dtype == pl.Categorical
    assert (
        pl.from_arrow(pa.array([["foo"], ["foo", "bar"]], pa.list_(pa.utf8()))).dtype
        == pl.List
    )


def test_view() -> None:
    a = pl.Series("a", [1.0, 2.0, 3.0])
    assert isinstance(a.view(), np.ndarray)
    assert np.all(a.view() == np.array([1, 2, 3]))


def test_ufunc() -> None:
    a = pl.Series("a", [1.0, 2.0, 3.0, 4.0])
    b = np.multiply(a, 4)
    assert isinstance(b, pl.Series)
    assert b == [4, 8, 12, 16]

    # test if null bitmask is preserved
    a = pl.Series("a", [1.0, None, 3.0])
    b = np.exp(a)
    assert b.null_count() == 1

    # test if it works with chunked series.
    a = pl.Series("a", [1.0, None, 3.0])
    b = pl.Series("b", [4.0, 5.0, None])
    a.append(b)
    assert a.n_chunks() == 2
    c = np.multiply(a, 3)
    testing.assert_series_equal(c, pl.Series("a", [3.0, None, 9.0, 12.0, 15.0, None]))


def test_get() -> None:
    a = pl.Series("a", [1, 2, 3])
    assert a[0] == 1
    assert a[:2] == [1, 2]
    assert a[range(1)] == [1, 2]
    assert a[range(0, 2, 2)] == [1, 3]


def test_set() -> None:
    a = pl.Series("a", [True, False, True])
    mask = pl.Series("msk", [True, False, True])
    a[mask] = False
    testing.assert_series_equal(a, pl.Series("", [False] * 3))


def test_set_value_as_list_fail() -> None:
    """ " it is not allowed to use a list to set values"""
    s = pl.Series("a", [1, 2, 3])
    with pytest.raises(ValueError):
        s[[0, 1]] = [4, 5]


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
    """only UInt32/UInt64/bool are allowed"""
    s = pl.Series("a", [1, 2, 3])
    s[key] = 4
    testing.assert_series_equal(s, pl.Series("a", [1, 4, 4]))


def test_set_np_array_boolean_mask() -> None:
    a = pl.Series("a", [1, 2, 3])
    mask = np.array([True, False, True])
    a[mask] = 4
    testing.assert_series_equal(a, pl.Series("a", [4, 2, 4]))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_set_np_array(dtype: Any) -> None:
    a = pl.Series("a", [1, 2, 3])
    idx = np.array([0, 2], dtype=dtype)
    a[idx] = 4
    testing.assert_series_equal(a, pl.Series("a", [4, 2, 4]))


@pytest.mark.parametrize("idx", [[0, 2], (0, 2)])
def test_set_list_and_tuple(idx: Union[list, tuple]) -> None:
    a = pl.Series("a", [1, 2, 3])
    a[idx] = 4
    testing.assert_series_equal(a, pl.Series("a", [4, 2, 4]))


def test_fill_null() -> None:
    a = pl.Series("a", [1, 2, None])
    verify_series_and_expr_api(a, pl.Series("a", [1, 2, 2]), "fill_null", "forward")

    verify_series_and_expr_api(
        a, pl.Series("a", [1, 2, 14], dtype=Int64), "fill_null", 14
    )


def test_apply() -> None:
    a = pl.Series("a", [1, 2, None])
    b = a.apply(lambda x: x ** 2)
    assert b == [1, 4, None]

    a = pl.Series("a", ["foo", "bar", None])
    b = a.apply(lambda x: x + "py")
    assert b == ["foopy", "barpy", None]

    b = a.apply(lambda x: len(x), return_dtype=pl.Int32)
    assert b == [3, 3, None]

    b = a.apply(lambda x: len(x))
    assert b == [3, 3, None]

    # just check that it runs (somehow problem with conditional compilation)
    a = pl.Series("a", [2, 2, 3]).cast(pl.Datetime)
    a.apply(lambda x: x)
    a = pl.Series("a", [2, 2, 3]).cast(pl.Date)
    a.apply(lambda x: x)


def test_shift() -> None:
    a = pl.Series("a", [1, 2, 3])
    testing.assert_series_equal(a.shift(1), pl.Series("a", [None, 1, 2]))
    testing.assert_series_equal(a.shift(-1), pl.Series("a", [2, 3, None]))
    testing.assert_series_equal(a.shift(-2), pl.Series("a", [3, None, None]))
    testing.assert_series_equal(a.shift_and_fill(-1, 10), pl.Series("a", [2, 3, 10]))


def test_rolling() -> None:
    a = pl.Series("a", [1, 2, 3, 2, 1])
    testing.assert_series_equal(a.rolling_min(2), pl.Series("a", [None, 1, 2, 2, 1]))
    testing.assert_series_equal(a.rolling_max(2), pl.Series("a", [None, 2, 3, 3, 2]))
    testing.assert_series_equal(a.rolling_sum(2), pl.Series("a", [None, 3, 5, 5, 3]))
    testing.assert_series_equal(
        a.rolling_mean(2), pl.Series("a", [None, 1.5, 2.5, 2.5, 1.5])
    )
    assert a.rolling_std(2).to_list()[1] == pytest.approx(0.7071067811865476)
    assert a.rolling_var(2).to_list()[1] == pytest.approx(0.5)
    testing.assert_series_equal(
        a.rolling_median(4), pl.Series("a", [None, None, None, 2, 2], dtype=Float64)
    )
    testing.assert_series_equal(
        a.rolling_quantile(0, "nearest", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    testing.assert_series_equal(
        a.rolling_quantile(0, "lower", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    testing.assert_series_equal(
        a.rolling_quantile(0, "higher", 3),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    assert a.rolling_skew(4).null_count() == 3


def test_object() -> None:
    vals = [[12], "foo", 9]
    a = pl.Series("a", vals)
    assert a.dtype == pl.Object
    assert a.to_list() == vals
    assert a[1] == "foo"


def test_repeat() -> None:
    s = pl.repeat(1, 10)
    assert s.dtype == pl.Int64
    assert s.len() == 10
    s = pl.repeat("foo", 10)
    assert s.dtype == pl.Utf8
    assert s.len() == 10
    s = pl.repeat(1.0, 5)
    assert s.dtype == pl.Float64
    assert s.len() == 5
    assert s == [1.0, 1.0, 1.0, 1.0, 1.0]
    s = pl.repeat(True, 5)
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
def test_create_list_series(arrow_available: bool) -> None:
    pl.internals.series._PYARROW_AVAILABLE = arrow_available
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
    a = pl.Series()
    assert a.dtype == pl.Float32
    a = pl.Series("name", [])
    assert a.dtype == pl.Float32
    a = pl.Series(values=(), dtype=pl.Int8)
    assert a.dtype == pl.Int8


def test_describe() -> None:
    num_s = pl.Series([1, 2, 3])
    float_s = pl.Series([1.3, 4.6, 8.9])
    str_s = pl.Series(["abc", "pqr", "xyz"])
    bool_s = pl.Series([True, False, None, True, True])
    date_s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
    empty_s = pl.Series(np.empty(0))

    assert num_s.describe().shape == (6, 2)
    assert float_s.describe().shape == (6, 2)
    assert str_s.describe().shape == (3, 2)
    assert bool_s.describe().shape == (3, 2)
    assert date_s.describe().shape == (4, 2)

    with pytest.raises(ValueError):
        assert empty_s.describe()


def test_is_in() -> None:
    s = pl.Series([1, 2, 3])

    out = s.is_in([1, 2])
    assert out == [True, True, False]
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [1, 4]})

    assert df[pl.col("a").is_in(pl.col("b")).alias("mask")]["mask"] == [True, False]


def test_str_slice() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    assert df["a"].str.slice(-3) == ["bar", "foo"]

    assert df[[pl.col("a").str.slice(2, 4)]]["a"] == ["obar", "rfoo"]


def test_arange_expr() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    out = df[[pl.arange(0, pl.col("a").count() * 10)]]
    assert out.shape == (20, 1)
    assert out.select_at_idx(0)[-1] == 19

    # eager arange
    out2 = pl.arange(0, 10, 2, eager=True)
    assert out2 == [0, 2, 4, 8, 8]

    out3 = pl.arange(pl.Series([0, 19]), pl.Series([3, 39]), step=2, eager=True)
    assert out3.dtype == pl.List  # type: ignore
    assert out3[0].to_list() == [0, 2]  # type: ignore


def test_round() -> None:
    a = pl.Series("f", [1.003, 2.003])
    b = a.round(2)
    assert b == [1.00, 2.00]


def test_apply_list_out() -> None:
    s = pl.Series("count", [3, 2, 2])
    out = s.apply(lambda val: pl.repeat(val, val))
    assert out[0] == [3, 3, 3]
    assert out[1] == [2, 2]
    assert out[2] == [2, 2]


def test_is_first() -> None:
    s = pl.Series("", [1, 1, 2])
    assert s.is_first() == [True, False, True]


def test_reinterpret() -> None:
    s = pl.Series("a", [1, 1, 2], dtype=pl.UInt64)
    assert s.reinterpret(signed=True).dtype == pl.Int64
    df = pl.DataFrame([s])
    assert df[[pl.col("a").reinterpret(signed=True)]]["a"].dtype == pl.Int64


def test_mode() -> None:
    s = pl.Series("a", [1, 1, 2])
    assert s.mode() == [1]
    df = pl.DataFrame([s])
    assert df[[pl.col("a").mode()]]["a"] == [1]


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


def test_rank_dispatch() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    testing.assert_series_equal(
        s.rank("dense"), pl.Series("a", [2, 3, 4, 3, 3, 4, 1], dtype=UInt32)
    )

    df = pl.DataFrame([s])
    assert df.select(pl.col("a").rank("dense"))["a"] == [2, 3, 4, 3, 3, 4, 1]

    testing.assert_series_equal(
        s.rank("dense", reverse=True),
        pl.Series("a", [3, 2, 1, 2, 2, 1, 4], dtype=UInt32),
    )


def test_diff_dispatch() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    expected = pl.Series("a", [1, 1, -1, 0, 1, -3])

    testing.assert_series_equal(s.diff(null_behavior="drop"), expected)

    df = pl.DataFrame([s])
    testing.assert_series_equal(
        df.select(pl.col("a").diff())["a"], pl.Series("a", [None, 1, 1, -1, 0, 1, -3])
    )


def test_pct_change_dispatch() -> None:
    s = pl.Series("a", [1, 2, 4, 8, 16, 32, 64])
    expected = pl.Series("a", [None, None, float("inf"), 3.0, 3.0, 3.0, 3.0])
    verify_series_and_expr_api(s, expected, "pct_change", 2)


def test_skew_dispatch() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert s.skew(True) == pytest.approx(-0.5953924651018018)
    assert s.skew(False) == pytest.approx(-0.7717168360221258)

    df = pl.DataFrame([s])
    assert np.isclose(df.select(pl.col("a").skew(False))["a"][0], -0.7717168360221258)


def test_kurtosis_dispatch() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    expected = -0.6406250000000004

    assert s.kurtosis() == pytest.approx(expected)
    df = pl.DataFrame([s])
    assert np.isclose(df.select(pl.col("a").kurtosis())["a"][0], expected)


def test_arr_lengths_dispatch() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    testing.assert_series_equal(s.arr.lengths(), pl.Series("a", [2, 3], dtype=UInt32))
    df = pl.DataFrame([s])
    testing.assert_series_equal(
        df.select(pl.col("a").arr.lengths())["a"], pl.Series("a", [2, 3], dtype=UInt32)
    )


def test_arr_arithmetic() -> None:
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    testing.assert_series_equal(s.arr.sum(), pl.Series("a", [3, 6]))
    testing.assert_series_equal(s.arr.mean(), pl.Series("a", [1.5, 2.0]))
    testing.assert_series_equal(s.arr.max(), pl.Series("a", [2, 3]))
    testing.assert_series_equal(s.arr.min(), pl.Series("a", [1, 1]))


def test_arr_ordering() -> None:
    s = pl.Series("a", [[2, 1], [1, 3, 2]])
    testing.assert_series_equal(s.arr.sort(), pl.Series("a", [[1, 2], [1, 2, 3]]))
    testing.assert_series_equal(s.arr.reverse(), pl.Series("a", [[1, 2], [2, 3, 1]]))


def test_arr_unique() -> None:
    s = pl.Series("a", [[2, 1], [1, 2, 2]])
    result = s.arr.unique()
    assert len(result) == 2
    assert sorted(result[0]) == [1, 2]
    assert sorted(result[1]) == [1, 2]


def test_sqrt_dispatch() -> None:
    s = pl.Series("a", [1, 2])
    testing.assert_series_equal(s.sqrt(), pl.Series("a", [1.0, np.sqrt(2)]))
    df = pl.DataFrame([s])
    testing.assert_series_equal(
        df.select(pl.col("a").sqrt())["a"], pl.Series("a", [1.0, np.sqrt(2)])
    )


def test_range() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    assert s[2:5].series_equal(s[range(2, 5)])
    df = pl.DataFrame([s])
    assert df[2:5].frame_equal(df[range(2, 5)])


def test_strict_cast() -> None:
    with pytest.raises(RuntimeError):
        pl.Series("a", [2 ** 16]).cast(dtype=pl.Int16, strict=True)
    with pytest.raises(RuntimeError):
        pl.DataFrame({"a": [2 ** 16]}).select([pl.col("a").cast(pl.Int16, strict=True)])


def test_list_concat_dispatch() -> None:
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
    testing.assert_series_equal(s // 2, pl.Series("a", [0, 1, 1]))
    testing.assert_series_equal(
        pl.DataFrame([s]).select(pl.col("a") // 2)["a"], pl.Series("a", [0, 1, 1])
    )


def test_true_divide() -> None:
    s = pl.Series("a", [1, 2])
    testing.assert_series_equal(s / 2, pl.Series("a", [0.5, 1.0]))
    testing.assert_series_equal(
        pl.DataFrame([s]).select(pl.col("a") / 2)["a"], pl.Series("a", [0.5, 1.0])
    )

    # rtruediv
    testing.assert_series_equal(
        pl.DataFrame([s]).select(2 / pl.col("a"))["literal"],
        pl.Series("literal", [2.0, 1.0]),
    )

    # https://github.com/pola-rs/polars/issues/1369
    vals = [3000000000, 2, 3]
    foo = pl.Series(vals)
    testing.assert_series_equal(foo / 1, pl.Series(vals, dtype=Float64))
    testing.assert_series_equal(
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
    testing.assert_series_equal(a & b, pl.Series("a", [1, 0, 1]))
    testing.assert_series_equal(a | b, pl.Series("a", [3, 6, 7]))
    testing.assert_series_equal(a ^ b, pl.Series("a", [2, 6, 6]))

    df = pl.DataFrame([a, b])
    out = df.select(
        [
            (pl.col("a") & pl.col("b")).alias("and"),
            (pl.col("a") | pl.col("b")).alias("or"),
            (pl.col("a") ^ pl.col("b")).alias("xor"),
        ]
    )
    testing.assert_series_equal(out["and"], pl.Series("and", [1, 0, 1]))
    testing.assert_series_equal(out["or"], pl.Series("or", [3, 6, 7]))
    testing.assert_series_equal(out["xor"], pl.Series("xor", [2, 6, 6]))


def test_to_numpy() -> None:
    pl.internals.series._PYARROW_AVAILABLE = False
    a = pl.Series("a", [1, 2, 3])
    assert np.all(a.to_numpy() == np.array([1, 2, 3]))
    a = pl.Series("a", [1, 2, None])
    np.testing.assert_array_equal(a.to_numpy(), np.array([1.0, 2.0, np.nan]))


def test_from_sequences() -> None:
    # test int, str, bool, flt
    values = [
        [[1], [None, 3]],
        [["foo"], [None, "bar"]],
        [[True], [None, False]],
        [[1.0], [None, 3.0]],
    ]

    for vals in values:
        pl.internals.series._PYARROW_AVAILABLE = False
        a = pl.Series("a", vals)
        pl.internals.series._PYARROW_AVAILABLE = True
        b = pl.Series("a", vals)
        assert a.series_equal(b, null_equal=True)
        assert a.to_list() == vals


def test_comparisons_int_series_to_float() -> None:
    srs_int = pl.Series([1, 2, 3, 4])
    testing.assert_series_equal(srs_int - 1.0, pl.Series([0, 1, 2, 3]))
    testing.assert_series_equal(srs_int + 1.0, pl.Series([2, 3, 4, 5]))
    testing.assert_series_equal(srs_int * 2.0, pl.Series([2, 4, 6, 8]))
    # todo: this is inconsistent
    testing.assert_series_equal(srs_int / 2.0, pl.Series([0.5, 1.0, 1.5, 2.0]))
    testing.assert_series_equal(srs_int % 2.0, pl.Series([1, 0, 1, 0]))
    testing.assert_series_equal(4.0 % srs_int, pl.Series([0, 0, 1, 0]))

    testing.assert_series_equal(srs_int // 2.0, pl.Series([0, 1, 1, 2]))
    testing.assert_series_equal(srs_int < 3.0, pl.Series([True, True, False, False]))
    testing.assert_series_equal(srs_int <= 3.0, pl.Series([True, True, True, False]))
    testing.assert_series_equal(srs_int > 3.0, pl.Series([False, False, False, True]))
    testing.assert_series_equal(srs_int >= 3.0, pl.Series([False, False, True, True]))
    testing.assert_series_equal(srs_int == 3.0, pl.Series([False, False, True, False]))
    testing.assert_series_equal(srs_int - True, pl.Series([0, 1, 2, 3]))


def test_comparisons_float_series_to_int() -> None:
    srs_float = pl.Series([1.0, 2.0, 3.0, 4.0])
    testing.assert_series_equal(srs_float - 1, pl.Series([0.0, 1.0, 2.0, 3.0]))
    testing.assert_series_equal(srs_float + 1, pl.Series([2.0, 3.0, 4.0, 5.0]))
    testing.assert_series_equal(srs_float * 2, pl.Series([2.0, 4.0, 6.0, 8.0]))
    testing.assert_series_equal(srs_float / 2, pl.Series([0.5, 1.0, 1.5, 2.0]))
    testing.assert_series_equal(srs_float % 2, pl.Series([1.0, 0.0, 1.0, 0.0]))
    testing.assert_series_equal(4 % srs_float, pl.Series([0.0, 0.0, 1.0, 0.0]))

    testing.assert_series_equal(srs_float // 2, pl.Series([0.0, 1.0, 1.0, 2.0]))
    testing.assert_series_equal(srs_float < 3, pl.Series([True, True, False, False]))
    testing.assert_series_equal(srs_float <= 3, pl.Series([True, True, True, False]))
    testing.assert_series_equal(srs_float > 3, pl.Series([False, False, False, True]))
    testing.assert_series_equal(srs_float >= 3, pl.Series([False, False, True, True]))
    testing.assert_series_equal(srs_float == 3, pl.Series([False, False, True, False]))
    testing.assert_series_equal(srs_float - True, pl.Series([0.0, 1.0, 2.0, 3.0]))


def test_comparisons_bool_series_to_int() -> None:
    srs_bool = pl.Series([True, False])
    # todo: do we want this to work?
    testing.assert_series_equal(srs_bool / 1, pl.Series([True, False], dtype=Float64))
    with pytest.raises(TypeError, match=r"\-: 'Series' and 'int'"):
        srs_bool - 1
    with pytest.raises(TypeError, match=r"\+: 'Series' and 'int'"):
        srs_bool + 1
    with pytest.raises(TypeError, match=r"\%: 'Series' and 'int'"):
        srs_bool % 2
    with pytest.raises(TypeError, match=r"\*: 'Series' and 'int'"):
        srs_bool * 1
    with pytest.raises(
        TypeError, match=r"'<' not supported between instances of 'Series' and 'int'"
    ):
        srs_bool < 2
    with pytest.raises(
        TypeError, match=r"'>' not supported between instances of 'Series' and 'int'"
    ):
        srs_bool > 2


def test_trigonometry_functions() -> None:
    srs_float = pl.Series("t", [0.0, np.pi])
    assert np.allclose(srs_float.sin(), np.array([0.0, 0.0]))
    assert np.allclose(srs_float.cos(), np.array([1.0, -1.0]))
    assert np.allclose(srs_float.tan(), np.array([0.0, -0.0]))

    srs_float = pl.Series("t", [1.0, 0.0, -1])
    assert np.allclose(srs_float.arcsin(), np.array([1.571, 0.0, -1.571]), atol=0.01)
    assert np.allclose(srs_float.arccos(), np.array([0.0, 1.571, 3.142]), atol=0.01)
    assert np.allclose(srs_float.arctan(), np.array([0.785, 0.0, -0.785]), atol=0.01)


def test_abs() -> None:
    # ints
    s = pl.Series([1, -2, 3, -4])
    testing.assert_series_equal(s.abs(), pl.Series([1, 2, 3, 4]))
    testing.assert_series_equal(np.abs(s), pl.Series([1, 2, 3, 4]))  # type: ignore

    # floats
    s = pl.Series([1.0, -2.0, 3, -4.0])
    testing.assert_series_equal(s.abs(), pl.Series([1.0, 2.0, 3.0, 4.0]))
    testing.assert_series_equal(
        np.abs(s), pl.Series([1.0, 2.0, 3.0, 4.0])  # type: ignore
    )
    testing.assert_series_equal(
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

    verify_series_and_expr_api(s, expected, "argsort")

    expected_reverse = pl.Series("a", [0, 2, 1, 4, 3], dtype=UInt32)
    verify_series_and_expr_api(s, expected_reverse, "argsort", True)


def test_arg_min_and_arg_max() -> None:
    s = pl.Series("a", [5, 3, 4, 1, 2])
    assert s.arg_min() == 3
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
    s = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("b", [4.0, 5.0, 6.0])
    assert s.dot(s2) == 32


def test_sample() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5])
    assert len(s.sample(n=2)) == 2
    assert len(s.sample(frac=0.4)) == 2

    assert len(s.sample(n=2, with_replacement=True)) == 2

    # on a series of length 5, you cannot sample more than 5 items
    with pytest.raises(Exception):
        s.sample(n=10, with_replacement=False)
    # unless you use with_replacement=True
    assert len(s.sample(n=10, with_replacement=True)) == 10


def test_peak_max_peak_min() -> None:
    s = pl.Series("a", [4, 1, 3, 2, 5])
    result = s.peak_min()
    expected = pl.Series([False, True, False, True, False])
    testing.assert_series_equal(result, expected)

    result = s.peak_max()
    expected = pl.Series([True, False, True, False, True])
    testing.assert_series_equal(result, expected)


def test_shrink_to_fit() -> None:
    s = pl.Series("a", [4, 1, 3, 2, 5])
    assert s.shrink_to_fit(in_place=True) is None

    s = pl.Series("a", [4, 1, 3, 2, 5])
    assert isinstance(s.shrink_to_fit(in_place=False), pl.Series)


def test_str_concat() -> None:
    s = pl.Series(["1", None, "2"])
    result = s.str_concat()
    expected = pl.Series(["1-null-2"])
    testing.assert_series_equal(result, expected)


def test_str_lengths() -> None:
    s = pl.Series(["messi", "ronaldo", None])
    expected = pl.Series([5, 7, None], dtype=UInt32)
    verify_series_and_expr_api(s, expected, "str.lengths")


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
        s.str.encode("utf8")


def test_str_decode() -> None:
    hex_encoded = pl.Series(["666f6f", "626172", None])
    base64_encoded = pl.Series(["Zm9v", "YmFy", None])
    expected = pl.Series(["foo", "bar", None])

    verify_series_and_expr_api(hex_encoded, expected, "str.decode", "hex")
    verify_series_and_expr_api(base64_encoded, expected, "str.decode", "base64")


def test_str_decode_exception() -> None:
    s = pl.Series(["not a valid", "626172", None])
    with pytest.raises(Exception):
        s.str.decode(encoding="hex", strict=True)
    with pytest.raises(Exception):
        s.str.decode(encoding="base64", strict=True)
    with pytest.raises(ValueError):
        s.str.decode("utf8")


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


def test_str_rstrip() -> None:
    s = pl.Series([" hello ", "world\t "])
    expected = pl.Series([" hello", "world"])
    testing.assert_series_equal(s.str.rstrip(), expected)


def test_str_lstrip() -> None:
    s = pl.Series([" hello ", "\t world"])
    expected = pl.Series(["hello ", "world"])
    testing.assert_series_equal(s.str.lstrip(), expected)


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
    verify_series_and_expr_api(a, pl.Series("a", [0, 4, 1], dtype=UInt32), "dt.weekday")
    verify_series_and_expr_api(a, pl.Series("a", [21, 40, 8], dtype=UInt32), "dt.week")
    verify_series_and_expr_api(a, pl.Series("a", [19, 4, 20], dtype=UInt32), "dt.day")
    verify_series_and_expr_api(
        a, pl.Series("a", [139, 278, 51], dtype=UInt32), "dt.ordinal_day"
    )

    assert a.dt.median() == date(2024, 10, 4)
    assert a.dt.mean() == date(2024, 10, 4)


def test_dt_datetimes() -> None:
    s = pl.Series(["2020-01-01 00:00:00", "2020-02-02 03:20:10"])
    s = s.str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S")

    # hours, minutes, seconds and nanoseconds
    verify_series_and_expr_api(s, pl.Series("", [0, 3], dtype=UInt32), "dt.hour")
    verify_series_and_expr_api(s, pl.Series("", [0, 20], dtype=UInt32), "dt.minute")
    verify_series_and_expr_api(s, pl.Series("", [0, 10], dtype=UInt32), "dt.second")
    verify_series_and_expr_api(s, pl.Series("", [0, 0], dtype=UInt32), "dt.nanosecond")

    # epoch methods
    verify_series_and_expr_api(
        s, pl.Series("", [18262, 18294], dtype=Int32), "dt.epoch_days"
    )
    verify_series_and_expr_api(
        s,
        pl.Series("", [1_577_836_800, 1_580_613_610], dtype=Int64),
        "dt.epoch_seconds",
    )
    verify_series_and_expr_api(
        s,
        pl.Series("", [1_577_836_800_000, 1_580_613_610_000], dtype=Int64),
        "dt.epoch_milliseconds",
    )


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
    for values in [[None], ["foo", "bar"], [None, "foo", "bar"]]:
        expected = pl.Series("a", values, dtype=pl.Utf8).cast(pl.Categorical)
        a = pl.Series("a", values, dtype=pl.Categorical)
        testing.assert_series_equal(a, expected)


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
    expected = pl.Series("a", [2, 3, 1])
    testing.assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).shuffle(2)).to_series()
    testing.assert_series_equal(out, expected)


def test_to_physical() -> None:
    # casting an int result in an int
    a = pl.Series("a", [1, 2, 3])
    verify_series_and_expr_api(a, a, "to_physical")

    # casting a date results in an Int32
    a = pl.Series("a", [date(2020, 1, 1)] * 3)
    expected = pl.Series("a", [18262] * 3, dtype=Int32)
    verify_series_and_expr_api(a, expected, "to_physical")


def test_is_between_datetime() -> None:
    s = pl.Series("a", [datetime(2020, 1, 1, 10, 0, 0), datetime(2020, 1, 1, 20, 0, 0)])
    start = datetime(2020, 1, 1, 12, 0, 0)
    end = datetime(2020, 1, 1, 23, 0, 0)
    expected = pl.Series("a", [False, True])

    # only on the expression api
    result = s.to_frame().with_column(pl.col("*").is_between(start, end))["is_between"]
    testing.assert_series_equal(result.rename("a"), expected)


@pytest.mark.parametrize("f", ["sin", "cos", "tan", "arcsin", "arccos", "arctan"])
def test_trigonometric(f: str) -> None:
    s = pl.Series("a", [0.0])
    expected = pl.Series("a", getattr(np, f)(s.to_numpy()))
    verify_series_and_expr_api(s, expected, f)


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


def test_ewm_std_var() -> None:
    a = pl.Series("a", [2, 5, 3])

    assert (a.ewm_std(alpha=0.5) ** 2).to_list() == a.ewm_var(alpha=0.5).to_list()


def test_extend() -> None:
    a = pl.Series("a", [1, 2, 3])
    expected = pl.Series("a", [1, 2, 3, 1, 1, 1])
    verify_series_and_expr_api(a, expected, "extend", 1, 3)

    expected = pl.Series("a", [1, 2, 3, None, None, None])
    verify_series_and_expr_api(a, expected, "extend", None, 3)


def test_any_all() -> None:
    a = pl.Series("a", [True, False, True])
    expected = pl.Series("a", [True])
    verify_series_and_expr_api(a, expected, "any")
    expected = pl.Series("a", [False])
    verify_series_and_expr_api(a, expected, "all")

    a = pl.Series("a", [True, True, True])
    expected = pl.Series("a", [True])
    verify_series_and_expr_api(a, expected, "any")
    expected = pl.Series("a", [True])
    verify_series_and_expr_api(a, expected, "all")

    a = pl.Series("a", [False, False, False])
    expected = pl.Series("a", [False])
    verify_series_and_expr_api(a, expected, "any")
    expected = pl.Series("a", [False])
    verify_series_and_expr_api(a, expected, "all")


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


def test_strip() -> None:
    a = pl.Series("a", ["trailing  ", "  leading", "  both  "])
    expected = pl.Series("a", ["trailing", "  leading", "  both"])
    verify_series_and_expr_api(a, expected, "str.rstrip")
    expected = pl.Series("a", ["trailing  ", "leading", "both  "])
    verify_series_and_expr_api(a, expected, "str.lstrip")
    expected = pl.Series("a", ["trailing", "leading", "both"])
    verify_series_and_expr_api(a, expected, "str.strip")


def test_ceil() -> None:
    a = pl.Series("a", [1.8, 1.2, 3.0])
    expected = pl.Series("a", [2.0, 2.0, 3.0])
    verify_series_and_expr_api(a, expected, "ceil")


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

    expected = pl.Series("b", [3600 * 24 * 1000])
    verify_series_and_expr_api(duration, expected, "dt.milliseconds")

    expected = pl.Series("b", [3600 * 24 * int(1e9)])
    verify_series_and_expr_api(duration, expected, "dt.nanoseconds")

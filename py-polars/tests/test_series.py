from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars import testing
from polars.datatypes import Float64, Int32, Int64, UInt32


def series() -> pl.Series:
    return pl.Series("a", [1, 2])


def test_cum_agg() -> None:
    s = pl.Series("a", [1, 2, 3, 2])
    assert s.cumsum().series_equal(pl.Series("a", [1, 3, 6, 8]))
    assert s.cummin().series_equal(pl.Series("a", [1, 1, 1, 1]))
    assert s.cummax().series_equal(pl.Series("a", [1, 2, 3, 3]))
    assert s.cumprod().series_equal(pl.Series("a", [1, 2, 6, 12]))


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


def test_to_frame(series: pl.Series) -> None:
    assert series.to_frame().shape == (2, 1)


def test_bitwise_ops() -> None:
    a = pl.Series([True, False, True])
    b = pl.Series([False, True, True])
    assert a & b == [False, False, True]
    assert a | b == [True, True, True]
    assert ~a == [False, True, False]


def test_equality(series: pl.Series) -> None:
    a = series
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


def test_agg(series: pl.Series) -> None:
    assert series.mean() == 1.5
    assert series.min() == 1
    assert series.max() == 2


def test_arithmetic(series: pl.Series) -> None:
    a = series
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


def test_various(series: pl.Series) -> None:
    a = series

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


def test_fill_null() -> None:
    a = pl.Series("a", [1, 2, None])
    b = a.fill_null("forward")
    assert b == [1, 2, 2]
    b = a.fill_null(14)
    testing.assert_series_equal(b, pl.Series("a", [1, 2, 14], dtype=Int64))


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
        a.rolling_quantile(3, 0, "nearest"),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    testing.assert_series_equal(
        a.rolling_quantile(3, 0, "lower"),
        pl.Series("a", [None, None, 1, 2, 1], dtype=Float64),
    )
    testing.assert_series_equal(
        a.rolling_quantile(3, 0, "higher"),
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


def test_create_list_series() -> None:
    for b in [True, False]:
        pl.internals.series._PYARROW_AVAILABLE = b
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
    out = pl.arange(0, 10, 2, eager=True)
    assert out == [0, 2, 4, 8, 8]

    out = pl.arange(pl.Series([0, 19]), pl.Series([3, 39]), step=2, eager=True)
    assert out.dtype == pl.List
    assert out[0].to_list() == [0, 2]


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
    testing.assert_series_equal(
        s.str.json_path_match("$.a"),
        pl.Series(
            [
                "1",
                None,
                "2",
                "2.1",
                "true",
            ]
        ),
    )


def test_extract_regex() -> None:
    s = pl.Series(
        [
            "http://vote.com/ballon_dor?candidate=messi&ref=polars",
            "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
            "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ]
    )
    testing.assert_series_equal(
        s.str.extract(r"candidate=(\w+)", 1),
        pl.Series(
            [
                "messi",
                None,
                "ronaldo",
            ]
        ),
    )


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
    )  # type: ignore


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
    result = s.argsort()
    expected = pl.Series("a", [3, 4, 1, 2, 0])
    assert result.series_equal(expected)

    result_reverse = s.argsort(True)
    expected_reverse = pl.Series("a", [0, 2, 1, 4, 3])
    assert result_reverse.series_equal(expected_reverse)


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
    result = s.str.lengths()
    expected = pl.Series([5, 7, None], dtype=UInt32)
    testing.assert_series_equal(result, expected)


def test_str_contains() -> None:
    s = pl.Series(["messi", "ronaldo", "ibrahimovic"])
    result = s.str.contains("mes")
    expected = pl.Series([True, False, False])
    testing.assert_series_equal(result, expected)


def test_str_replace_str_replace_all() -> None:
    s = pl.Series(["hello", "world", "test"])
    result = s.str.replace("o", "0")
    expected = pl.Series(["hell0", "w0rld", "test"])
    testing.assert_series_equal(result, expected)

    s = pl.Series(["hello", "world", "test"])
    result = s.str.replace_all("o", "0")
    expected = pl.Series(["hell0", "w0rld", "test"])
    testing.assert_series_equal(result, expected)


def test_str_to_lowercase() -> None:
    s = pl.Series(["Hello", "WORLD"])
    result = s.str.to_lowercase()
    expected = pl.Series(["hello", "world"])
    testing.assert_series_equal(result, expected)


def test_str_to_uppercase() -> None:
    s = pl.Series(["Hello", "WORLD"])
    result = s.str.to_uppercase()
    expected = pl.Series(["HELLO", "WORLD"])
    testing.assert_series_equal(result, expected)


def test_str_rstrip() -> None:
    s = pl.Series([" hello ", "world\t "])
    result = s.str.rstrip()
    expected = pl.Series([" hello", "world"])
    testing.assert_series_equal(result, expected)


def test_str_lstrip() -> None:
    s = pl.Series([" hello ", "\t world"])
    result = s.str.lstrip()
    expected = pl.Series(["hello ", "world"])
    testing.assert_series_equal(result, expected)


def test_dt_strftime() -> None:
    a = pl.Series("a", [10000, 20000, 30000], dtype=pl.Date)
    assert a.dtype == pl.Date
    a = a.dt.strftime("%F")
    assert a[2] == "2052-02-20"


def test_dt_year_month_week_day_ordinal_day() -> None:
    a = pl.Series("a", [10000, 20000, 30000], dtype=pl.Date)
    testing.assert_series_equal(
        a.dt.year(), pl.Series("a", [1997, 2024, 2052], dtype=Int32)
    )
    testing.assert_series_equal(a.dt.month(), pl.Series("a", [5, 10, 2], dtype=UInt32))
    testing.assert_series_equal(a.dt.weekday(), pl.Series("a", [0, 4, 1], dtype=UInt32))
    testing.assert_series_equal(a.dt.week(), pl.Series("a", [21, 40, 8], dtype=UInt32))
    testing.assert_series_equal(a.dt.day(), pl.Series("a", [19, 4, 20], dtype=UInt32))
    testing.assert_series_equal(
        a.dt.ordinal_day(), pl.Series("a", [139, 278, 51], dtype=UInt32)
    )

    assert a.dt.median() == date(2024, 10, 4)
    assert a.dt.mean() == date(2024, 10, 4)


def test_compare_series_value_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.Series([2, 3, 4])
    with pytest.raises(AssertionError, match="Series are different\n\nValue mismatch"):
        testing.assert_series_equal(srs1, srs2)


def test_compare_series_type_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.DataFrame({"col1": [2, 3, 4]})
    with pytest.raises(AssertionError, match="Series are different\n\nType mismatch"):
        testing.assert_series_equal(srs1, srs2)  # type: ignore


def test_compare_series_name_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")
    with pytest.raises(AssertionError, match="Series are different\n\nName mismatch"):
        testing.assert_series_equal(srs1, srs2)


def test_compare_series_shape_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3, 4], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")
    with pytest.raises(AssertionError, match="Series are different\n\nShape mismatch"):
        testing.assert_series_equal(srs1, srs2)


def test_compare_series_value_exact_mismatch() -> None:
    srs1 = pl.Series([1.0, 2.0, 3.0])
    srs2 = pl.Series([1.0, 2.0 + 1e-7, 3.0])
    with pytest.raises(
        AssertionError, match="Series are different\n\nExact value mismatch"
    ):
        testing.assert_series_equal(srs1, srs2, check_exact=True)


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

    out = a.log10()
    expected = pl.Series("a", [0.0, 2.0, 3.0])
    testing.assert_series_equal(out, expected)
    a = pl.Series("a", [1, 100, 1000])

    out = a.log()
    expected = pl.Series("a", np.log(a.to_numpy()))
    testing.assert_series_equal(out, expected)

    out = a.exp()
    expected = pl.Series("a", np.exp(a.to_numpy()))
    testing.assert_series_equal(out, expected)


def test_shuffle() -> None:
    a = pl.Series("a", [1, 2, 3])
    out = a.shuffle(2)
    expected = pl.Series("a", [2, 3, 1])
    testing.assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).shuffle(2)).to_series()
    testing.assert_series_equal(out, expected)

from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars.datatypes import *


def create_series() -> pl.Series:
    return pl.Series("a", [1, 2])


def test_cum_agg():
    s = create_series()
    assert s.cumsum() == [1, 2]
    assert s.cummin() == [1, 1]
    assert s.cummax() == [1, 2]


def test_init_inputs():
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

    # Bad inputs
    with pytest.raises(ValueError):
        pl.Series([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        pl.Series({"a": [1, 2, 3]})
    with pytest.raises(OverflowError):
        pl.Series("bigint", [2 ** 64])


def test_concat():
    s = pl.Series("a", [2, 1, 3])

    assert pl.concat([s, s]).len() == 6
    # check if s remains unchanged
    assert s.len() == 3


def test_to_frame():
    assert create_series().to_frame().shape == (2, 1)


def test_bitwise_ops():
    a = pl.Series([True, False, True])
    b = pl.Series([False, True, True])
    assert a & b == [False, False, True]
    assert a | b == [True, True, True]


def test_equality():
    a = create_series()
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


def test_various():
    a = create_series()

    assert a.is_null().sum() == 0
    assert a.name == "a"
    a.rename("b", in_place=True)
    assert a.name == "b"
    assert a.len() == 2
    assert len(a) == 2
    b = a.slice(1, 1)
    assert b.len() == 1
    assert b.series_equal(pl.Series("", [2]))
    a.append(b)
    assert a.series_equal(pl.Series("", [1, 2, 2]))

    a = pl.Series("a", range(20))
    assert a.head(5).len() == 5
    assert a.tail(5).len() == 5
    assert a.head(5) != a.tail(5)

    a = pl.Series("a", [2, 1, 4])
    a.sort(in_place=True)
    assert a.series_equal(pl.Series("", [1, 2, 4]))
    a = pl.Series("a", [2, 1, 1, 4, 4, 4])
    assert a.arg_unique().to_list() == [0, 1, 3]

    assert a.take([2, 3]).series_equal(pl.Series("", [1, 4]))
    assert a.is_numeric()
    a = pl.Series("bool", [True, False])
    assert not a.is_numeric()


def test_filter():
    a = pl.Series("a", range(20))
    assert a[a > 1].len() == 18
    assert a[a < 1].len() == 1
    assert a[a <= 1].len() == 2
    assert a[a >= 1].len() == 19
    assert a[a == 1].len() == 1
    assert a[a != 1].len() == 19


def test_cast():
    a = pl.Series("a", range(20))

    assert a.cast(Float32).dtype == Float32
    assert a.cast(Float64).dtype == Float64
    assert a.cast(Int32).dtype == Int32
    assert a.cast(UInt32).dtype == UInt32
    assert a.cast(Datetime).dtype == Datetime
    assert a.cast(Date).dtype == Date


def test_to_python():
    a = pl.Series("a", range(20))
    b = a.to_list()
    assert isinstance(b, list)
    assert len(b) == 20

    a = pl.Series("a", [1, None, 2])
    assert a.null_count() == 1
    assert a.to_list() == [1, None, 2]


def test_sort():
    a = pl.Series("a", [2, 1, 3])
    assert a.sort().to_list() == [1, 2, 3]
    assert a.sort(reverse=True) == [3, 2, 1]


def test_rechunk():
    a = pl.Series("a", [1, 2, 3])
    b = pl.Series("b", [4, 5, 6])
    a.append(b)
    assert a.n_chunks() == 2
    assert a.rechunk(in_place=False).n_chunks() == 1
    a.rechunk(in_place=True)
    assert a.n_chunks() == 1


def test_indexing():
    a = pl.Series("a", [1, 2, None])
    assert a[1] == 2
    assert a[2] == None
    b = pl.Series("b", [True, False])
    assert b[0]
    assert not b[1]
    a = pl.Series("a", ["a", None])
    assert a[0] == "a"
    assert a[1] == None
    a = pl.Series("a", [0.1, None])
    assert a[0] == 0.1
    assert a[1] == None


def test_arrow():
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


def test_view():
    a = pl.Series("a", [1.0, 2.0, 3.0])
    assert isinstance(a.view(), np.ndarray)
    assert np.all(a.view() == np.array([1, 2, 3]))


def test_ufunc():
    a = pl.Series("a", [1.0, 2.0, 3.0, 4.0])
    b = np.multiply(a, 4)
    assert isinstance(b, pl.Series)
    assert b == [4, 8, 12, 16]

    # test if null bitmask is preserved
    a = pl.Series("a", [1.0, None, 3.0])
    b = np.exp(a)
    assert b.null_count() == 1


def test_get():
    a = pl.Series("a", [1, 2, 3])
    assert a[0] == 1
    assert a[:2] == [1, 2]


def test_set():
    a = pl.Series("a", [True, False, True])
    mask = pl.Series("msk", [True, False, True])
    a[mask] = False


def test_fill_null():
    a = pl.Series("a", [1, 2, None])
    b = a.fill_null("forward")
    assert b == [1, 2, 2]
    b = a.fill_null(14)
    assert b.to_list() == [1, 2, 14]


def test_apply():
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


def test_shift():
    a = pl.Series("a", [1, 2, 3])
    assert a.shift(1).to_list() == [None, 1, 2]
    assert a.shift(-1).to_list() == [2, 3, None]
    assert a.shift(-2).to_list() == [3, None, None]
    assert a.shift_and_fill(-1, 10).to_list() == [2, 3, 10]


def test_rolling():
    a = pl.Series("a", [1, 2, 3, 2, 1])
    assert a.rolling_min(2).to_list() == [None, 1, 2, 2, 1]
    assert a.rolling_max(2).to_list() == [None, 2, 3, 3, 2]
    assert a.rolling_sum(2).to_list() == [None, 3, 5, 5, 3]
    assert np.isclose(a.rolling_std(2).to_list()[1], 0.7071067811865476)
    assert np.isclose(a.rolling_var(2).to_list()[1], 0.5)
    assert a.rolling_median(4).to_list() == [None, None, None, 2, 2]
    assert a.rolling_quantile(3, 0.5).to_list() == [None, None, 2, 2, 2]
    assert a.rolling_skew(4).null_count() == 3


def test_object():
    vals = [[12], "foo", 9]
    a = pl.Series("a", vals)
    assert a.dtype == pl.Object
    assert a.to_list() == vals
    assert a[1] == "foo"


def test_repeat():
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


def test_median():
    s = pl.Series([1, 2, 3])
    assert s.median() == 2


def test_quantile():
    s = pl.Series([1, 2, 3])
    assert s.quantile(0.5) == 2


def test_shape():
    s = pl.Series([1, 2, 3])
    assert s.shape == (3,)


def test_create_list_series():
    pass
    # may Segfault: see https://github.com/pola-rs/polars/issues/518
    # a = [[1, 2], None, [None, 3]]
    # s = pl.Series("", a)
    # assert s.to_list() == a


def test_iter():
    s = pl.Series("", [1, 2, 3])

    iter = s.__iter__()
    assert iter.__next__() == 1
    assert iter.__next__() == 2
    assert iter.__next__() == 3
    assert sum(s) == 6


def test_empty():
    a = pl.Series(dtype=pl.Int8)
    assert a.dtype == pl.Int8
    a = pl.Series()
    assert a.dtype == pl.Float32
    a = pl.Series("name", [])
    assert a.dtype == pl.Float32
    a = pl.Series(values=(), dtype=pl.Int8)
    assert a.dtype == pl.Int8


def test_describe():
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


def test_is_in():
    s = pl.Series([1, 2, 3])

    out = s.is_in([1, 2])
    assert out == [True, True, False]
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [1, 4]})

    assert df[pl.col("a").is_in(pl.col("b")).alias("mask")]["mask"] == [True, False]


def test_str_slice():
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    assert df["a"].str.slice(-3) == ["bar", "foo"]

    assert df[[pl.col("a").str.slice(2, 4)]]["a"] == ["obar", "rfoo"]


def test_arange_expr():
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


def test_strftime():
    a = pl.Series("a", [10000, 20000, 30000], dtype=pl.Date)
    assert a.dtype == pl.Date
    a = a.dt.strftime("%F")
    assert a[2] == "2052-02-20"


def test_timestamp():
    from datetime import datetime

    a = pl.Series("a", [10000, 20000, 30000], dtype=pl.Datetime)
    assert a.dt.timestamp() == [10000, 20000, 30000]
    out = a.dt.to_python_datetime()
    assert isinstance(out[0], datetime)
    assert a.dt.min() == out[0]
    assert a.dt.max() == out[2]

    df = pl.DataFrame([out])
    # test if rows returns objects
    assert isinstance(df.row(0)[0], datetime)


def test_from_pydatetime():
    dates = [
        datetime(2021, 1, 1),
        datetime(2021, 1, 2),
        datetime(2021, 1, 3),
        datetime(2021, 1, 4, 12, 12),
        None,
    ]
    s = pl.Series("name", dates)
    assert s.dtype == pl.Datetime
    assert s.name == "name"
    assert s.null_count() == 1
    assert s.dt[0] == dates[0]
    # fmt dates and nulls
    print(s)

    dates = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3), None]
    s = pl.Series("name", dates)
    assert s.dtype == pl.Date
    assert s.name == "name"
    assert s.null_count() == 1
    assert s.dt[0] == dates[0]

    # fmt dates and nulls
    print(s)


def test_from_pandas_nan_to_none():
    from pyarrow import ArrowInvalid

    df = pd.Series([2, np.nan, None], name="pd")
    out_true = pl.from_pandas(df)
    out_false = pl.from_pandas(df, nan_to_none=False)
    df.loc[2] = pd.NA
    assert [val is None for val in out_true]
    assert [np.isnan(val) for val in out_false[1:]]
    with pytest.raises(ArrowInvalid, match="Could not convert"):
        pl.from_pandas(df, nan_to_none=False)


def test_round():
    a = pl.Series("f", [1.003, 2.003])
    b = a.round(2)
    assert b == [1.00, 2.00]


def test_apply_list_out():
    s = pl.Series("count", [3, 2, 2])
    out = s.apply(lambda val: pl.repeat(val, val))
    assert out[0] == [3, 3, 3]
    assert out[1] == [2, 2]
    assert out[2] == [2, 2]


def test_is_first():
    s = pl.Series("", [1, 1, 2])
    assert s.is_first() == [True, False, True]


def test_reinterpret():
    s = pl.Series("a", [1, 1, 2], dtype=pl.UInt64)
    assert s.reinterpret(signed=True).dtype == pl.Int64
    df = pl.DataFrame([s])
    assert df[[pl.col("a").reinterpret(signed=True)]]["a"].dtype == pl.Int64


def test_mode():
    s = pl.Series("a", [1, 1, 2])
    assert s.mode() == [1]
    df = pl.DataFrame([s])
    assert df[[pl.col("a").mode()]]["a"] == [1]


def test_jsonpath_single():
    s = pl.Series(['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}'])
    print(s.str.json_path_match("$.a"))
    assert s.str.json_path_match("$.a").to_list() == [
        "1",
        None,
        "2",
        "2.1",
        "true",
    ]


def test_rank_dispatch():
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert list(s.rank("dense")) == [2, 3, 4, 3, 3, 4, 1]

    df = pl.DataFrame([s])
    df.select(pl.col("a").rank("dense"))["a"] == [2, 3, 4, 3, 3, 4, 1]


def test_diff_dispatch():
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    expected = [1, 1, -1, 0, 1, -3]

    assert list(s.diff(null_behavior="drop")) == expected

    df = pl.DataFrame([s])
    assert df.select(pl.col("a").diff())["a"].to_list() == [None, 1, 1, -1, 0, 1, -3]


def test_skew_dispatch():
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert np.isclose(s.skew(True), -0.5953924651018018)
    assert np.isclose(s.skew(False), -0.7717168360221258)

    df = pl.DataFrame([s])
    assert np.isclose(df.select(pl.col("a").skew(False))["a"][0], -0.7717168360221258)


def test_kurtosis_dispatch():
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    expected = -0.6406250000000004

    assert np.isclose(s.kurtosis(), expected)
    df = pl.DataFrame([s])
    assert np.isclose(df.select(pl.col("a").kurtosis())["a"][0], expected)


def test_arr_lengths_dispatch():
    s = pl.Series("a", [[1, 2], [1, 2, 3]])
    assert s.arr.lengths().to_list() == [2, 3]
    df = pl.DataFrame([s])
    assert df.select(pl.col("a").arr.lengths())["a"].to_list() == [2, 3]


def test_sqrt_dispatch():
    s = pl.Series("a", [1, 2])
    assert s.sqrt().to_list() == [1, np.sqrt(2)]
    df = pl.DataFrame([s])
    assert df.select(pl.col("a").sqrt())["a"].to_list() == [1, np.sqrt(2)]


def test_range():
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    assert s[2:5].series_equal(s[range(2, 5)])
    df = pl.DataFrame([s])
    assert df[2:5].frame_equal(df[range(2, 5)])


def test_strict_cast():
    with pytest.raises(RuntimeError):
        pl.Series("a", [2 ** 16]).cast(dtype=pl.Int16, strict=True)
    with pytest.raises(RuntimeError):
        pl.DataFrame({"a": [2 ** 16]}).select([pl.col("a").cast(pl.Int16, strict=True)])


def test_list_concat_dispatch():
    s0 = pl.Series("a", [[1, 2]])
    s1 = pl.Series("b", [[3, 4, 5]])
    expected = pl.Series("a", [[1, 2, 3, 4, 5]])

    out = s0.arr.concat([s1])
    assert out.series_equal(expected)

    out = s0.arr.concat(s1)
    assert out.series_equal(expected)

    df = pl.DataFrame([s0, s1])
    assert df.select(pl.concat_list(["a", "b"]).alias("concat"))["concat"].series_equal(
        expected
    )
    assert df.select(pl.col("a").arr.concat("b").alias("concat"))[
        "concat"
    ].series_equal(expected)
    assert df.select(pl.col("a").arr.concat(["b"]).alias("concat"))[
        "concat"
    ].series_equal(expected)


def test_floor_divide():
    s = pl.Series("a", [1, 2, 3])
    assert (s // 2).to_list() == [0, 1, 1]
    assert pl.DataFrame([s]).select(pl.col("a") // 2)["a"].to_list() == [0, 1, 1]


def test_true_divide():
    s = pl.Series("a", [1, 2])
    assert (s / 2).to_list() == [0.5, 1.0]
    assert pl.DataFrame([s]).select(pl.col("a") / 2)["a"].to_list() == [0.5, 1.0]

    # https://github.com/pola-rs/polars/issues/1369
    vals = [3000000000, 2, 3]
    foo = pl.Series(vals)
    assert (foo / 1).to_list() == vals
    assert pl.DataFrame({"a": vals}).select([pl.col("a") / 1])["a"].to_list() == vals


def test_invalid_categorical():
    s = pl.Series("cat_series", ["a", "b", "b", "c", "a"]).cast(pl.Categorical)
    assert s.std() is None
    assert s.var() is None
    assert s.median() is None
    assert s.quantile(0.5) is None
    assert s.mode().to_list() == [None]


def test_bitwise():
    a = pl.Series("a", [1, 2, 3])
    b = pl.Series("b", [3, 4, 5])
    assert (a & b).to_list() == [1, 0, 1]
    assert (a | b).to_list() == [3, 6, 7]
    assert (a ^ b).to_list() == [2, 6, 6]

    df = pl.DataFrame([a, b])
    out = df.select(
        [
            (pl.col("a") & pl.col("b")).alias("and"),
            (pl.col("a") | pl.col("b")).alias("or"),
            (pl.col("a") ^ pl.col("b")).alias("xor"),
        ]
    )
    out["and"].to_list() == [1, 0, 1]
    out["or"].to_list() == [3, 6, 7]
    out["xor"].to_list() == [2, 6, 6]


def test_to_numpy():
    pl.eager.series._PYARROW_AVAILABLE = False
    a = pl.Series("a", [1, 2, 3])
    a.to_numpy() == np.array([1, 2, 3])
    a = pl.Series("a", [1, 2, None])
    a.to_numpy() == np.array([1.0, 2.0, np.nan])


def test_from_sequences():
    # test int, str, bool, flt
    values = [
        [[1], [None, 3]],
        [["foo"], [None, "bar"]],
        [[True], [None, False]],
        [[1.0], [None, 3.0]],
    ]

    for vals in values:
        pl.internals.construction._PYARROW_AVAILABLE = False
        a = pl.Series("a", vals)
        pl.internals.construction._PYARROW_AVAILABLE = True
        b = pl.Series("a", vals)
        assert a.series_equal(b, null_equal=True)
        assert a.to_list() == vals

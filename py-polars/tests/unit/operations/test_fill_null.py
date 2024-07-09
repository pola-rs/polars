import polars as pl
from polars.testing import assert_series_equal


def test_fill_null_minimal_upcast_4056() -> None:
    df = pl.DataFrame({"a": [-1, 2, None]})
    df = df.with_columns(pl.col("a").cast(pl.Int8))
    assert df.with_columns(pl.col(pl.Int8).fill_null(-1)).dtypes[0] == pl.Int8
    assert df.with_columns(pl.col(pl.Int8).fill_null(-1000)).dtypes[0] == pl.Int16


def test_fill_enum_upcast() -> None:
    dtype = pl.Enum(["a", "b"])
    s = pl.Series(["a", "b", None], dtype=dtype)
    s_filled = s.fill_null("b")
    expected = pl.Series(["a", "b", "b"], dtype=dtype)
    assert s_filled.dtype == dtype
    assert_series_equal(s_filled, expected)


def test_fill_null_static_schema_4843() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": [1, None, 4],
        }
    ).lazy()

    df2 = df1.select([pl.col(pl.Int64).fill_null(0)])
    df3 = df2.select(pl.col(pl.Int64))
    assert df3.collect_schema() == {"a": pl.Int64, "b": pl.Int64}


def test_fill_null_f32_with_lit() -> None:
    # ensure the literal integer does not upcast the f32 to an f64
    df = pl.DataFrame({"a": [1.1, 1.2]}, schema=[("a", pl.Float32)])
    assert df.fill_null(value=0).dtypes == [pl.Float32]

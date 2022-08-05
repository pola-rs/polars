import pytest

import polars as pl


def test_schema_on_agg() -> None:
    df = pl.DataFrame({"a": ["x", "x", "y", "n"], "b": [1, 2, 3, 4]})

    assert (
        df.lazy()
        .groupby("a")
        .agg(
            [
                pl.col("b").min().alias("min"),
                pl.col("b").max().alias("max"),
                pl.col("b").sum().alias("sum"),
                pl.col("b").first().alias("first"),
                pl.col("b").last().alias("last"),
            ]
        )
    ).schema == {
        "a": pl.Utf8,
        "min": pl.Int64,
        "max": pl.Int64,
        "sum": pl.Int64,
        "first": pl.Int64,
        "last": pl.Int64,
    }


def test_fill_null_minimal_upcast_4056() -> None:
    df = pl.DataFrame({"a": [-1, 2, None]})
    df = df.with_columns(pl.col("a").cast(pl.Int8))
    assert df.with_column(pl.col(pl.Int8).fill_null(-1)).dtypes[0] == pl.Int8
    assert df.with_column(pl.col(pl.Int8).fill_null(-1000)).dtypes[0] == pl.Int32


def test_with_column_duplicates() -> None:
    df = pl.DataFrame({"a": [0, None, 2, 3, None], "b": [None, 1, 2, 3, None]})
    assert df.with_columns([pl.all().alias("same")]).columns == ["a", "b", "same"]


def test_pow_dtype() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5],
        }
    ).lazy()

    df = df.with_columns([pl.col("foo").cast(pl.UInt32)]).with_columns(
        [
            (pl.col("foo") * 2**2).alias("scaled_foo"),
            (pl.col("foo") * 2**2.1).alias("scaled_foo2"),
        ]
    )
    assert df.collect().dtypes == [pl.UInt32, pl.UInt32, pl.Float64]


def test_bool_numeric_supertype() -> None:
    df = pl.DataFrame({"v": [1, 2, 3, 4, 5, 6]})
    for dt in [
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
    ]:
        assert (
            df.select([(pl.col("v") < 3).sum().cast(dt) / pl.count()])[0, 0] - 0.3333333
            <= 0.00001
        )


def test_with_context() -> None:
    df_a = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "c", None]}).lazy()

    df_b = pl.DataFrame({"c": ["foo", "ham"]})

    assert (
        df_a.with_context(df_b.lazy()).select([pl.col("b") + pl.col("c").first()])
    ).collect().to_dict(False) == {"b": ["afoo", "cfoo", None]}

    with pytest.raises(pl.ComputeError):
        (df_a.with_context(df_b.lazy()).select(["a", "c"])).collect()

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


def test_from_dicst_nested_nulls() -> None:
    assert pl.from_dicts([{"a": [None, None]}, {"a": [1, 2]}]).to_dict(False) == {
        "a": [[None, None], [1, 2]]
    }


def test_schema_err() -> None:
    df = pl.DataFrame({"foo": [None, 1, 2], "bar": [1, 2, 3]}).lazy()
    with pytest.raises(pl.NotFoundError):
        df.groupby("not-existent").agg(pl.col("bar").max().alias("max_bar")).schema


def test_schema_inference_from_rows() -> None:
    # these have to upcast to float
    assert pl.from_records([[1, 2.1, 3], [4, 5, 6.4]]).to_dict(False) == {
        "column_0": [1.0, 2.1, 3.0],
        "column_1": [4.0, 5.0, 6.4],
    }
    assert pl.from_dicts([{"a": 1, "b": 2}, {"a": 3.1, "b": 4.5}]).to_dict(False) == {
        "a": [1.0, 3.1],
        "b": [2.0, 4.5],
    }


def test_lazy_map_schema() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    # identity
    assert df.lazy().map(lambda x: x).collect().frame_equal(df)

    def custom(df: pl.DataFrame) -> pl.Series:
        return df["a"]

    with pytest.raises(
        pl.ComputeError,
        match="Expected 'LazyFrame.map' to return a 'DataFrame', got a",
    ):
        df.lazy().map(custom).collect()  # type: ignore[arg-type]

    def custom2(
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        # changes schema
        return df.select(pl.all().cast(pl.Utf8))

    with pytest.raises(
        pl.ComputeError,
        match="The output schema of 'LazyFrame.map' is incorrect. Expected",
    ):
        df.lazy().map(custom2).collect()

    assert df.lazy().map(custom2, validate_output_schema=False).collect().to_dict(
        False
    ) == {"a": ["1", "2", "3"], "b": ["a", "b", "c"]}


def test_join_as_of_by_schema() -> None:
    a = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
    b = pl.DataFrame({"a": [1], "b": [2], "d": [4]}).lazy()
    q = a.join_asof(b, on="a", by="b")
    assert q.collect().columns == q.columns


def test_unknown_apply() -> None:
    df = pl.DataFrame(
        {"Amount": [10, 1, 1, 5], "Flour": ["1000g", "100g", "50g", "75g"]}
    )

    q = df.lazy().select(
        [
            pl.col("Amount"),
            pl.col("Flour").apply(lambda x: 100.0) / pl.col("Amount"),
        ]
    )

    assert q.collect().to_dict(False) == {
        "Amount": [10, 1, 1, 5],
        "Flour": [10.0, 100.0, 100.0, 20.0],
    }
    assert q.dtypes == [pl.Int64, pl.Unknown]


def test_remove_redundant_mapping_4668() -> None:
    df = pl.DataFrame([["a"]] * 2, ["A", "B "]).lazy()
    clean_name_dict = {x: " ".join(x.split()) for x in df.columns}
    df = df.rename(clean_name_dict)
    assert df.columns == ["A", "B"]


def test_fold_all_schema() -> None:
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
            "optional": [28, 300, None, 2, -30],
        }
    )
    # divide because of overflow
    assert df.select(pl.sum(pl.all().hash(seed=1) // int(1e8))).dtypes == [pl.UInt64]

import io

import polars as pl


def test_categorical_outer_join() -> None:
    with pl.StringCache():
        df1 = pl.DataFrame(
            [
                pl.Series("key1", [42]),
                pl.Series("key2", ["bar"], dtype=pl.Categorical),
                pl.Series("val1", [1]),
            ]
        ).lazy()

        df2 = pl.DataFrame(
            [
                pl.Series("key1", [42]),
                pl.Series("key2", ["bar"], dtype=pl.Categorical),
                pl.Series("val2", [2]),
            ]
        ).lazy()

    out = df1.join(df2, on=["key1", "key2"], how="outer").collect()
    expected = pl.DataFrame({"val1": [1], "key1": [42], "key2": ["bar"], "val2": [2]})

    assert out.frame_equal(expected)
    with pl.StringCache():
        dfa = pl.DataFrame(
            [
                pl.Series("key", ["foo", "bar"], dtype=pl.Categorical),
                pl.Series("val1", [3, 1]),
            ]
        )
        dfb = pl.DataFrame(
            [
                pl.Series("key", ["bar", "baz"], dtype=pl.Categorical),
                pl.Series("val2", [6, 8]),
            ]
        )

    df = dfa.join(dfb, on="key", how="outer")
    # the cast is important to test the rev map
    assert df["key"].cast(pl.Utf8).to_list() == ["bar", "baz", "foo"]


def test_read_csv_categorical() -> None:
    f = io.BytesIO()
    f.write(b"col1,col2,col3,col4,col5,col6\n'foo',2,3,4,5,6\n'bar',8,9,10,11,12")
    f.seek(0)
    df = pl.read_csv(f, has_header=True, dtypes={"col1": pl.Categorical})
    assert df["col1"].dtype == pl.Categorical


def test_categorical_lexical_sort() -> None:
    df = pl.DataFrame(
        {"cats": ["z", "z", "k", "a", "b"], "vals": [3, 1, 2, 2, 3]}
    ).with_columns(
        [
            pl.col("cats").cast(pl.Categorical).cat.set_ordering("lexical"),
        ]
    )

    out = df.sort(["cats"])
    assert out["cats"].dtype == pl.Categorical
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 3, 1]}
    )
    assert out.with_column(pl.col("cats").cast(pl.Utf8)).frame_equal(expected)
    out = df.sort(["cats", "vals"])
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 1, 3]}
    )
    assert out.with_column(pl.col("cats").cast(pl.Utf8)).frame_equal(expected)
    out = df.sort(["vals", "cats"])

    expected = pl.DataFrame(
        {"cats": ["z", "a", "k", "b", "z"], "vals": [1, 2, 2, 3, 3]}
    )
    assert out.with_column(pl.col("cats").cast(pl.Utf8)).frame_equal(expected)


def test_categorical_lexical_ordering_after_concat() -> None:
    with pl.StringCache():
        ldf1 = (
            pl.DataFrame([pl.Series("key1", [8, 5]), pl.Series("key2", ["fox", "baz"])])
            .lazy()
            .with_column(
                pl.col("key2").cast(pl.Categorical).cat.set_ordering("lexical")
            )
        )
        ldf2 = (
            pl.DataFrame(
                [pl.Series("key1", [6, 8, 6]), pl.Series("key2", ["fox", "foo", "bar"])]
            )
            .lazy()
            .with_column(
                pl.col("key2").cast(pl.Categorical).cat.set_ordering("lexical")
            )
        )
        df = (
            pl.concat([ldf1, ldf2])
            .with_column(pl.col("key2").cat.set_ordering("lexical"))
            .collect()
        )

        df.sort(["key1", "key2"])

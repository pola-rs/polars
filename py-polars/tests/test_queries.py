import polars as pl


def test_sort_by_bools() -> None:
    # tests dispatch
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    out = df.with_column((pl.col("foo") % 2 == 1).alias("foo_odd")).sort(
        by=["foo", "foo_odd"]
    )
    assert out.shape == (3, 4)


def test_type_coercion_when_then_otherwise_2806() -> None:
    df = pl.DataFrame({"names": ["foo", "spam", "spam"], "nrs": [1, 2, 3]})
    out = df.select(
        [
            pl.when((pl.col("names") == "spam"))
            .then((pl.col("nrs") * 2))
            .otherwise(pl.lit("other"))
            .alias("new_col"),
        ]
    ).to_series()
    expected = pl.Series("new_col", ["other", "4", "6"])
    assert out.to_list() == expected.to_list()

import polars as pl


def test_stringcache() -> None:
    with pl.StringCache():
        # create a reasonable sized columns so the categorical map
        # is reallocated
        df = pl.DataFrame({"cats": pl.arange(0, 1500, eager=True)}).select(
            [pl.col("cats").cast(pl.Utf8).cast(pl.Categorical)]
        )
        assert df.filter(pl.col("cats").is_in(["1", "2"])).to_dict(False) == {
            "cats": ["1", "2"]
        }

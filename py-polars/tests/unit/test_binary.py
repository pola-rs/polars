import polars as pl


def test_binary_conversions() -> None:
    df = pl.DataFrame({"blob": [b"abc", None, b"cde"]}).with_column(
        pl.col("blob").cast(pl.Utf8).alias("decoded_blob")
    )

    assert df.to_dict(False) == {
        "blob": [b"abc", None, b"cde"],
        "decoded_blob": ["abc", None, "cde"],
    }
    assert df[0, 0] == b"abc"
    assert df[1, 0] is None
    assert df.dtypes == [pl.Binary, pl.Utf8]

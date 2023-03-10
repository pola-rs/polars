import polars as pl


def test_corr() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 4],
            "b": [-1, 23, 8],
        }
    )
    assert df.corr().to_dict(False) == {
        "a": [1.0, 0.18898223650461357],
        "b": [0.1889822365046136, 1.0],
    }

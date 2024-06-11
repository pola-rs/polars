import polars as pl


def test_lazy_rename() -> None:
    df = pl.DataFrame({"x": [1], "y": [2]})

    result = df.lazy().rename({"y": "x", "x": "y"}).select(["x", "y"])
    assert result.collect().to_dict(as_series=False) == {"x": [2], "y": [1]}

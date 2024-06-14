import polars as pl


def test_lazy_rename() -> None:
    df = pl.DataFrame({"x": [1], "y": [2]})

    result = df.lazy().rename({"y": "x", "x": "y"}).select(["x", "y"])
    assert result.collect().to_dict(as_series=False) == {"x": [2], "y": [1]}


def test_remove_redundant_mapping_4668() -> None:
    lf = pl.LazyFrame([["a"]] * 2, ["A", "B "]).lazy()
    clean_name_dict = {x: " ".join(x.split()) for x in lf.collect_schema()}
    lf = lf.rename(clean_name_dict)
    assert lf.collect_schema().names() == ["A", "B"]

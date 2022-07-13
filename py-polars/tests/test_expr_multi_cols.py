import polars as pl


def test_exclude_name_from_dtypes() -> None:
    df = pl.DataFrame({"a": ["a"], "b": ["b"]})

    assert df.with_column(pl.col(pl.Utf8).exclude("a").suffix("_foo")).frame_equal(
        pl.DataFrame({"a": ["a"], "b": ["b"], "b_foo": ["b"]})
    )

import polars as pl


def test_exclude_name_from_dtypes() -> None:
    df = pl.DataFrame({"a": ["a"], "b": ["b"]})

    assert df.with_column(pl.col(pl.Utf8).exclude("a").suffix("_foo")).frame_equal(
        pl.DataFrame({"a": ["a"], "b": ["b"], "b_foo": ["b"]})
    )


def test_fold_regex_expand() -> None:
    df = pl.DataFrame(
        {
            "x": [0, 1, 2],
            "y_1": [1.1, 2.2, 3.3],
            "y_2": [1.0, 2.5, 3.5],
        }
    )
    assert df.with_column(
        pl.fold(acc=pl.lit(0), f=lambda acc, x: acc + x, exprs=pl.col("^y_.*$")).alias(
            "y_sum"
        ),
    ).to_dict(False) == {
        "x": [0, 1, 2],
        "y_1": [1.1, 2.2, 3.3],
        "y_2": [1.0, 2.5, 3.5],
        "y_sum": [2.1, 4.7, 6.8],
    }


def test_expanding_sum() -> None:
    df = pl.DataFrame(
        {
            "x": [0, 1, 2],
            "y_1": [1.1, 2.2, 3.3],
            "y_2": [1.0, 2.5, 3.5],
        }
    )
    assert df.with_column(pl.sum(pl.col(r"^y_.*$")).alias("y_sum"))[
        "y_sum"
    ].to_list() == [2.1, 4.7, 6.8]

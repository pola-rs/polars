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


def test_argsort_argument_expansion() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "sort_order": [9, 8, 7],
        }
    )
    assert df.select(
        pl.col("col1").sort_by(pl.col("sort_order").arg_sort()).suffix("_suffix")
    ).to_dict(False) == {"col1_suffix": [3, 2, 1]}
    assert df.select(
        pl.col("^col.*$").sort_by(pl.col("sort_order")).arg_sort()
    ).to_dict(False) == {"col1": [2, 1, 0], "col2": [2, 1, 0]}
    assert df.select(
        pl.all().exclude("sort_order").sort_by(pl.col("sort_order")).arg_sort()
    ).to_dict(False) == {"col1": [2, 1, 0], "col2": [2, 1, 0]}


def test_append_root_columns() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2],
            "col2": [10, 20],
            "other": [100, 200],
        }
    )
    assert (
        df.select(
            [
                pl.col("col2").append(pl.col("other")),
                pl.col("col1").append(pl.col("other")).keep_name(),
                pl.col("col1").append(pl.col("other")).prefix("prefix_"),
                pl.col("col1").append(pl.col("other")).suffix("_suffix"),
            ]
        )
    ).columns == ["col2", "col1", "prefix_col1", "col1_suffix"]

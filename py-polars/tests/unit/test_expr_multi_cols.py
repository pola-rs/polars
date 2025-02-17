import polars as pl
from polars.testing import assert_frame_equal


def test_exclude_name_from_dtypes() -> None:
    df = pl.DataFrame({"a": ["a"], "b": ["b"]})

    assert_frame_equal(
        df.with_columns(pl.col(pl.String).exclude("a").name.suffix("_foo")),
        pl.DataFrame({"a": ["a"], "b": ["b"], "b_foo": ["b"]}),
    )


def test_fold_regex_expand() -> None:
    df = pl.DataFrame(
        {
            "x": [0, 1, 2],
            "y_1": [1.1, 2.2, 3.3],
            "y_2": [1.0, 2.5, 3.5],
        }
    )
    assert df.with_columns(
        pl.fold(
            acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=pl.col("^y_.*$")
        ).alias("y_sum"),
    ).to_dict(as_series=False) == {
        "x": [0, 1, 2],
        "y_1": [1.1, 2.2, 3.3],
        "y_2": [1.0, 2.5, 3.5],
        "y_sum": [2.1, 4.7, 6.8],
    }


def test_arg_sort_argument_expansion() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "sort_order": [9, 8, 7],
        }
    )
    assert df.select(
        pl.col("col1").sort_by(pl.col("sort_order").arg_sort()).name.suffix("_suffix")
    ).to_dict(as_series=False) == {"col1_suffix": [3, 2, 1]}
    assert df.select(
        pl.col("^col.*$").sort_by(pl.col("sort_order")).arg_sort()
    ).to_dict(as_series=False) == {"col1": [2, 1, 0], "col2": [2, 1, 0]}
    assert df.select(
        pl.all().exclude("sort_order").sort_by(pl.col("sort_order")).arg_sort()
    ).to_dict(as_series=False) == {"col1": [2, 1, 0], "col2": [2, 1, 0]}


def test_multiple_columns_length_9137() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1],
            "b": ["c", "d"],
        }
    )

    # list is larger than groups
    cmp_list = ["a", "b", "c"]

    assert df.group_by("a").agg(pl.col("b").is_in(cmp_list)).to_dict(
        as_series=False
    ) == {
        "a": [1],
        "b": [[True, False]],
    }


def test_regex_in_cols() -> None:
    df = pl.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "val1": ["a", "b", "c"],
            "val2": ["A", "B", "C"],
        }
    )

    assert df.select(pl.col("^col.*$").name.prefix("matched_")).to_dict(
        as_series=False
    ) == {
        "matched_col1": [1, 2, 3],
        "matched_col2": [4, 5, 6],
    }

    assert df.with_columns(
        pl.col("^col.*$", "^val.*$").name.prefix("matched_")
    ).to_dict(as_series=False) == {
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
        "val1": ["a", "b", "c"],
        "val2": ["A", "B", "C"],
        "matched_col1": [1, 2, 3],
        "matched_col2": [4, 5, 6],
        "matched_val1": ["a", "b", "c"],
        "matched_val2": ["A", "B", "C"],
    }
    assert df.select(pl.col("^col.*$", "val1").name.prefix("matched_")).to_dict(
        as_series=False
    ) == {
        "matched_col1": [1, 2, 3],
        "matched_col2": [4, 5, 6],
        "matched_val1": ["a", "b", "c"],
    }

    assert df.select(pl.col("^col.*$", "val1").exclude("col2")).to_dict(
        as_series=False
    ) == {
        "col1": [1, 2, 3],
        "val1": ["a", "b", "c"],
    }

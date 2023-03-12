import polars as pl
from polars.testing import assert_frame_equal


def test_nested_when_then_and_wildcard_expansion_6284() -> None:
    df = pl.DataFrame(
        {
            "1": ["a", "b"],
            "2": ["c", "d"],
        }
    )

    out0 = df.with_columns(
        pl.when(pl.any(pl.all() == "a"))
        .then("a")
        .otherwise(pl.when(pl.any(pl.all() == "d")).then("d").otherwise(None))
        .alias("result")
    )

    out1 = df.with_columns(
        pl.when(pl.any(pl.all() == "a"))
        .then("a")
        .when(pl.any(pl.all() == "d"))
        .then("d")
        .otherwise(None)
        .alias("result")
    )

    assert_frame_equal(out0, out1)
    assert out0.to_dict(False) == {
        "1": ["a", "b"],
        "2": ["c", "d"],
        "result": ["a", "d"],
    }


def test_expression_literal_series_order() -> None:
    """
    Tests order of operations between Series and pli.Expr.

    Tests add, sub, mul, true_div, floor_div and mod operations
    between Series, pl.col and pl.lit expressions.
    """

    s = pl.Series([1, 2, 3])
    df = pl.DataFrame({"a": [10, 20, 30]})

    # Add
    assert df.select(pl.col("a") + s).to_dict(False) == {"a": [11, 22, 33]}
    assert df.select(s + pl.col("a")).to_dict(False) == {"a": [11, 22, 33]}

    assert df.select(pl.col("a") + pl.lit(s)).to_dict(False) == {"a": [11, 22, 33]}
    # TODO: fix col name in pli.Expr
    # assert df.select(pl.lit(s) + pl.col("a")).to_dict(False) == {"a": [11, 22, 33]}

    # TODO: Change ret type from pli.Expr to Series in pli.Expr?
    # assert (pl.lit(1) + s).to_list() == [2, 3, 4]
    assert (s + pl.lit(1)).to_list() == [2, 3, 4]

    # Sub
    assert df.select(pl.col("a") - s).to_dict(False) == {"a": [9, 18, 27]}
    # TODO: extend the fix to non-commutative operators (-, /, //, %)
    # assert df.select(s - pl.col("a")).to_dict(False) == {"a": [-9, -18, -27]}

    assert df.select(pl.col("a") - pl.lit(s)).to_dict(False) == {"a": [9, 18, 27]}
    # TODO: fix col name in pli.Expr
    # assert df.select(pl.lit(s) - pl.col("a")).to_dict(False) == {"": [-9, -18, -27]}

    # TODO: Change ret type from pli.Expr to Series in pli.Expr?
    # assert (pl.lit(1) - s).to_list() == [0, -1, -2]
    assert (s - pl.lit(1)).to_list() == [0, 1, 2]

    # Mul
    assert df.select(pl.col("a") * s).to_dict(False) == {"a": [10, 40, 90]}
    assert df.select(s * pl.col("a")).to_dict(False) == {"a": [10, 40, 90]}

    assert df.select(pl.col("a") * pl.lit(s)).to_dict(False) == {"a": [10, 40, 90]}
    # TODO: fix col name in pli.Expr
    # assert df.select(pl.lit(s) * pl.col("a")).to_dict(False) == {"": [10, 40, 90]}

    # TODO: Change ret type from pli.Expr to Series in pli.Expr?
    # assert (pl.lit(1) * s).to_list() == [1, 2, 3]
    assert (s * pl.lit(1)).to_list() == [1, 2, 3]

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
    s = pl.Series("b", [1, 2, 3])
    df = pl.DataFrame({"a": [10, 20, 30]})

    # Add
    assert df.select(pl.col("a") + s).to_dict(False) == {"a": [11, 22, 33]}
    assert df.select(s + pl.col("a")).to_dict(False) == {"b": [11, 22, 33]}

    assert df.select(pl.col("a") + pl.lit(s)).to_dict(False) == {"a": [11, 22, 33]}
    assert df.select(pl.lit(s) + pl.col("a")).to_dict(False) == {"b": [11, 22, 33]}

    assert df.select(pl.lit(1) + s).to_dict(False) == {"literal": [2, 3, 4]}
    assert df.select(s + pl.lit(1)).to_dict(False) == {"b": [2, 3, 4]}

    # Sub
    assert df.select(pl.col("a") - s).to_dict(False) == {"a": [9, 18, 27]}
    assert df.select(s - pl.col("a")).to_dict(False) == {"b": [-9, -18, -27]}

    assert df.select(pl.col("a") - pl.lit(s)).to_dict(False) == {"a": [9, 18, 27]}
    assert df.select(pl.lit(s) - pl.col("a")).to_dict(False) == {"b": [-9, -18, -27]}

    assert df.select(pl.lit(1) - s).to_dict(False) == {"literal": [0, -1, -2]}
    assert df.select(s - pl.lit(1)).to_dict(False) == {"b": [0, 1, 2]}

    # Mul
    assert df.select(pl.col("a") * s).to_dict(False) == {"a": [10, 40, 90]}
    assert df.select(s * pl.col("a")).to_dict(False) == {"b": [10, 40, 90]}

    assert df.select(pl.col("a") * pl.lit(s)).to_dict(False) == {"a": [10, 40, 90]}
    assert df.select(pl.lit(s) * pl.col("a")).to_dict(False) == {"b": [10, 40, 90]}

    assert df.select(pl.lit(1) * s).to_dict(False) == {"literal": [1, 2, 3]}
    assert df.select(s * pl.lit(1)).to_dict(False) == {"b": [1, 2, 3]}

    # Div
    assert df.select(pl.col("a") / s).to_dict(False) == {"a": [10.0, 10.0, 10.0]}
    assert df.select(s / pl.col("a")).to_dict(False) == {"b": [0.1, 0.1, 0.1]}

    assert df.select(pl.col("a") / pl.lit(s)).to_dict(False) == {
        "a": [10.0, 10.0, 10.0]
    }
    assert df.select(pl.lit(s) / pl.col("a")).to_dict(False) == {"b": [0.1, 0.1, 0.1]}

    assert df.select(pl.lit(1) / s).to_dict(False) == {"literal": [1.0, 0.5, 1 / 3]}
    assert df.select(s / pl.lit(1)).to_dict(False) == {"b": [1.0, 2.0, 3.0]}

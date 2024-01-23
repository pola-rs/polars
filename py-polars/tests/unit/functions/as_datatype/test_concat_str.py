import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_concat_str_wildcard_expansion() -> None:
    # one function requires wildcard expansion the other need
    # this tests the nested behavior
    # see: #2867

    df = pl.DataFrame({"a": ["x", "Y", "z"], "b": ["S", "o", "S"]})
    assert df.select(
        pl.concat_str(pl.all()).str.to_lowercase()
    ).to_series().to_list() == ["xs", "yo", "zs"]


def test_concat_str_with_non_utf8_col() -> None:
    out = (
        pl.LazyFrame({"a": [0], "b": ["x"]})
        .select(pl.concat_str(["a", "b"], separator="-").fill_null(pl.col("a")))
        .collect()
    )
    expected = pl.Series("a", ["0-x"], dtype=pl.String)
    assert_series_equal(out.to_series(), expected)


def test_empty_df_concat_str_11701() -> None:
    df = pl.DataFrame({"a": []})
    out = df.select(pl.concat_str([pl.col("a").cast(pl.String), pl.lit("x")]))
    assert_frame_equal(out, pl.DataFrame({"a": []}, schema={"a": pl.String}))


def test_concat_str_ignore_nulls() -> None:
    df = pl.DataFrame({"a": ["a", None, "c"], "b": [None, 2, 3], "c": ["x", "y", "z"]})

    # ignore nulls
    out = df.select([pl.concat_str(["a", "b", "c"], separator="-", ignore_nulls=True)])
    assert out["a"].to_list() == ["a-x", "2-y", "c-3-z"]
    # propagate nulls
    out = df.select([pl.concat_str(["a", "b", "c"], separator="-", ignore_nulls=False)])
    assert out["a"].to_list() == [None, None, "c-3-z"]


@pytest.mark.parametrize(
    "expr",
    [
        "a" + pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=True),
        "a" + pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=False),
        pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=True) + "a",
        pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=False) + "a",
        pl.lit(None, dtype=pl.String)
        + pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=True),
        pl.lit(None, dtype=pl.String)
        + pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=False),
        pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=True)
        + pl.lit(None, dtype=pl.String),
        pl.concat_str(pl.lit("b"), pl.lit("c"), ignore_nulls=False)
        + pl.lit(None, dtype=pl.String),
        pl.lit(None, dtype=pl.String) + "a",
        "a" + pl.lit(None, dtype=pl.String),
        pl.concat_str(None, ignore_nulls=False)
        + pl.concat_str(pl.lit("b"), ignore_nulls=False),
        pl.concat_str(None, ignore_nulls=True)
        + pl.concat_str(pl.lit("b"), ignore_nulls=True),
    ],
)
def test_simplify_str_addition_concat_str(expr: pl.Expr) -> None:
    ldf = pl.LazyFrame({}).select(expr)
    print(ldf.collect(simplify_expression=True))
    assert_frame_equal(
        ldf.collect(simplify_expression=True), ldf.collect(simplify_expression=False)
    )

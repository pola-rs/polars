from datetime import date

import polars as pl


def test_sqrt_neg_inf() -> None:
    out = pl.DataFrame(
        {
            "val": [float("-Inf"), -9, 0, 9, float("Inf")],
        }
    ).with_column(pl.col("val").sqrt().alias("sqrt"))
    # comparing nans and infinities by string value as they are not cmp
    assert str(out["sqrt"].to_list()) == str(
        [float("NaN"), float("NaN"), 0.0, 3.0, float("Inf")]
    )


def test_arithmetic_with_logical_on_series_4920() -> None:
    assert (pl.Series([date(2022, 6, 3)]) - date(2022, 1, 1)).dtype == pl.Duration("ms")


def test_struct_arithmetic() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    ).select(pl.cumsum(["a", "c"]))
    assert df.select(pl.col("cumsum") * 2).to_dict(False) == {
        "cumsum": [{"a": 2, "c": 12}, {"a": 4, "c": 16}]
    }
    assert df.select(pl.col("cumsum") - 2).to_dict(False) == {
        "cumsum": [{"a": -1, "c": 4}, {"a": 0, "c": 6}]
    }
    assert df.select(pl.col("cumsum") + 2).to_dict(False) == {
        "cumsum": [{"a": 3, "c": 8}, {"a": 4, "c": 10}]
    }
    assert df.select(pl.col("cumsum") / 2).to_dict(False) == {
        "cumsum": [{"a": 0.5, "c": 3.0}, {"a": 1.0, "c": 4.0}]
    }
    assert df.select(pl.col("cumsum") // 2).to_dict(False) == {
        "cumsum": [{"a": 0, "c": 3}, {"a": 1, "c": 4}]
    }

    # inline, this check cumsum reports the right output type
    assert pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).select(
        pl.cumsum(["a", "c"]) * 3
    ).to_dict(False) == {"cumsum": [{"a": 3, "c": 18}, {"a": 6, "c": 24}]}

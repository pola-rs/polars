import polars as pl
from polars.testing import assert_frame_equal


def test_unique_predicate_pd() -> None:
    lf = pl.LazyFrame(
        {
            "x": ["abc", "abc"],
            "y": ["xxx", "xxx"],
            "z": [True, False],
        }
    )

    result = (
        lf.unique(subset=["x", "y"], maintain_order=True, keep="last")
        .filter(pl.col("z"))
        .collect()
    )
    expected = pl.DataFrame(schema={"x": pl.String, "y": pl.String, "z": pl.Boolean})
    assert_frame_equal(result, expected)

    result = (
        lf.unique(subset=["x", "y"], maintain_order=True, keep="any")
        .filter(pl.col("z"))
        .collect()
    )
    expected = pl.DataFrame({"x": ["abc"], "y": ["xxx"], "z": [True]})
    assert_frame_equal(result, expected)


def test_unique_on_list_df() -> None:
    assert pl.DataFrame(
        {"a": [1, 2, 3, 4, 4], "b": [[1, 1], [2], [3], [4, 4], [4, 4]]}
    ).unique(maintain_order=True).to_dict(as_series=False) == {
        "a": [1, 2, 3, 4],
        "b": [[1, 1], [2], [3], [4, 4]],
    }

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_rank_nulls() -> None:
    assert pl.Series([]).rank().to_list() == []
    assert pl.Series([None]).rank().to_list() == [None]
    assert pl.Series([None, None]).rank().to_list() == [None, None]


def test_rank_random_expr() -> None:
    df = pl.from_dict(
        {"a": [1] * 5, "b": [1, 2, 3, 4, 5], "c": [200, 100, 100, 50, 100]}
    )

    df_ranks1 = df.with_columns(
        pl.col("c").rank(method="random", seed=1).over("a").alias("rank")
    )
    df_ranks2 = df.with_columns(
        pl.col("c").rank(method="random", seed=1).over("a").alias("rank")
    )
    assert_frame_equal(df_ranks1, df_ranks2)


def test_rank_random_series() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    assert_series_equal(
        s.rank("random", seed=1), pl.Series("a", [2, 4, 7, 3, 5, 6, 1], dtype=pl.UInt32)
    )


def test_rank_df() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
        }
    )

    s = df.select(pl.col("a").rank(method="average").alias("b")).to_series()
    assert s.to_list() == [1.5, 1.5, 3.5, 3.5, 5.0]
    assert s.dtype == pl.Float64

    s = df.select(pl.col("a").rank(method="max").alias("b")).to_series()
    assert s.to_list() == [2, 2, 4, 4, 5]
    assert s.dtype == pl.get_index_type()


def test_rank_so_4109() -> None:
    # also tests ranks null behavior
    df = pl.from_dict(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "rank": [None, 3, 2, 4, 1, 4, 3, 2, 1, None, 3, 4, 4, 1, None, 3],
        }
    ).sort(by=["id", "rank"])

    assert df.group_by("id").agg(
        [
            pl.col("rank").alias("original"),
            pl.col("rank").rank(method="dense").alias("dense"),
            pl.col("rank").rank(method="average").alias("average"),
        ]
    ).to_dict(as_series=False) == {
        "id": [1, 2, 3, 4],
        "original": [[None, 2, 3, 4], [1, 2, 3, 4], [None, 1, 3, 4], [None, 1, 3, 4]],
        "dense": [[None, 1, 2, 3], [1, 2, 3, 4], [None, 1, 2, 3], [None, 1, 2, 3]],
        "average": [
            [None, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [None, 1.0, 2.0, 3.0],
            [None, 1.0, 2.0, 3.0],
        ],
    }


def test_rank_string_null_11252() -> None:
    rank = pl.Series([None, "", "z", None, "a"]).rank()
    assert rank.to_list() == [None, 1.0, 3.0, None, 2.0]


def test_rank_series() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert_series_equal(
        s.rank("dense"), pl.Series("a", [2, 3, 4, 3, 3, 4, 1], dtype=pl.UInt32)
    )

    df = pl.DataFrame([s])
    assert df.select(pl.col("a").rank("dense"))["a"].to_list() == [2, 3, 4, 3, 3, 4, 1]

    assert_series_equal(
        s.rank("dense", descending=True),
        pl.Series("a", [3, 2, 1, 2, 2, 1, 4], dtype=pl.UInt32),
    )

    assert s.rank(method="average").dtype == pl.Float64
    assert s.rank(method="max").dtype == pl.get_index_type()

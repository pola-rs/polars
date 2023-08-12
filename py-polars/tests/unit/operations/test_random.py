import random

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_shuffle_reseed() -> None:
    assert pl.DataFrame({"x": [1, 2, 3, 1, 2, 3], "c": [0, 0, 0, 1, 1, 1]}).groupby(
        "c", maintain_order=True
    ).agg(pl.col("x").shuffle(2)).to_dict(False) == {
        "c": [0, 1],
        "x": [[2, 1, 3], [3, 1, 2]],
    }


def test_sample_expr() -> None:
    a = pl.Series("a", range(0, 20))
    out = pl.select(
        pl.lit(a).sample(fraction=0.5, with_replacement=False, seed=1)
    ).to_series()

    assert out.shape == (10,)
    assert out.to_list() != out.sort().to_list()
    assert out.unique().shape == (10,)
    assert set(out).issubset(set(a))

    out = pl.select(pl.lit(a).sample(n=10, with_replacement=False, seed=1)).to_series()
    assert out.shape == (10,)
    assert out.to_list() != out.sort().to_list()
    assert out.unique().shape == (10,)

    # Setting random.seed should lead to reproducible results
    random.seed(1)
    result1 = pl.select(pl.lit(a).sample(n=10)).to_series()
    random.seed(1)
    result2 = pl.select(pl.lit(a).sample(n=10)).to_series()
    assert_series_equal(result1, result2)


def test_sample_df() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]})

    assert df.sample(n=2, seed=0).shape == (2, 3)
    assert df.sample(fraction=0.4, seed=0).shape == (1, 3)


def test_sample_series() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5])

    assert len(s.sample(n=2, seed=0)) == 2
    assert len(s.sample(fraction=0.4, seed=0)) == 2

    assert len(s.sample(n=2, with_replacement=True, seed=0)) == 2

    # on a series of length 5, you cannot sample more than 5 items
    with pytest.raises(pl.ShapeError):
        s.sample(n=10, with_replacement=False, seed=0)
    # unless you use with_replacement=True
    assert len(s.sample(n=10, with_replacement=True, seed=0)) == 10


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


def test_shuffle_expr() -> None:
    # setting 'random.seed' should lead to reproducible results
    s = pl.Series("a", range(20))
    s_list = s.to_list()

    random.seed(1)
    result1 = pl.select(pl.lit(s).shuffle()).to_series()

    random.seed(1)
    result2 = pl.select(a=pl.lit(s_list).shuffle()).to_series()
    assert_series_equal(result1, result2)


def test_shuffle_series() -> None:
    a = pl.Series("a", [1, 2, 3])
    out = a.shuffle(2)
    expected = pl.Series("a", [2, 1, 3])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).shuffle(2)).to_series()
    assert_series_equal(out, expected)

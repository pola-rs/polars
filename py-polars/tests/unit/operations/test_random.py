from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_shuffle_group_by_reseed() -> None:
    def unique_shuffle_groups(n: int, seed: int | None) -> int:
        ls = [1, 2, 3] * n  # 1, 2, 3, 1, 2, 3...
        groups = sorted(list(range(n)) * 3)  # 0, 0, 0, 1, 1, 1, ...
        df = pl.DataFrame({"l": ls, "group": groups})
        shuffled = df.group_by("group", maintain_order=True).agg(
            pl.col("l").shuffle(seed)
        )
        num_unique = shuffled.group_by("l").agg(pl.lit(0)).select(pl.len())
        return int(num_unique[0, 0])

    assert unique_shuffle_groups(50, None) > 1  # Astronomically unlikely.
    assert (
        unique_shuffle_groups(50, 0xDEADBEEF) == 1
    )  # Fixed seed should be always the same.


def test_sample_expr() -> None:
    a = pl.Series("a", range(20))
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

    # pl.set_random_seed should lead to reproducible results.
    pl.set_random_seed(1)
    result1 = pl.select(pl.lit(a).sample(n=10)).to_series()
    pl.set_random_seed(1)
    result2 = pl.select(pl.lit(a).sample(n=10)).to_series()
    assert_series_equal(result1, result2)


def test_sample_df() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]})

    assert df.sample().shape == (1, 3)
    assert df.sample(n=2, seed=0).shape == (2, 3)
    assert df.sample(fraction=0.4, seed=0).shape == (1, 3)
    assert df.sample(n=pl.Series([2]), seed=0).shape == (2, 3)
    assert df.sample(fraction=pl.Series([0.4]), seed=0).shape == (1, 3)
    assert df.select(pl.col("foo").sample(n=pl.Series([2]), seed=0)).shape == (2, 1)
    assert df.select(pl.col("foo").sample(fraction=pl.Series([0.4]), seed=0)).shape == (
        1,
        1,
    )
    with pytest.raises(ValueError, match="cannot specify both `n` and `fraction`"):
        df.sample(n=2, fraction=0.4)


def test_sample_n_expr() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 2],
            "val": [1, 2, 3, 2, 1, 1],
        }
    )

    out_df = df.sample(pl.Series([3]), seed=0)
    expected_df = pl.DataFrame({"group": [2, 2, 1], "val": [1, 1, 3]})
    assert_frame_equal(out_df, expected_df)

    agg_df = df.group_by("group", maintain_order=True).agg(
        pl.col("val").sample(pl.col("val").max(), seed=0)
    )
    expected_df = pl.DataFrame({"group": [1, 2], "val": [[1, 2, 3], [1, 1]]})
    assert_frame_equal(agg_df, expected_df)

    select_df = df.select(pl.col("val").sample(pl.col("val").max(), seed=0))
    expected_df = pl.DataFrame({"val": [1, 1, 3]})
    assert_frame_equal(select_df, expected_df)


def test_sample_empty_df() -> None:
    df = pl.DataFrame({"foo": []})

    # // If with replacement, then expect empty df
    assert df.sample(n=3, with_replacement=True).shape == (0, 1)
    assert df.sample(fraction=0.4, with_replacement=True).shape == (0, 1)

    # // If without replacement, then expect shape mismatch on sample_n not sample_frac
    with pytest.raises(ShapeError):
        df.sample(n=3, with_replacement=False)
    assert df.sample(fraction=0.4, with_replacement=False).shape == (0, 1)


def test_sample_series() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5])

    assert len(s.sample(n=2, seed=0)) == 2
    assert len(s.sample(fraction=0.4, seed=0)) == 2

    assert len(s.sample(n=2, with_replacement=True, seed=0)) == 2

    # on a series of length 5, you cannot sample more than 5 items
    with pytest.raises(ShapeError):
        s.sample(n=10, with_replacement=False, seed=0)
    # unless you use with_replacement=True
    assert len(s.sample(n=10, with_replacement=True, seed=0)) == 10


def test_shuffle_expr() -> None:
    # pl.set_random_seed should lead to reproducible results.
    s = pl.Series("a", range(20))

    pl.set_random_seed(1)
    result1 = pl.select(pl.lit(s).shuffle()).to_series()

    pl.set_random_seed(1)
    result2 = pl.select(pl.lit(s).shuffle()).to_series()
    assert_series_equal(result1, result2)


def test_shuffle_series() -> None:
    a = pl.Series("a", [1, 2, 3])
    out = a.shuffle(2)
    expected = pl.Series("a", [2, 1, 3])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).shuffle(2)).to_series()
    assert_series_equal(out, expected)


def test_sample_16232() -> None:
    k = 2
    p = 0

    df = pl.DataFrame({"a": [p] * k + [1 + p], "b": [[1] * p] * k + [range(1, p + 2)]})
    assert df.select(pl.col("b").list.sample(n=pl.col("a"), seed=0)).to_dict(
        as_series=False
    ) == {"b": [[], [], [1]]}

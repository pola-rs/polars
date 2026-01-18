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
    expected_df = pl.DataFrame({"group": [2, 1, 1], "val": [1, 2, 3]})
    assert_frame_equal(out_df, expected_df)

    agg_df = df.group_by("group", maintain_order=True).agg(
        pl.col("val").sample(pl.col("val").max(), seed=0)
    )
    expected_df = pl.DataFrame({"group": [1, 2], "val": [[1, 2, 3], [2, 1]]})
    assert_frame_equal(agg_df, expected_df)

    select_df = df.select(pl.col("val").sample(pl.col("val").max(), seed=0))
    expected_df = pl.DataFrame({"val": [1, 2, 3]})
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
    out = a.shuffle(1)
    expected = pl.Series("a", [2, 3, 1])
    assert_series_equal(out, expected)

    out = pl.select(pl.lit(a).shuffle(1)).to_series()
    assert_series_equal(out, expected)


def test_sample_16232() -> None:
    k = 2
    p = 0

    df = pl.DataFrame({"a": [p] * k + [1 + p], "b": [[1] * p] * k + [range(1, p + 2)]})
    assert df.select(pl.col("b").list.sample(n=pl.col("a"), seed=0)).to_dict(
        as_series=False
    ) == {"b": [[], [], [1]]}


def _split_exact_n(df: pl.LazyFrame, height: int, n: int) -> list[pl.LazyFrame]:
    base, rem = divmod(height, n)
    out = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append(df.slice(start, size))
        start += size
    return out


def _force_n_chunks_like(df: pl.LazyFrame, height: int, n: int) -> pl.LazyFrame:
    parts = _split_exact_n(df, height, n)
    # keep order, build from multiple record batches
    return pl.concat(parts, rechunk=False)


def test_sample_lazy_frame_bernoulli() -> None:
    """Test LazyFrame.sample uses Bernoulli sampling (probabilistic, order-preserving)."""
    n = 10000
    lf = pl.LazyFrame({"foo": range(n), "bar": range(n, 2 * n)})

    fraction = 0.3
    result = lf.sample(fraction=fraction, seed=42).collect()

    # Bernoulli: expect ~n*fraction rows, Binomial(n, fraction) distribution
    expected = n * fraction
    std = (n * fraction * (1 - fraction)) ** 0.5
    assert abs(result.shape[0] - expected) < 5 * std
    assert result.shape[1] == 2

    # Test that sampled values come from original data
    assert result["foo"].is_in(lf.collect()["foo"]).all()


def test_sample_lazy_frame_preserves_order() -> None:
    """Test that Bernoulli sampling preserves row order."""
    lf = pl.LazyFrame({"foo": range(1000)})

    result_without_replacement = lf.sample(fraction=0.5, seed=0).collect()
    values_without_replacement = result_without_replacement["foo"].to_list()
    assert values_without_replacement == sorted(values_without_replacement)

    result_with_replacement = lf.sample(fraction=0.5, with_replacement=True, seed=0).collect()
    values_with_replacement = result_with_replacement["foo"].to_list()
    assert values_with_replacement == sorted(values_with_replacement)

    result_without_replacement_force_n_chunks = _force_n_chunks_like(lf, 1000, 10).sample(fraction=0.5, seed=0).collect()
    values_without_replacement_force_n_chunks = result_without_replacement_force_n_chunks["foo"].to_list()
    assert values_without_replacement_force_n_chunks == sorted(values_without_replacement_force_n_chunks)


def test_sample_lazy_frame_with_replacement() -> None:
    """Test LazyFrame.sample with replacement uses Poisson sampling."""
    n = 1000
    lf = pl.LazyFrame({"foo": range(n)})

    fraction = 0.6
    result = lf.sample(fraction=fraction, with_replacement=True, seed=42).collect()

    # Poisson: expect ~n*fraction rows, sum of Poisson(fraction) has mean=n*fraction, var=n*fraction
    expected = n * fraction
    std = (n * fraction) ** 0.5
    assert abs(result.shape[0] - expected) < 5 * std

    # Some values should appear multiple times
    assert result["foo"].n_unique() < result.shape[0]


def test_sample_lazy_frame_series_fraction() -> None:
    """Test LazyFrame.sample with Series for fraction parameter."""
    n = 1000
    lf = pl.LazyFrame({"foo": range(n), "bar": range(n, 2 * n)})

    # Test with Series for fraction
    fraction = 0.4
    result = lf.sample(fraction=pl.Series([fraction]), seed=0).collect()

    # Bernoulli: expect ~n*fraction rows
    expected = n * fraction
    std = (n * fraction * (1 - fraction)) ** 0.5
    assert abs(result.shape[0] - expected) < 5 * std


def test_sample_lazy_frame_reproducibility() -> None:
    """Test that LazyFrame.sample is reproducible with same seed."""
    lf = pl.LazyFrame({"foo": range(100)})

    result1 = lf.sample(fraction=0.5, seed=42).collect()
    result2 = lf.sample(fraction=0.5, seed=42).collect()

    assert_frame_equal(result1, result2)

    lf_10_chunks = _force_n_chunks_like(lf, 100, 10)
    result1_10_chunks = lf_10_chunks.sample(fraction=0.5, seed=42, with_replacement=True).collect()
    result2_10_chunks = lf_10_chunks.sample(fraction=0.5, seed=42, with_replacement=True).collect()
    assert_frame_equal(result1_10_chunks, result2_10_chunks)


def test_sample_lazy_frame_different_seeds() -> None:
    """Test that LazyFrame.sample produces different results with different seeds."""
    lf = pl.LazyFrame({"foo": range(100)})

    result1 = lf.sample(fraction=0.5, seed=1).collect()
    result2 = lf.sample(fraction=0.5, seed=2).collect()

    # Results should be different (with very high probability for 100 rows)
    assert not result1.equals(result2)


def test_sample_lazy_frame_empty() -> None:
    """Test LazyFrame.sample with empty DataFrame."""
    lf = pl.LazyFrame({"foo": []})

    # Empty DataFrame should remain empty
    result = lf.sample(fraction=0.5, seed=0).collect()
    assert result.shape == (0, 1)

    # With replacement should also be empty
    result = lf.sample(fraction=0.5, with_replacement=True, seed=0).collect()
    assert result.shape == (0, 1)


def test_sample_lazy_frame_without_replacement_over_100() -> None:
    """Test that sampling > 100% without replacement raises error."""
    lf = pl.LazyFrame({"foo": [1, 2, 3, 4, 5]})

    # Without replacement, fraction > 1.0 should error
    with pytest.raises(pl.exceptions.ComputeError):
        lf.sample(fraction=2.0, with_replacement=False, seed=0).collect()


def test_sample_lazy_frame_invalid_series_fraction() -> None:
    """Test that Series with more than one element raises error."""
    lf = pl.LazyFrame({"foo": [1, 2, 3]})

    with pytest.raises(ValueError, match="Sample fraction must be a single value"):
        lf.sample(fraction=pl.Series([0.5, 0.6]), seed=0).collect()


def test_sample_lazy_frame_small_fraction() -> None:
    """Test sampling with very small fraction."""
    n = 100000
    lf = pl.LazyFrame({"foo": range(n)})

    fraction = 0.001
    result = lf.sample(fraction=fraction, seed=42).collect()

    # Should get approximately 100 rows (Binomial(100000, 0.001))
    expected = n * fraction
    std = (n * fraction * (1 - fraction)) ** 0.5
    assert abs(result.shape[0] - expected) < 5 * std

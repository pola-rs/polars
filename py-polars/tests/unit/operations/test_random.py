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


# ---------------------------------------------------------------------------
# Reproducibility of `.over()` with `pl.set_random_seed` — issues #27307, #15464
# ---------------------------------------------------------------------------


# Category 1: Core reproducibility — parametrised 20x to catch scheduling-
# dependent regressions that a single-pair comparison might pass by coincidence.


@pytest.mark.parametrize("run", range(20))
def test_shuffle_over_respects_global_seed_15464(run: int) -> None:
    df = pl.DataFrame(
        {
            "grp": ["A", "A", "A", "B", "B", "C", "C", "D"],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.int_range(pl.len()).shuffle().over("grp"))

    pl.set_random_seed(0)
    result2 = df.with_columns(pl.int_range(pl.len()).shuffle().over("grp"))

    assert_frame_equal(result1, result2)


@pytest.mark.parametrize("run", range(20))
def test_sample_over_respects_global_seed(run: int) -> None:
    df = pl.DataFrame(
        {
            "grp": ["A", "A", "A", "B", "B", "C", "C"],
            "val": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    pl.set_random_seed(42)
    result1 = df.with_columns(
        pl.col("val").sample(n=pl.len(), shuffle=True).over("grp")
    )

    pl.set_random_seed(42)
    result2 = df.with_columns(
        pl.col("val").sample(n=pl.len(), shuffle=True).over("grp")
    )

    assert_frame_equal(result1, result2)


@pytest.mark.parametrize("run", range(20))
def test_shuffle_over_reproducible_second_seed(run: int) -> None:
    # Belt-and-braces: different seed + different shape than test 1 to guard
    # against a fix that happens to work only for one (seed, shape) combination.
    df = pl.DataFrame(
        {
            "grp": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "val": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )

    pl.set_random_seed(123)
    result1 = df.with_columns(pl.col("val").shuffle().over("grp"))

    pl.set_random_seed(123)
    result2 = df.with_columns(pl.col("val").shuffle().over("grp"))

    assert_frame_equal(result1, result2)


# Category 2: Per-group seed independence.


def test_shuffle_over_different_per_group_seeds() -> None:
    # With size-10 identical-data groups, P(collision for a fixed seed)
    # ≈ 1/10! ≈ 2.8e-7. Using a fixed seed rather than a search loop —
    # if this specific seed ever collides, change the literal; do NOT
    # reintroduce a loop (CI-hostile).
    n = 10
    df = pl.DataFrame(
        {"grp": ["A"] * n + ["B"] * n, "val": list(range(n)) * 2}
    )

    pl.set_random_seed(0)
    result = df.with_columns(pl.col("val").shuffle().over("grp"))
    a_vals = result.filter(pl.col("grp") == "A")["val"].to_list()
    b_vals = result.filter(pl.col("grp") == "B")["val"].to_list()

    assert a_vals != b_vals, (
        f"Groups A and B got identical shuffles: {a_vals}. "
        "Per-group seeds aren't being differentiated."
    )


def test_shuffle_over_different_seeds_different_results() -> None:
    # Verifies the fix actually consumes the global seed (not hardcoded).
    # Size-10 groups → 10! × 10! ≈ 1.3e13 combinations → no accidental collision.
    n = 10
    df = pl.DataFrame(
        {"grp": ["A"] * n + ["B"] * n, "val": list(range(n * 2))}
    )

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.col("val").shuffle().over("grp"))

    pl.set_random_seed(999)
    result2 = df.with_columns(pl.col("val").shuffle().over("grp"))

    assert not result1.equals(result2)


# Category 3: Backward compatibility — explicit seeds.


def test_shuffle_over_explicit_seed_deterministic() -> None:
    df = pl.DataFrame(
        {
            "grp": ["A", "A", "A", "B", "B", "C", "C", "D"],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    result1 = df.with_columns(pl.col("val").shuffle(seed=42).over("grp"))
    result2 = df.with_columns(pl.col("val").shuffle(seed=42).over("grp"))

    assert_frame_equal(result1, result2)


def test_shuffle_over_explicit_seed_same_per_group() -> None:
    # With identical data and the same explicit seed, both groups must
    # produce identical shuffles — preserving the documented pre-fix behavior.
    df = pl.DataFrame(
        {"grp": ["A", "A", "A", "B", "B", "B"], "val": [1, 2, 3, 1, 2, 3]}
    )

    result = df.with_columns(pl.col("val").shuffle(seed=42).over("grp"))
    a_vals = result.filter(pl.col("grp") == "A")["val"].to_list()
    b_vals = result.filter(pl.col("grp") == "B")["val"].to_list()

    assert a_vals == b_vals


# Category 4: Backward compatibility — sequential operations.


def test_shuffle_over_does_not_break_subsequent_random_ops() -> None:
    # .over() must advance the global RNG by a deterministic count so that
    # downstream random ops (without a seed reset) see identical state.
    df = pl.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})

    pl.set_random_seed(0)
    _ = df.with_columns(pl.col("val").shuffle().over("grp"))
    after1 = df.with_columns(pl.col("val").shuffle())

    pl.set_random_seed(0)
    _ = df.with_columns(pl.col("val").shuffle().over("grp"))
    after2 = df.with_columns(pl.col("val").shuffle())

    assert_frame_equal(after1, after2)


def test_multiple_shuffle_over_calls_reproducible() -> None:
    # Multiple .over() expressions in the same with_columns — exercises
    # the window-cache interaction between partitioned random ops.
    df = pl.DataFrame(
        {
            "grp": ["A", "A", "A", "B", "B", "B"],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
        }
    )

    pl.set_random_seed(0)
    result1 = df.with_columns(
        pl.col("x").shuffle().over("grp").alias("x_shuffled"),
        pl.col("y").shuffle().over("grp").alias("y_shuffled"),
    )

    pl.set_random_seed(0)
    result2 = df.with_columns(
        pl.col("x").shuffle().over("grp").alias("x_shuffled"),
        pl.col("y").shuffle().over("grp").alias("y_shuffled"),
    )

    assert_frame_equal(result1, result2)


# Category 5: Edge cases.


def test_shuffle_over_single_element_groups() -> None:
    df = pl.DataFrame({"grp": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]})

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.col("val").shuffle().over("grp"))

    pl.set_random_seed(0)
    result2 = df.with_columns(pl.col("val").shuffle().over("grp"))

    assert_frame_equal(result1, result2)
    assert_series_equal(result1["val"], df["val"])


def test_shuffle_over_single_group() -> None:
    df = pl.DataFrame({"grp": ["A", "A", "A", "A", "A"], "val": [1, 2, 3, 4, 5]})

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.col("val").shuffle().over("grp"))

    pl.set_random_seed(0)
    result2 = df.with_columns(pl.col("val").shuffle().over("grp"))

    assert_frame_equal(result1, result2)


def test_shuffle_over_many_groups() -> None:
    # 200 groups → rayon will use multiple workers. Most likely scenario
    # to expose a remaining scheduling-dependent regression.
    n_groups = 200
    group_size = 10
    df = pl.DataFrame(
        {
            "grp": [i for i in range(n_groups) for _ in range(group_size)],
            "val": list(range(group_size)) * n_groups,
        }
    )

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.col("val").shuffle().over("grp"))

    pl.set_random_seed(0)
    result2 = df.with_columns(pl.col("val").shuffle().over("grp"))

    assert_frame_equal(result1, result2)


def test_shuffle_over_with_null_values() -> None:
    df = pl.DataFrame(
        {"grp": ["A", "A", "A", "B", "B", "B"], "val": [1, None, 3, None, 5, None]}
    )

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.col("val").shuffle().over("grp"))

    pl.set_random_seed(0)
    result2 = df.with_columns(pl.col("val").shuffle().over("grp"))

    assert_frame_equal(result1, result2)
    # Shuffle is an intra-group permutation, so the total null count is invariant.
    assert result1["val"].null_count() == 3


def test_shuffle_over_multi_column_partition() -> None:
    df = pl.DataFrame(
        {
            "a": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
            "b": [1, 1, 2, 2, 1, 1, 2, 2],
            "val": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.col("val").shuffle().over("a", "b"))

    pl.set_random_seed(0)
    result2 = df.with_columns(pl.col("val").shuffle().over("a", "b"))

    assert_frame_equal(result1, result2)


def test_shuffle_over_unequal_group_sizes() -> None:
    df = pl.DataFrame(
        {
            "grp": ["A"] * 1 + ["B"] * 5 + ["C"] * 20 + ["D"] * 2,
            "val": list(range(28)),
        }
    )

    pl.set_random_seed(0)
    result1 = df.with_columns(pl.col("val").shuffle().over("grp"))

    pl.set_random_seed(0)
    result2 = df.with_columns(pl.col("val").shuffle().over("grp"))

    assert_frame_equal(result1, result2)


# Category 6: Non-regression — non-`.over()` paths must be untouched.


def test_shuffle_without_over_still_works() -> None:
    s = pl.Series("a", range(20))

    pl.set_random_seed(1)
    result1 = pl.select(pl.lit(s).shuffle()).to_series()

    pl.set_random_seed(1)
    result2 = pl.select(pl.lit(s).shuffle()).to_series()

    assert_series_equal(result1, result2)


def test_shuffle_group_by_agg_still_works() -> None:
    # group_by().agg() takes a different code path than .over() — this
    # test pins the non-regression.
    df = pl.DataFrame(
        {"grp": ["A", "A", "A", "B", "B", "B"], "val": [1, 2, 3, 4, 5, 6]}
    )

    result1 = df.group_by("grp", maintain_order=True).agg(
        pl.col("val").shuffle(seed=0xDEADBEEF)
    )
    result2 = df.group_by("grp", maintain_order=True).agg(
        pl.col("val").shuffle(seed=0xDEADBEEF)
    )

    assert_frame_equal(result1, result2)

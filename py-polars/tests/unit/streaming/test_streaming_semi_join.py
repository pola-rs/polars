"""Streaming semi joins built from either side."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars._typing import JoinBuildSide
    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")

BUILD_SIDES: list[JoinBuildSide] = [
    "auto",
    "force_left",
    "force_right",
    "prefer_left",
    "prefer_right",
]


def assert_semi(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    expected: pl.DataFrame,
    *,
    build_side: JoinBuildSide = "auto",
    **kwargs: Any,
) -> pl.LazyFrame:
    """Check the semi join, and the anti join as the rest of the left rows."""
    q = left.join(right, how="semi", build_side=build_side, **kwargs)
    reference = q.collect(engine="in-memory", optimizations=pl.QueryOptFlags.none())
    out = q.collect(engine="streaming")
    assert_frame_equal(out, expected, check_row_order=False)
    assert_frame_equal(reference, expected, check_row_order=False)

    anti = left.join(right, how="anti", build_side=build_side, **kwargs)
    anti_out = anti.collect(engine="streaming")
    assert_frame_equal(
        anti_out,
        anti.collect(engine="in-memory", optimizations=pl.QueryOptFlags.none()),
        check_row_order=False,
    )
    assert anti_out.height + out.height == left.collect().height
    return q


def build_side_chosen(
    q: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> str | None:
    """Collect on the streaming engine and report the sampled build side."""
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    q.collect(engine="streaming")
    err = capfd.readouterr().err
    lines = [line for line in err.splitlines() if "join build side chosen" in line]
    assert len(lines) <= 1, err
    return lines[0].rsplit(" ", 1)[1] if lines else None


@pytest.mark.parametrize("build_side", BUILD_SIDES)
def test_duplicates_on_both_sides(build_side: JoinBuildSide) -> None:
    # Every left row with a matching key survives once, however often the
    # key occurs on the right.
    left = pl.LazyFrame({"k": [1, 1, 2, 3, 3, 3], "v": list("abcdef")})
    right = pl.LazyFrame({"k": [3, 3, 3, 1, 1, 9]})
    expected = pl.DataFrame({"k": [1, 1, 3, 3, 3], "v": list("abdef")})
    assert_semi(left, right, expected, on="k", build_side=build_side)


@pytest.mark.parametrize("build_side", BUILD_SIDES)
def test_empty_sides(build_side: JoinBuildSide) -> None:
    left = pl.LazyFrame({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    right = pl.LazyFrame({"k": [1, 2, 3]})
    empty_left = left.clear()
    empty_right = right.clear()
    assert_semi(empty_left, right, empty_left.collect(), on="k", build_side=build_side)
    assert_semi(left, empty_right, empty_left.collect(), on="k", build_side=build_side)
    assert_semi(
        empty_left, empty_right, empty_left.collect(), on="k", build_side=build_side
    )


@pytest.mark.parametrize("build_side", BUILD_SIDES)
def test_disjoint_keys(build_side: JoinBuildSide) -> None:
    left = pl.LazyFrame({"k": [1, 2, 3], "v": [1, 2, 3]})
    right = pl.LazyFrame({"k": [4, 5, 6]})
    assert_semi(left, right, left.clear().collect(), on="k", build_side=build_side)


@pytest.mark.parametrize("build_side", BUILD_SIDES)
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_null_keys(build_side: JoinBuildSide, nulls_equal: bool) -> None:
    left = pl.LazyFrame({"k": [None, 1, None, 2, 3], "v": [0, 1, 2, 3, 4]})
    right = pl.LazyFrame({"k": [None, 2, 3, None]})
    if nulls_equal:
        expected = pl.DataFrame({"k": [None, None, 2, 3], "v": [0, 2, 3, 4]})
    else:
        expected = pl.DataFrame({"k": [2, 3], "v": [3, 4]})
    assert_semi(
        left, right, expected, on="k", nulls_equal=nulls_equal, build_side=build_side
    )


@pytest.mark.parametrize("build_side", BUILD_SIDES)
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_composite_and_string_keys(
    build_side: JoinBuildSide, nulls_equal: bool
) -> None:
    left = pl.LazyFrame(
        {
            "a": [1, 1, 2, 2, None, 3],
            "b": ["x", "y", "x", None, "x", "long string key beyond twelve"],
            "v": [0, 1, 2, 3, 4, 5],
        }
    )
    right = pl.LazyFrame(
        {
            "a": [1, 2, None, 3, 3],
            "b": ["y", None, "x", "long string key beyond twelve", "other"],
        }
    )
    if nulls_equal:
        expected = left.collect().filter(pl.col("v").is_in([1, 3, 4, 5]))
    else:
        expected = left.collect().filter(pl.col("v").is_in([1, 5]))
    assert_semi(
        left,
        right,
        expected,
        on=["a", "b"],
        nulls_equal=nulls_equal,
        build_side=build_side,
    )
    # A single string key uses a different hash table than composite keys.
    expected = left.collect()
    if not nulls_equal:
        expected = expected.filter(pl.col("v") != 3)
    assert_semi(
        left,
        right,
        expected,
        on="b",
        nulls_equal=nulls_equal,
        build_side=build_side,
    )


@pytest.mark.parametrize("build_side", BUILD_SIDES)
def test_skewed_and_large(build_side: JoinBuildSide) -> None:
    # One hot key on the right, many left rows with duplicate keys, more rows
    # than one morsel.
    rng = np.random.default_rng(0)
    left_keys = rng.integers(0, 50_000, 300_000)
    left = pl.LazyFrame({"k": left_keys, "v": np.arange(300_000)})
    right_keys = np.concatenate(
        [np.full(200_000, 7), rng.integers(0, 100_000, 100_000)]
    )
    right = pl.LazyFrame({"k": right_keys, "w": np.arange(300_000)})
    expected = left.collect().filter(pl.col("k").is_in(np.unique(right_keys)))
    assert expected.height > 0
    assert_semi(left, right, expected, on="k", build_side=build_side)


def test_downstream_limit() -> None:
    # A limit stops the join early, whichever side is built.
    left = pl.LazyFrame({"k": np.arange(200_000) % 1000, "v": np.arange(200_000)})
    right = pl.LazyFrame({"k": np.arange(0, 1000, 3)})
    for build_side in BUILD_SIDES:
        q = left.join(right, on="k", how="semi", build_side=build_side).head(10)
        out = q.collect(engine="streaming")
        assert out.height == 10
        assert (out["k"] % 3 == 0).all()
        assert_frame_equal(
            out,
            left.collect().filter(pl.col("v").is_in(out["v"])),
            check_row_order=False,
        )


def test_sampling_builds_smaller_left(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10000")
    small = pl.LazyFrame({"k": np.arange(0, 2000, 2), "v": np.arange(1000)})
    large = pl.LazyFrame({"k": np.arange(500_000) % 5000, "w": np.arange(500_000)})
    expected = small.collect().filter(pl.col("k") < 5000)

    q = assert_semi(small, large, expected, on="k")
    assert build_side_chosen(q, plmonkeypatch, capfd) == "left"
    q = small.join(large, on="k", how="anti")
    assert build_side_chosen(q, plmonkeypatch, capfd) == "left"

    # The large side is complete and its distinct keys are few: it is built.
    expected = large.collect().filter((pl.col("k") % 2 == 0) & (pl.col("k") < 2000))
    q = assert_semi(large, small, expected, on="k")
    assert build_side_chosen(q, plmonkeypatch, capfd) == "right"
    q = large.join(small, on="k", how="anti")
    assert build_side_chosen(q, plmonkeypatch, capfd) == "right"


def test_sampling_keeps_right_build_for_wide_left(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The left side has few rows but retains far more bytes than the distinct
    # keys of the right side, so the right side is built.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10000")
    wide = pl.LazyFrame({"k": np.arange(1000), "payload": ["x" * 2000] * 1000})
    right = pl.LazyFrame({"k": np.arange(500_000) % 100})
    expected = wide.collect().filter(pl.col("k") < 100)
    q = assert_semi(wide, right, expected, on="k")
    assert build_side_chosen(q, plmonkeypatch, capfd) == "right"


def test_sampling_keeps_right_build_for_incomplete_left(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The right side ends after one wide key while the narrow left side has
    # only been sampled. Sampling stops, but the left side is not complete,
    # so it is not built however few bytes its sample retains. An IO plugin
    # feeds the left side one batch at a time, unlike an in-memory frame.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10000000")

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        for i in range(60):
            yield pl.DataFrame(
                {"k": ["a"] * 50_000, "v": np.arange(50_000) + i * 50_000}
            )

    left = register_io_source(source, schema={"k": pl.String, "v": pl.Int64})
    right = pl.LazyFrame({"k": ["x" * 8_000_000]})
    q = assert_semi(left, right, left.clear().collect(), on="k")
    assert build_side_chosen(q, plmonkeypatch, capfd) == "right"


def test_sampling_hints(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # Both sides exceed the sample, so only a hint picks the left side.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "100")
    left = pl.LazyFrame({"k": np.arange(10_000), "v": np.arange(10_000)})
    right = pl.LazyFrame({"k": np.arange(0, 20_000, 4)})
    expected = left.collect().filter(pl.col("k") % 4 == 0)
    chosen: dict[JoinBuildSide, str | None] = {
        "auto": "right",
        "prefer_left": "left",
        "prefer_right": "right",
        "force_left": None,
        "force_right": None,
    }
    for build_side, side in chosen.items():
        q = assert_semi(left, right, expected, on="k", build_side=build_side)
        assert build_side_chosen(q, plmonkeypatch, capfd) == side


def test_ordered_and_is_in_keep_right_build(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10000")
    left = pl.LazyFrame({"k": [5, 3, 5, 1, 2, 3], "v": [0, 1, 2, 3, 4, 5]})
    right = pl.LazyFrame({"k": np.arange(100_000) % 4})

    # An ordered semi join keeps the left order and is never built from the left.
    q = left.join(right, on="k", how="semi", maintain_order="left")
    expected = pl.DataFrame({"k": [3, 1, 2, 3], "v": [1, 3, 4, 5]})
    assert_frame_equal(q.collect(engine="streaming"), expected)
    assert build_side_chosen(q, plmonkeypatch, capfd) is None

    q = left.join(
        right, on="k", how="semi", maintain_order="left", build_side="force_left"
    )
    assert_frame_equal(q.collect(engine="streaming"), expected)

    q = left.select(pl.col("k").is_in(right.select("k").collect()["k"].implode()))
    expected = pl.DataFrame({"k": [False, True, False, True, True, True]})
    assert_frame_equal(q.collect(engine="streaming"), expected)


def test_anti_null_keys_sampled_left_build(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # Null left keys must come out of a left-built anti join unless a null on
    # the right matches them.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10000")
    small = pl.LazyFrame({"k": [None, 0, None, 1, 10, 5], "v": [0, 1, 2, 3, 4, 5]})
    large = pl.LazyFrame({"k": np.arange(0, 4_000_000, 2)}).with_columns(
        pl.when(pl.col("k") == 2).then(None).otherwise(pl.col("k")).alias("k")
    )
    for nulls_equal, kept in [(False, [0, 2, 3, 5]), (True, [3, 5])]:
        q = small.join(large, on="k", how="anti", nulls_equal=nulls_equal)
        expected = small.collect().filter(pl.col("v").is_in(kept))
        assert_frame_equal(
            q.collect(engine="streaming"), expected, check_row_order=False
        )
        assert build_side_chosen(q, plmonkeypatch, capfd) == "left"


def test_anti_sampling_counts_null_keys_it_keeps(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # A left-built anti join keeps its null-key rows as groups, so the
    # sample must count them or the left side looks cheaper than it is.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10000")
    n = 1000
    left = pl.LazyFrame(
        {
            "a": pl.Series([None] * n, dtype=pl.Int64),
            "b": [f"{i:0>40}" for i in range(n)],
            "v": np.arange(n),
        }
    )
    right = pl.LazyFrame(
        {"a": np.arange(1500) % 3, "b": [f"{i:0>40}" for i in range(1500)]}
    )
    q = left.join(right, on=["a", "b"], how="anti")
    assert_frame_equal(
        q.collect(engine="streaming"), left.collect(), check_row_order=False
    )
    assert build_side_chosen(q, plmonkeypatch, capfd) == "right"


def test_left_build_output_projection() -> None:
    # Output columns are a subset of the left columns, in a projected order.
    left = pl.LazyFrame({"a": [1, 2, 3], "k": [1, 2, 3], "b": ["x", "y", "z"]})
    right = pl.LazyFrame({"k": [3, 1]})
    q = left.join(right, on="k", how="semi", build_side="force_left").select("b", "a")
    expected = pl.DataFrame({"b": ["x", "z"], "a": [1, 3]})
    assert_frame_equal(q.collect(engine="streaming"), expected, check_row_order=False)

    q = left.join(right, on="k", how="semi", build_side="force_left").select(pl.len())
    assert q.collect(engine="streaming").item() == 2

    q = left.join(right, on="k", how="anti", build_side="force_left").select("b", "a")
    expected = pl.DataFrame({"b": ["y"], "a": [2]})
    assert_frame_equal(q.collect(engine="streaming"), expected)

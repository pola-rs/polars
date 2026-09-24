"""The build side a streaming equi join picks when both sides fill the sample."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")

# Every side below has more rows than this, so both fill the sample.
SAMPLE_LIMIT = "100"


def unbounded(df: pl.DataFrame, repeats: int = 1) -> pl.LazyFrame:
    # An explode has no row bound, so this side fills the sample. Each key is
    # repeated `repeats` times in a row.
    return df.lazy().with_columns(pl.col("k").repeat_by(repeats)).explode("k")


def dim(n: int = 500) -> pl.DataFrame:
    return pl.DataFrame({"k": range(n), "d": range(n)})


def fact(n: int = 500) -> pl.DataFrame:
    return pl.DataFrame({"k": range(n), "v": range(n)})


def build_side_chosen(
    q: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> tuple[pl.DataFrame, str]:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    lines = capfd.readouterr().err.splitlines()
    # Only one join samples both sides, so only one scores them.
    assert len([line for line in lines if "estimated build scores are" in line]) == 1
    chosen = [line for line in lines if "build side chosen:" in line]
    assert len(chosen) == 1, lines
    return out, chosen[0].rsplit(" ", 1)[1]


def assert_matches_in_memory(q: pl.LazyFrame, out: pl.DataFrame) -> None:
    expected = q.collect(engine="in-memory")
    assert_frame_equal(out, expected, check_row_order=False)


@pytest.mark.parametrize("dim_left", [False, True])
def test_unique_side_is_built_against_a_duplicated_side(
    dim_left: bool, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", SAMPLE_LIMIT)
    d = unbounded(dim())
    f = unbounded(fact(), repeats=10)
    q = d.join(f, on="k") if dim_left else f.join(d, on="k")
    out, side = build_side_chosen(q, plmonkeypatch, capfd)
    assert side == ("left" if dim_left else "right")
    assert_matches_in_memory(q, out)


@pytest.mark.parametrize("dim_left", [False, True])
def test_somewhat_wider_unique_side_is_built(
    dim_left: bool, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The unique side is wider than the referencing side is long, but not by
    # enough to build the referencing side, whichever order they are in.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", SAMPLE_LIMIT)
    d = unbounded(dim().with_columns(pl.col("d").alias(f"d{i}") for i in range(10)))
    f = unbounded(fact().select("k"), repeats=4)
    q = d.join(f, on="k") if dim_left else f.join(d, on="k")
    out, side = build_side_chosen(q, plmonkeypatch, capfd)
    assert side == ("left" if dim_left else "right")
    assert_matches_in_memory(q, out)


def test_wide_unique_side_is_not_built(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", SAMPLE_LIMIT)
    n = 500
    wide = dim(n).with_columns(
        pl.format("{}" + "x" * 200, pl.col("d")).alias(f"s{i}") for i in range(6)
    )
    q = unbounded(fact(n), repeats=3).join(unbounded(wide), on="k")
    out, side = build_side_chosen(q, plmonkeypatch, capfd)
    assert side == "left"
    assert_matches_in_memory(q, out)


def test_mildly_repeated_side_is_compared_by_width(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The left side repeats every other key, too few repeats to be taken as
    # referencing the unique right side, so only the right side's extra column
    # counts and the left side is built.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", SAMPLE_LIMIT)
    left = (
        fact()
        .lazy()
        .with_columns(pl.col("k").repeat_by(pl.col("k") % 2 + 1))
        .explode("k")
    )
    right = unbounded(dim().with_columns(e=pl.col("d")))
    q = left.join(right, on="k")
    out, side = build_side_chosen(q, plmonkeypatch, capfd)
    assert side == "left"
    assert_matches_in_memory(q, out)


def test_similar_sides_build_the_right_side(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", SAMPLE_LIMIT)
    q = unbounded(fact()).join(unbounded(dim()), on="k")
    out, side = build_side_chosen(q, plmonkeypatch, capfd)
    assert side == "right"
    assert_matches_in_memory(q, out)


def test_null_keys_are_not_counted_as_keys(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # Three in four left keys are null and the others are unique, so the left
    # side is built against the right side, which repeats each key three times.
    # Counted as a key, the nulls would make the left side look repeated.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", SAMPLE_LIMIT)
    n = 800
    left = pl.DataFrame(
        {"k": [i if i % 4 == 0 else None for i in range(n)], "v": range(n)}
    )
    q = unbounded(left).join(unbounded(dim(n // 2), repeats=3), on="k")
    out, side = build_side_chosen(q, plmonkeypatch, capfd)
    assert side == "left"
    # Half the non-null left keys are on the right, three times each.
    assert out.height == 3 * n // 8
    assert_matches_in_memory(q, out)

    q = unbounded(left).join(
        unbounded(dim(n // 2), repeats=3), on="k", nulls_equal=True
    )
    out = q.collect(engine="streaming")
    assert_matches_in_memory(q, out)

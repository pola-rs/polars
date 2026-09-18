"""`S ⋈ GroupBy(R)` on grouping keys becomes `S ⋈ GroupBy(R ⋉ S)`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

ON = pl.QueryOptFlags(join_order=True)
OFF = pl.QueryOptFlags(join_order=False)


def scanned(tmp_path: Path, name: str, df: pl.DataFrame) -> pl.LazyFrame:
    # Parquet scans carry the row counts and key ranges the estimator needs.
    df.write_parquet(tmp_path / f"{name}.parquet")
    return pl.scan_parquet(tmp_path / f"{name}.parquet")


def frames(tmp_path: Path, nulls: bool = False) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    n = 100_000
    rows = pl.DataFrame(
        {
            "k": [i // 2 + 1 for i in range(n)],
            "k2": [i % 3 for i in range(n)],
            "v": [float(i % 7) for i in range(n)],
        }
    )
    # Sparse in the key range, with repeated keys and keys nothing groups on.
    keys = [i * 10 + 1 for i in range(100)]
    dim = pl.DataFrame(
        {
            "d": keys + [11, 11, 999_999],
            "w": [i % 4 for i in range(len(keys))] + [5, 6, 7],
        }
    )
    if nulls:
        rows = rows.with_columns(
            k=pl.when(pl.col("k") % 1000 == 1).then(None).otherwise(pl.col("k"))
        )
        dim = dim.with_columns(
            d=pl.when(pl.col("d") == 11).then(None).otherwise(pl.col("d"))
        )
    return scanned(tmp_path, "rows", rows), scanned(tmp_path, "dim", dim)


def restricted(plan: str) -> bool:
    return "SEMI JOIN" in plan and "CACHE" in plan and "PASS THROUGH ABOVE" in plan


def passed_through(
    lf: pl.LazyFrame,
    engine: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> bool:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    lf.collect(engine=engine, optimizations=ON)
    return "semi join passed through" in capfd.readouterr().err


def assert_same_result(lf: pl.LazyFrame) -> None:
    expected = lf.collect(optimizations=OFF)
    for engine in ("in-memory", "streaming"):
        out = lf.collect(engine=engine, optimizations=ON)
        assert out.schema == lf.collect_schema()
        assert_frame_equal(out, expected, check_row_order=False)


def assert_restricted(lf: pl.LazyFrame) -> None:
    assert not restricted(lf.explain(optimizations=OFF))
    assert restricted(lf.explain(optimizations=ON))
    assert_same_result(lf)


def assert_not_restricted(lf: pl.LazyFrame) -> None:
    assert not restricted(lf.explain(optimizations=ON))
    assert_same_result(lf)


def test_restricts_group_by_input(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by("k").agg(
        pl.col("v").sum().alias("total"), pl.len(), pl.col("v").mean().alias("mean")
    )
    lf = dim.join(grouped, left_on="d", right_on="k")
    assert_restricted(lf)
    out = lf.collect(optimizations=ON)
    # Every dimension row keeps its own match; the repeated key comes out thrice.
    assert out.height == 102
    assert out.filter(pl.col("d") == 11)["total"].to_list() == [6.0, 6.0, 6.0]
    assert out["len"].unique().to_list() == [2]


def test_grouped_side_may_be_left(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    assert_restricted(grouped.join(dim, left_on="k", right_on="d"))


def test_join_on_some_of_the_grouping_keys(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by("k", "k2").agg(pl.col("v").sum())
    lf = dim.join(grouped, left_on="d", right_on="k")
    assert_restricted(lf)
    assert lf.collect(optimizations=ON).height == 102 * 2


def test_composite_keys(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    dim = dim.with_columns(k2=pl.col("w") % 3)
    grouped = rows.group_by("k", "k2").agg(pl.col("v").sum())
    assert_restricted(dim.join(grouped, left_on=["d", "k2"], right_on=["k", "k2"]))


def test_aliased_and_renamed_keys(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = (
        rows.group_by(pl.col("k").alias("key"))
        .agg(pl.col("v").sum().alias("total"))
        .select(pl.col("key").alias("id"), pl.col("total") * 2)
        .with_columns(pl.col("id").alias("id2"), half=pl.col("total") / 2)
    )
    assert_restricted(dim.join(grouped, left_on="d", right_on="id2"))


def test_filter_on_aggregate_between(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = (
        rows.group_by("k")
        .agg(pl.col("v").sum().alias("total"))
        .filter(pl.col("total") > 3)
    )
    lf = dim.join(grouped, left_on="d", right_on="k").filter(
        pl.col("w") < pl.col("total")
    )
    assert_restricted(lf)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_null_keys(tmp_path: Path, nulls_equal: bool) -> None:
    rows, dim = frames(tmp_path, nulls=True)
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    lf = dim.join(grouped, left_on="d", right_on="k", nulls_equal=nulls_equal)
    assert_restricted(lf)
    out = lf.collect(optimizations=ON)
    assert out["d"].null_count() == (3 if nulls_equal else 0)


def test_empty_dimension(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    dim = dim.filter(pl.col("w") > 100)
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    lf = dim.join(grouped, left_on="d", right_on="k")
    assert_restricted(lf)
    assert lf.collect(optimizations=ON).height == 0


def test_dimension_shared_with_another_reader(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    lf = pl.concat(
        [
            dim.join(grouped, left_on="d", right_on="k").select("d", "w"),
            dim.select("d", "w"),
        ]
    )
    assert_restricted(lf)


def test_nonselective_dimension_is_left_alone(tmp_path: Path) -> None:
    rows, _ = frames(tmp_path)
    # Every key of the rows is in the dimension.
    dim = scanned(
        tmp_path, "full", pl.DataFrame({"d": list(range(1, 50_001)), "w": 1})
    )
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    assert_not_restricted(dim.join(grouped, left_on="d", right_on="k"))


def test_join_on_aggregate_is_left_alone(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by("k").agg(pl.col("k2").first())
    assert_not_restricted(dim.join(grouped, left_on="w", right_on="k2"))


def test_computed_grouping_key_is_left_alone(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by((pl.col("k") + 1).alias("k")).agg(pl.col("v").sum())
    assert_not_restricted(dim.join(grouped, left_on="d", right_on="k"))


@pytest.mark.parametrize(
    "grouped",
    [
        lambda rows: rows.group_by("k", maintain_order=True).agg(pl.col("v").sum()),
        lambda rows: rows.group_by("k").map_groups(
            lambda df: df.head(1), schema={"k": pl.Int64, "k2": pl.Int64, "v": pl.Float64}
        ),
        lambda rows: rows.group_by("k")
        .agg(pl.col("v").sum())
        .filter(pl.col("v") == pl.col("v").max()),
        lambda rows: rows.group_by("k")
        .agg(pl.col("v").sum())
        .select("k", pl.col("v").cast(pl.Int8, strict=True)),
        lambda rows: rows.group_by("k")
        .agg(pl.col("v").sum())
        .select("k", pl.col("v").map_elements(lambda x: x, return_dtype=pl.Float64)),
    ],
    ids=["maintain_order", "map_groups", "non_elementwise", "fallible", "udf"],
)
def test_unsafe_group_by_is_left_alone(tmp_path: Path, grouped: Any) -> None:
    rows, dim = frames(tmp_path)
    assert_not_restricted(dim.join(grouped(rows), left_on="d", right_on="k"))


def test_sliced_group_by_is_left_alone(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    # Which groups the slice keeps is not fixed, so only the plan is checked.
    grouped = rows.group_by("k").agg(pl.col("v").sum()).head(10)
    lf = dim.join(grouped, left_on="d", right_on="k")
    assert not restricted(lf.explain(optimizations=ON))


def test_dimension_built_from_the_groups_is_left_alone(tmp_path: Path) -> None:
    rows, _ = frames(tmp_path)
    grouped = rows.group_by("k").agg(pl.col("v").sum().alias("total"))
    dim = grouped.filter(pl.col("total") > 4).select(pl.col("k").alias("d"))
    assert_not_restricted(dim.join(grouped, left_on="d", right_on="k"))


def test_shared_groups_are_left_alone(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by("k").agg(pl.col("v").sum().alias("total"))
    lf = pl.concat(
        [
            dim.join(grouped, left_on="d", right_on="k").select("d", "total"),
            grouped.select(pl.col("k").alias("d"), "total"),
        ]
    )
    assert_not_restricted(lf)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_large_dimension_passes_through_at_run_time(
    tmp_path: Path,
    engine: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    rows, _ = frames(tmp_path)
    # The name filter is estimated to keep a fifth of the rows, and keeps them all.
    n = 60_000
    dim = scanned(
        tmp_path, "big", pl.DataFrame({"d": [i % 50_000 + 1 for i in range(n)], "name": "a"})
    ).filter(pl.col("name").str.starts_with("a"))
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    lf = dim.join(grouped, left_on="d", right_on="k")
    assert_restricted(lf)
    assert passed_through(lf, engine, plmonkeypatch, capfd)
    small = frames(tmp_path)[1]
    assert not passed_through(
        small.join(grouped, left_on="d", right_on="k"), engine, plmonkeypatch, capfd
    )


def test_matching_rows_pass_through_at_run_time(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # One hot key the dimension holds: the estimate is selective, the rows are not.
    n = 100_000
    rows = scanned(
        tmp_path,
        "hot",
        pl.DataFrame(
            {
                "k": [1 if i % 10 else i + 1 for i in range(n)],
                "v": [float(i % 7) for i in range(n)],
            }
        ),
    )
    dim = scanned(tmp_path, "low", pl.DataFrame({"d": list(range(1, 1001)), "w": 1}))
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    lf = dim.join(grouped, left_on="d", right_on="k")
    assert_restricted(lf)
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = lf.collect(engine="streaming", optimizations=ON)
    assert "probe rows matched" in capfd.readouterr().err
    assert_frame_equal(out, lf.collect(optimizations=OFF), check_row_order=False)


def test_rewrite_is_idempotent(tmp_path: Path) -> None:
    rows, dim = frames(tmp_path)
    grouped = rows.group_by("k").agg(pl.col("v").sum())
    lf = dim.join(grouped, left_on="d", right_on="k")
    plan = lf.explain(optimizations=ON)
    assert plan.count("SEMI JOIN:") == 1
    assert_restricted(lf)

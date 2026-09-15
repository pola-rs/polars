"""A forced hash join publishes its build-key range to the scan under its probe side."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")

N_ROW_GROUPS = 10
ROWS_PER_GROUP = 100


@pytest.fixture
def fact(tmp_path: Path) -> pl.LazyFrame:
    # Keys sorted into row groups of a hundred, so a key range maps to row groups.
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    df = pl.DataFrame({"k": range(n), "k2": [i % 7 for i in range(n)], "v": range(n)})
    path = tmp_path / "fact.parquet"
    df.write_parquet(path, row_group_size=ROWS_PER_GROUP, statistics="full")
    return pl.scan_parquet(path)


def dim(*keys: int, key: str = "k") -> pl.LazyFrame:
    # Fifty rows: bounded well under an eighth of the fact rows, so it is forced as
    # build side against the unfiltered fact.
    lf = pl.LazyFrame({key: list(range(0, 1000, 20)), "d": list(range(50))})
    if keys:
        lf = lf.filter(pl.col(key).is_in(list(keys)))
    return lf


def tiny(*keys: int, key: str = "k") -> pl.LazyFrame:
    # Few enough rows to be forced against a filtered fact, whose estimate is lower.
    return pl.LazyFrame({key: list(keys), "e": list(range(len(keys)))})


def row_groups_read(
    q: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> tuple[pl.DataFrame, str | None]:
    """Collect on the streaming engine and report what the reader logged."""
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    lines = [line for line in err.splitlines() if "Predicate pushdown: reading" in line]
    assert len(lines) <= 1, err
    return out, lines[0].split("reading ")[1] if lines else None


def assert_matches_in_memory(q: pl.LazyFrame, out: pl.DataFrame) -> None:
    expected = q.collect(engine="in-memory")
    assert_frame_equal(out.sort(out.columns), expected.sort(expected.columns))


def test_range_prunes_row_groups(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    q = fact.join(dim(220, 240), on="k")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceRight" in plan
    assert plan.count("dynamic_predicate") == 1

    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.height == 2
    assert_matches_in_memory(q, out)


def test_build_side_on_the_left(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    q = dim(200, 420).join(fact, on="k")
    assert "BUILD SIDE: ForceLeft" in q.explain(engine="streaming")

    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "3 / 10 row groups"
    assert_matches_in_memory(q, out)


def test_composite_key_filters_every_column(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    other = pl.LazyFrame({"k": [230, 231, 232], "k2": [6, 6, 6], "d": [1, 2, 3]})
    q = fact.join(other, on=["k", "k2"])
    assert q.explain(engine="streaming").count("dynamic_predicate") == 2

    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert_matches_in_memory(q, out)


def test_static_and_dynamic_predicates_combine(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    q = fact.filter(pl.col("v") >= 225).join(tiny(220, 240, 260), on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.get_column("k").sort().to_list() == [240, 260]
    assert_matches_in_memory(q, out)


def test_filter_reaches_the_scan_through_a_forced_join(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The second dimension's range crosses the first join on its probe side.
    q = fact.join(dim(), on="k").join(tiny(220, 240, key="v"), on="v")
    plan = q.explain(engine="streaming")
    assert plan.count("BUILD SIDE: ForceRight") == 2
    assert plan.count("dynamic_predicate") == 2

    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.height == 2
    assert_matches_in_memory(q, out)


def test_filter_reaches_the_scan_through_renames(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    q = (
        fact.select(pl.col("k").alias("key"), (pl.col("v") * 2).alias("w"))
        .filter(pl.col("w") > 100)
        .with_columns(pl.col("w") + 1)
        .join(tiny(220, 240, key="key"), on="key")
    )
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert_matches_in_memory(q, out)


def test_not_through_a_sampling_join(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The first join's build side has no row bound, so it samples and may read the
    # scan before the second join has built; only the first join's own filter applies.
    unbounded = pl.LazyFrame({"k": list(range(0, 1000, 3))}).select(
        pl.col("k").repeat_by(2).explode()
    )
    q = fact.join(unbounded, on="k").join(dim(220, 240, key="v"), on="v")
    plan = q.explain(engine="streaming")
    assert plan.count("dynamic_predicate") == 0

    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups is None
    assert_matches_in_memory(q, out)


@pytest.mark.parametrize(
    "between",
    [
        lambda lf: lf.with_columns(pl.col("v").cum_sum()),
        lambda lf: lf.filter(pl.col("k") > pl.col("k").mean()),
        # Filtering a window's partition key is sound, but the window is run as a
        # group-by that reads the scan before the join has built.
        lambda lf: lf.with_columns(pl.col("v").sum().over("k")),
    ],
)
def test_barriers_between_join_and_scan(
    fact: pl.LazyFrame,
    between: object,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = between(fact).join(dim(*range(1000)), on="k")  # type: ignore[operator]
    assert "dynamic_predicate" not in q.explain(engine="streaming")
    out = q.collect(engine="streaming")
    assert_matches_in_memory(q, out)


def test_shared_scan_is_not_filtered(fact: pl.LazyFrame) -> None:
    # The scan is cached for two consumers; a filter for one of them must not reach it.
    q = fact.join(dim(220, 240), on="k").join(
        fact.select("k", pl.col("v").alias("w")), on="k"
    )
    plan = q.explain(engine="streaming")
    assert "CACHE" in plan
    assert "dynamic_predicate" not in plan
    out = q.collect(engine="streaming")
    assert_matches_in_memory(q, out)


def test_slice_on_an_intermediate_join_is_a_barrier(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The slice is absorbed into the first join, which then neither publishes nor
    # lets the second join's range through.
    q = fact.join(dim(), on="k").head(5000).join(tiny(40, key="v"), on="v")
    plan = q.explain(engine="streaming")
    assert "dynamic_predicate" not in plan
    assert "Force" not in plan
    out = q.collect(engine="streaming")
    assert_matches_in_memory(q, out)


def test_empty_build_side_reads_nothing(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    q = fact.join(dim(-1), on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "0 / 10 row groups"
    assert out.height == 0
    assert_matches_in_memory(q, out)


def test_all_null_build_keys_read_nothing(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    nulls = pl.LazyFrame({"k": [None, None], "d": [1, 2]}, schema={"k": pl.Int64, "d": pl.Int64})
    q = fact.join(nulls, on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "0 / 10 row groups"
    assert out.height == 0


def test_probe_nulls_are_dropped(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    df = pl.DataFrame({"k": [None if i % 3 == 0 else i for i in range(100)]})
    path = tmp_path / "nulls.parquet"
    df.write_parquet(path, row_group_size=10, statistics="full")
    q = pl.scan_parquet(path).join(tiny(11, 22, 25), on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "2 / 10 row groups"
    assert out.get_column("k").sort().to_list() == [11, 22, 25]


def test_no_statistics_still_correct(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    path = tmp_path / "nostats.parquet"
    pl.DataFrame({"k": range(n), "v": range(n)}).write_parquet(
        path, row_group_size=ROWS_PER_GROUP, statistics=False
    )
    q = pl.scan_parquet(path).join(dim(220, 240), on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "10 / 10 row groups"
    assert out.get_column("k").sort().to_list() == [220, 240]


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda f: f.join(dim(220, 240), on="k", how="left"), "left join"),
        (lambda f: f.join(dim(220, 240), on="k", nulls_equal=True), "nulls equal"),
        (lambda f: f.join(dim(220, 240), on="k", maintain_order="left"), "ordered"),
        (lambda f: f.join(dim(220, 240), on="k", validate="m:1"), "validated"),
        (
            lambda f: f.with_columns(pl.col("k").cast(pl.Float64)).join(
                dim(220, 240).with_columns(pl.col("k").cast(pl.Float64)), on="k"
            ),
            "float key",
        ),
    ],
)
def test_joins_without_runtime_filters(
    fact: pl.LazyFrame, make: object, reason: str
) -> None:
    q = make(fact)  # type: ignore[operator]
    assert "dynamic_predicate" not in q.explain(engine="streaming"), reason
    out = q.collect(engine="streaming")
    assert_matches_in_memory(q, out)


def test_user_forced_build_side_is_left_alone(fact: pl.LazyFrame) -> None:
    q = fact.join(dim(220, 240), on="k", build_side="force_left")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceLeft" in plan
    assert "dynamic_predicate" not in plan


def test_ipc_scan(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    path = tmp_path / "fact.ipc"
    pl.DataFrame({"k": range(n), "v": range(n)}).write_ipc(path)
    q = pl.scan_ipc(path).join(dim(220, 240), on="k")
    assert "dynamic_predicate" in q.explain(engine="streaming")
    out = q.collect(engine="streaming")
    assert out.get_column("k").sort().to_list() == [220, 240]
    assert_matches_in_memory(q, out)


def test_build_side_of_several_morsels(
    fact: pl.LazyFrame,
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The build side arrives in several morsels over several pipelines; the ranges
    # they each saw are merged.
    path = tmp_path / "dim.parquet"
    pl.DataFrame({"k": [300, 301, 302, 303, 304, 660, 661, 662, 663, 664]}).write_parquet(
        path, row_group_size=1
    )
    q = fact.join(pl.scan_parquet(path), on="k")
    assert "dynamic_predicate" in q.explain(engine="streaming")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    assert "reading 4 / 10 row groups" in err
    assert out.get_column("k").sort().to_list() == list(range(300, 305)) + list(
        range(660, 665)
    )

"""A forced hash join publishes its build-key range to the scan under its probe side."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal as D
from math import inf, nan
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from polars._typing import JoinStrategy
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


def tiny(*keys: Any, key: str = "k") -> pl.LazyFrame:
    # Few enough rows to be forced against a filtered fact, whose estimate is lower.
    # Only a filtered side is worth publishing, hence the filter.
    lf = pl.LazyFrame({key: list(keys), "e": list(range(len(keys)))})
    return lf.filter(pl.col("e") >= 0)


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
    other = pl.LazyFrame(
        {"k": [230, 231, 232], "k2": [6, 6, 6], "d": [1, 2, 3]}
    ).filter(pl.col("d") > 0)
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
    q = fact.join(dim(*range(0, 1000, 20)), on="k").join(
        tiny(220, 240, key="v"), on="v"
    )
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
    # The first join's build side has no row bound and is not filtered, so it
    # samples both sides and may read the scan before the second join has built:
    # neither join gets to publish.
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
    between: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> None:
    q = between(fact).join(dim(*range(1000)), on="k")
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
    q = (
        fact.join(dim(*range(0, 1000, 20)), on="k")
        .head(5000)
        .join(tiny(40, key="v"), on="v")
    )
    plan = q.explain(engine="streaming")
    assert "dynamic_predicate" not in plan
    assert "Force" not in plan
    out = q.collect(engine="streaming")
    assert_matches_in_memory(q, out)


def test_empty_build_side_reads_nothing(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # An inner join with an empty build side is done; the scan is never opened.
    q = fact.join(dim(-1), on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups is None
    assert out.height == 0
    assert_matches_in_memory(q, out)


def test_all_null_build_keys_read_nothing(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    nulls = pl.LazyFrame(
        {"k": [None, None], "d": [1, 2]}, schema={"k": pl.Int64, "d": pl.Int64}
    ).filter(pl.col("d") > 0)
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
    fact: pl.LazyFrame, make: Callable[[pl.LazyFrame], pl.LazyFrame], reason: str
) -> None:
    q = make(fact)
    assert "dynamic_predicate" not in q.explain(engine="streaming"), reason
    out = q.collect(engine="streaming")
    assert_matches_in_memory(q, out)


def test_unfiltered_build_side_is_not_published(fact: pl.LazyFrame) -> None:
    # An unfiltered dimension spans its whole key domain; its range prunes nothing.
    q = fact.join(pl.LazyFrame({"k": list(range(0, 1000, 20))}), on="k")
    plan = q.explain(engine="streaming")
    assert "dynamic_predicate" not in plan
    assert "Force" not in plan


def test_user_forced_build_side_is_left_alone(fact: pl.LazyFrame) -> None:
    q = fact.join(dim(220, 240), on="k", build_side="force_left")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceLeft" in plan
    assert "dynamic_predicate" not in plan


def fact_frame() -> pl.DataFrame:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    return pl.DataFrame({"k": range(n), "v": range(n)})


def fact_ipc(tmp_path: Path) -> pl.LazyFrame:
    path = tmp_path / "fact.ipc"
    fact_frame().write_ipc(path)
    return pl.scan_ipc(path)


def fact_csv(tmp_path: Path) -> pl.LazyFrame:
    path = tmp_path / "fact.csv"
    fact_frame().write_csv(path)
    return pl.scan_csv(path)


def assert_no_runtime_filter(q: pl.LazyFrame) -> None:
    plan = q.explain(engine="streaming")
    assert "dynamic_predicate" not in plan
    assert "Force" not in plan
    out = q.collect(engine="streaming")
    assert out.get_column("k").sort().to_list() == [220, 240]
    assert_matches_in_memory(q, out)


@pytest.mark.parametrize(
    "probe",
    [fact_ipc, fact_csv, lambda _tmp_path: fact_frame().lazy()],
    ids=["ipc", "csv", "in-memory"],
)
def test_only_parquet_probes_get_a_filter(
    tmp_path: Path, probe: Callable[[Path], pl.LazyFrame]
) -> None:
    assert_no_runtime_filter(probe(tmp_path).join(dim(220, 240), on="k"))
    assert_no_runtime_filter(dim(220, 240).join(probe(tmp_path), on="k"))


def test_parquet_build_side_does_not_make_a_probe_eligible(
    tmp_path: Path, fact: pl.LazyFrame
) -> None:
    # The probe side's format decides; a parquet build side is read like any other.
    path = tmp_path / "dim.parquet"
    pl.DataFrame({"k": list(range(0, 1000, 20)), "d": list(range(50))}).write_parquet(
        path
    )
    build = pl.scan_parquet(path).filter(pl.col("k").is_in([220, 240]))
    assert_no_runtime_filter(fact_ipc(tmp_path).join(build, on="k"))
    assert_no_runtime_filter(build.join(fact_ipc(tmp_path), on="k"))

    # A build side read from IPC still publishes to a parquet probe.
    ipc = tmp_path / "dim.ipc"
    pl.DataFrame({"k": list(range(0, 1000, 20)), "d": list(range(50))}).write_ipc(ipc)
    q = fact.join(pl.scan_ipc(ipc).filter(pl.col("k").is_in([220, 240])), on="k")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceRight" in plan
    assert plan.count("dynamic_predicate") == 1


def test_only_the_parquet_join_of_a_mixed_plan_gets_a_filter(
    tmp_path: Path,
    fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = pl.concat(
        [
            fact.join(dim(220, 240), on="k").select("k", "v", "d"),
            fact_ipc(tmp_path).join(dim(220, 240), on="k"),
        ]
    )
    plan = q.explain(engine="streaming")
    assert plan.count("dynamic_predicate") == 1
    assert plan.count("Force") == 1
    assert plan.index("Force") < plan.index("Ipc SCAN")

    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.get_column("k").sort().to_list() == [220, 220, 240, 240]
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
    pl.DataFrame(
        {"k": [300, 301, 302, 303, 304, 660, 661, 662, 663, 664]}
    ).write_parquet(path, row_group_size=1)
    q = fact.join(pl.scan_parquet(path).filter(pl.col("k") > 0), on="k")
    assert "dynamic_predicate" in q.explain(engine="streaming")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    assert "reading 4 / 10 row groups" in err
    assert out.get_column("k").sort().to_list() == list(range(300, 305)) + list(
        range(660, 665)
    )


def test_row_filtering_stays_on_without_live_columns(tmp_path: Path) -> None:
    # A predicate reading no column of the file still filters rows; whether a scan
    # filters rows is not derived from the columns it reads.
    path = tmp_path / "fact.parquet"
    pl.DataFrame({"k": range(10)}).write_parquet(path)

    out = pl.scan_parquet(path, use_statistics=False).filter(pl.lit(False))
    assert out.collect(engine="streaming").height == 0
    assert out.collect(engine="in-memory").height == 0

    # An inserted missing column is bound as a constant, leaving no live columns.
    with_missing = pl.scan_parquet(
        path, schema={"k": pl.Int64, "m": pl.Int64}, missing_columns="insert"
    ).filter(pl.col("m") == 1)
    assert with_missing.collect(engine="streaming").height == 0
    assert with_missing.collect(engine="in-memory").height == 0


def typed_join(
    tmp_path: Path,
    dtype: pl.DataType,
    values: list[Any],
    keys: list[Any],
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> tuple[str | None, bool]:
    """Join a typed fact with `keys`; report the row groups read and filter presence.

    The fact holds five values per row group, in the order given, which must be the
    dtype's own. The result is checked against the values in `keys` and against the
    same query with statistics disabled.
    """
    s = pl.Series("k", values, dtype=dtype)
    assert s.equals(s.sort())
    path = tmp_path / "typed.parquet"
    pl.DataFrame({"k": s, "v": range(len(values))}).write_parquet(
        path, row_group_size=5, statistics="full"
    )
    # The build side is bounded to two rows so it is forced against twenty fact
    # rows. A single key is padded with a null key, which matches nothing and
    # stays out of the range; a build side of one row is not estimated smaller
    # after its filter.
    assert 1 <= len(keys) <= 2
    keys_s = pl.Series("k", keys, dtype=dtype)
    assert keys_s.n_unique() == len(keys)
    build_keys = keys_s.extend(pl.Series([None] * (2 - len(keys)), dtype=dtype))
    build = pl.LazyFrame({"k": build_keys, "e": [0, 1]}).filter(pl.col("e") >= 0)

    q = pl.scan_parquet(path).join(build, on="k")
    filtered = "dynamic_predicate" in q.explain(engine="streaming")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)

    expected = pl.DataFrame({"k": s}).filter(pl.col("k").is_in(keys_s))
    assert_series_equal(out.get_column("k").sort(), expected.get_column("k"))
    no_stats = pl.scan_parquet(path, use_statistics=False).join(build, on="k")
    assert_frame_equal(
        out.sort(out.columns), no_stats.collect(engine="streaming").sort(out.columns)
    )
    return groups, filtered


I64_MIN = -(2**63)
I64_MAX = 2**63 - 1
U64_MAX = 2**64 - 1

INT64 = (
    [I64_MIN, -(2**62), -1000, -1, 0]
    + [1, 2, 3, 4, 5]
    + [100, 200, 300, 400, 500]
    + [2**62, I64_MAX - 2, I64_MAX - 1, I64_MAX, I64_MAX]
)
# Crosses the signed maximum inside the second row group.
UINT64 = (
    [0, 1, 2, 3, 4]
    + [5, I64_MAX - 1, I64_MAX, I64_MAX + 1, I64_MAX + 2]
    + [I64_MAX + 10, I64_MAX + 11, I64_MAX + 12, I64_MAX + 13, I64_MAX + 14]
    + [U64_MAX - 4, U64_MAX - 3, U64_MAX - 2, U64_MAX - 1, U64_MAX]
)
INT8 = (
    list(range(-128, -123))
    + list(range(-2, 3))
    + list(range(10, 15))
    + list(range(123, 128))
)
# Crosses the signed maximum inside the third row group.
UINT8 = (
    list(range(5))
    + list(range(100, 105))
    + list(range(126, 131))
    + list(range(251, 256))
)
# Strings compare by their UTF-8 bytes, so multi-byte characters sort last.
STRINGS = (
    ["a", "b", "c", "d", "e"]
    + ["f", "g", "h", "i", "j"]
    + ["k", "é", "ü", "ā", "ж"]
    + ["中", "日", "本", "🐍", "🦀"]
)
BINARY = (
    [b"\x00", b"\x01", b"a", b"z", b"\x7f"]
    + [b"\x80", b"\x81", b"\x82", b"\x83", b"\x84"]
    + [b"\xf0", b"\xf1", b"\xf2", b"\xf3", b"\xf4"]
    + [b"\xfb", b"\xfc", b"\xfd", b"\xfe", b"\xff"]
)
DECIMALS = (
    [D("-99999999.99"), D("-1000.50"), D("-1.01"), D("-1.00"), D("-0.01")]
    + [D("0.00"), D("0.01"), D("1.00"), D("1.01"), D("2.00")]
    + [D("10.00"), D("20.00"), D("30.00"), D("40.00"), D("50.00")]
    + [D("100.00"), D("200.00"), D("300.00"), D("400.00"), D("99999999.99")]
)
DATES = (
    [
        date(1, 1, 1),
        date(1900, 1, 1),
        date(1969, 12, 30),
        date(1969, 12, 31),
        date(1970, 1, 1),
    ]
    + [date(1970, 1, 2) + timedelta(days=i) for i in range(5)]
    + [date(2000, 1, 1) + timedelta(days=i) for i in range(5)]
    + [date(2100, 1, 1) + timedelta(days=i) for i in range(4)]
    + [date(9999, 12, 31)]
)
DATETIMES = (
    [datetime(1900, 1, 1) + timedelta(seconds=i) for i in range(5)]
    + [datetime(1969, 12, 31, 23, 59, 59) + timedelta(microseconds=i) for i in range(5)]
    + [datetime(1970, 1, 1) + timedelta(microseconds=i) for i in range(5)]
    + [datetime(2200, 1, 1) + timedelta(seconds=i) for i in range(5)]
)
DURATIONS = (
    [timedelta(days=-10000) + timedelta(milliseconds=i) for i in range(5)]
    + [timedelta(milliseconds=i - 2) for i in range(5)]
    + [timedelta(seconds=i + 1) for i in range(5)]
    + [timedelta(days=10000) + timedelta(milliseconds=i) for i in range(5)]
)
TIMES = (
    [time(0, 0, 0, i) for i in range(5)]
    + [time(6, 0, 0, i) for i in range(5)]
    + [time(12, 0, 0, i) for i in range(5)]
    + [time(23, 59, 59, 999995 + i) for i in range(5)]
)


@pytest.mark.parametrize(
    ("dtype", "values", "keys", "groups"),
    [
        (pl.Int64, INT64, [I64_MIN, -1000], "1 / 4"),
        (pl.Int64, INT64, [I64_MAX - 1, I64_MAX], "1 / 4"),
        (pl.Int64, INT64, [-1, 1], "2 / 4"),
        (pl.UInt64, UINT64, [I64_MAX + 1, I64_MAX + 2], "1 / 4"),
        (pl.UInt64, UINT64, [U64_MAX - 1, U64_MAX], "1 / 4"),
        (pl.UInt64, UINT64, [4, 5], "2 / 4"),
        (pl.Int8, INT8, [-128, -127], "1 / 4"),
        (pl.Int8, INT8, [126, 127], "1 / 4"),
        (pl.Int8, INT8, [2, 10], "2 / 4"),
        (pl.UInt8, UINT8, [128, 130], "1 / 4"),
        (pl.UInt8, UINT8, [254, 255], "1 / 4"),
        (pl.Boolean, [False] * 10 + [True] * 10, [True], "2 / 4"),
        (pl.Boolean, [False] * 10 + [True] * 10, [False], "2 / 4"),
        (pl.String, STRINGS, ["ü", "ā"], "1 / 4"),
        (pl.String, STRINGS, ["🐍", "🦀"], "1 / 4"),
        (pl.String, STRINGS, ["k", "中"], "2 / 4"),
        (pl.Binary, BINARY, [b"\x7f", b"\x80"], "2 / 4"),
        (pl.Binary, BINARY, [b"\x80", b"\x84"], "1 / 4"),
        (pl.Binary, BINARY, [b"\xfe", b"\xff"], "1 / 4"),
        (pl.Decimal(10, 2), DECIMALS, [D("-1000.50"), D("-1.01")], "1 / 4"),
        (pl.Decimal(10, 2), DECIMALS, [D("30.00"), D("100.00")], "2 / 4"),
        (pl.Date, DATES, [date(1, 1, 1), date(1900, 1, 1)], "1 / 4"),
        (pl.Date, DATES, [date(2100, 1, 2), date(9999, 12, 31)], "1 / 4"),
        (
            pl.Datetime("us"),
            DATETIMES,
            [
                datetime(1969, 12, 31, 23, 59, 59, 3),
                datetime(1969, 12, 31, 23, 59, 59, 4),
            ],
            "1 / 4",
        ),
        (
            pl.Datetime("us"),
            DATETIMES,
            [datetime(1900, 1, 1), datetime(1900, 1, 1, 0, 0, 1)],
            "1 / 4",
        ),
        (
            pl.Duration("ms"),
            DURATIONS,
            [timedelta(milliseconds=-1), timedelta(0)],
            "1 / 4",
        ),
        (
            pl.Duration("ms"),
            DURATIONS,
            [timedelta(days=10000), timedelta(days=10000, milliseconds=1)],
            "1 / 4",
        ),
        (pl.Time, TIMES, [time(23, 59, 59, 999998), time(23, 59, 59, 999999)], "1 / 4"),
        (pl.Time, TIMES, [time(0, 0, 0, 4), time(6, 0, 0, 0)], "2 / 4"),
    ],
)
def test_ranges_of_supported_key_types(
    tmp_path: Path,
    dtype: pl.DataType,
    values: list[Any],
    keys: list[Any],
    groups: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    read, filtered = typed_join(tmp_path, dtype, values, keys, plmonkeypatch, capfd)
    assert filtered
    assert read == f"{groups} row groups"


@pytest.mark.parametrize(
    ("dtype", "values", "keys"),
    [
        (
            pl.Int128,
            [-(2**127), -(2**64), -1, 0, 1] + list(range(10, 25)),
            [-(2**64), 12],
        ),
        (
            pl.UInt128,
            list(range(15)) + [2**63, 2**64, 2**100, 2**127, 2**128 - 1],
            [2**100, 2**128 - 1],
        ),
        (
            pl.Float64,
            [-inf, -1.5, -0.0, 0.0, 1.5] + [float(i) for i in range(2, 17)],
            [-0.0, inf],
        ),
        (pl.Float32, [float(i) for i in range(19)] + [nan], [nan, 3.0]),
        (pl.Categorical, [f"c{i:02}" for i in range(20)], ["c03", "c17"]),
        (
            pl.Enum([f"e{i:02}" for i in range(20)]),
            [f"e{i:02}" for i in range(20)],
            ["e03", "e17"],
        ),
    ],
)
def test_unsupported_key_types_get_no_range(
    tmp_path: Path,
    dtype: pl.DataType,
    values: list[Any],
    keys: list[Any],
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    read, filtered = typed_join(tmp_path, dtype, values, keys, plmonkeypatch, capfd)
    assert not filtered
    assert read in (None, "4 / 4 row groups")


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128])
def test_static_predicate_on_128_bit_column(tmp_path: Path, dtype: pl.DataType) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"k": pl.Series([1, 2, 3, 4], dtype=dtype)}).write_parquet(
        path, row_group_size=2, statistics="full"
    )
    q = pl.scan_parquet(path).filter(pl.col("k") > 2)
    assert q.collect(engine="streaming").get_column("k").to_list() == [3, 4]


def test_int96_timestamps_have_no_bounds(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # Statistics of INT96 timestamps are not ordered as polars orders datetimes;
    # the range is published but skips nothing. An empty build side ends the join
    # before the scan opens.
    import pyarrow.parquet as pq

    stamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    path = tmp_path / "int96.parquet"
    pq.write_table(
        pl.DataFrame({"k": stamps, "v": range(20)}).to_arrow(),
        path,
        use_deprecated_int96_timestamps=True,
        row_group_size=5,
    )
    assert pq.read_metadata(path).row_group(0).column(0).physical_type == "INT96"
    fact = pl.scan_parquet(path)

    build = pl.LazyFrame({"k": stamps[12:14], "e": [0, 1]}).filter(pl.col("e") >= 0)
    q = fact.join(build, on="k")
    assert "dynamic_predicate" in q.explain(engine="streaming")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "4 / 4 row groups"
    assert out.get_column("v").sort().to_list() == [12, 13]

    empty = build.filter(pl.col("k").is_in([datetime(1999, 1, 1)]))
    out, groups = row_groups_read(fact.join(empty, on="k"), plmonkeypatch, capfd)
    assert groups is None
    assert out.height == 0


def test_pruned_metadata_keeps_pruning(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # Metadata pruned to the projection keeps the column orders of its leaves,
    # including those under a projected struct.
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    path = tmp_path / "nested.parquet"
    pl.DataFrame(
        {"s": [{"a": i, "b": str(i)} for i in range(n)], "k": range(n), "v": range(n)}
    ).write_parquet(path, row_group_size=ROWS_PER_GROUP, statistics="full")
    plmonkeypatch.setenv("POLARS_PRUNE_PARQUET_METADATA", "1")

    q = pl.scan_parquet(path).select("s", "k").join(dim(220, 240), on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.sort("k").get_column("s").to_list() == [
        {"a": 220, "b": "220"},
        {"a": 240, "b": "240"},
    ]

    q = pl.scan_parquet(path).select("s").filter(pl.col("s").struct.field("a") == 555)
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.height == 1


def unbounded_dim(*keys: int, key: str = "k") -> pl.LazyFrame:
    # A join of two frames has no row bound; its estimate is small once filtered, so
    # it is preferred as build side rather than forced.
    a = pl.LazyFrame({key: list(range(0, 1000, 20)), "d": list(range(50))})
    b = pl.LazyFrame({key: list(range(0, 1000, 20)), "e": list(range(50))})
    return a.join(b, on=key).filter(pl.col(key).is_in(list(keys)))


def test_preferred_build_side_publishes(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    for q in (
        fact.join(unbounded_dim(220, 240), on="k"),
        unbounded_dim(220, 240).join(fact, on="k"),
    ):
        plan = q.explain(engine="streaming")
        assert "BUILD SIDE: Prefer" in plan
        assert "Force" not in plan
        assert plan.count("dynamic_predicate") == 1

        out, groups = row_groups_read(q, plmonkeypatch, capfd)
        assert groups == "1 / 10 row groups"
        assert out.get_column("k").sort().to_list() == [220, 240]
        assert_matches_in_memory(q, out)


def test_empty_preferred_build_side_reads_nothing(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    q = fact.join(unbounded_dim(-1), on="k")
    assert "BUILD SIDE: Prefer" in q.explain(engine="streaming")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups is None
    assert out.height == 0


@pytest.mark.parametrize(
    ("limit", "groups"),
    [
        # The preferred side is sampled first. When it ends under the limit its
        # range is published before the other side is sampled and a build side is
        # chosen. When it reaches the limit both sides are sampled and the scan
        # has already opened.
        ("3", "1 / 10 row groups"),
        ("2", "10 / 10 row groups"),
        ("1", "10 / 10 row groups"),
        # Without sampling the preference is followed outright.
        ("0", "1 / 10 row groups"),
    ],
)
def test_preferred_side_against_the_sample_limit(
    fact: pl.LazyFrame,
    limit: str,
    groups: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", limit)
    q = fact.join(unbounded_dim(220, 240), on="k")
    out, read = row_groups_read(q, plmonkeypatch, capfd)
    assert read == groups
    assert out.get_column("k").sort().to_list() == [220, 240]
    assert_matches_in_memory(q, out)


@pytest.fixture
def clustered_fact(tmp_path: Path) -> pl.LazyFrame:
    # `k2` names the row group, so a range of it maps to row groups too.
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    df = pl.DataFrame(
        {"k": range(n), "k2": [i // ROWS_PER_GROUP for i in range(n)], "v": range(n)}
    )
    path = tmp_path / "clustered.parquet"
    df.write_parquet(path, row_group_size=ROWS_PER_GROUP, statistics="full")
    return pl.scan_parquet(path)


def forced_over_preferred(fact: pl.LazyFrame) -> pl.LazyFrame:
    # The inner join prefers its unbounded side; the outer one forces `k2` and
    # carries its range across the inner join to the scan.
    inner = fact.join(unbounded_dim(*range(0, 1000, 40)), on="k")
    return inner.join(tiny(3, 4, key="k2"), on="k2")


def test_forced_range_crosses_a_preferred_join(
    clustered_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = forced_over_preferred(clustered_fact)
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceRight" in plan
    assert "BUILD SIDE: Prefer" in plan
    assert plan.count("dynamic_predicate") == 2

    # The inner join builds its preferred side and keeps the scan closed until
    # the outer join has published.
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "2 / 10 row groups"
    assert out.get_column("k").sort().to_list() == list(range(320, 500, 40))
    assert_matches_in_memory(q, out)


def test_saturated_preferred_join_keeps_the_result_exact(
    clustered_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The inner join's preferred side outgrows the sample, so it samples both sides
    # and opens the scan before the outer join has built; nothing may be skipped
    # that the outer range would have kept.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "1")
    q = forced_over_preferred(clustered_fact)
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups is not None
    assert out.get_column("k").sort().to_list() == list(range(320, 500, 40))
    assert_matches_in_memory(q, out)


def test_range_is_judged_against_the_scan_it_prunes(
    clustered_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The inner join's output is estimated no larger than the fifty-row dimension,
    # but the dimension's range prunes the thousand-row scan behind it.
    inner = clustered_fact.join(unbounded_dim(420, 540, 640), on="k")
    k2dim = pl.LazyFrame({"k2": list(range(50)), "d2": list(range(50))})
    q = inner.join(k2dim.filter(pl.col("k2").is_in([5])), on="k2")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceRight" in plan
    assert plan.count("dynamic_predicate") == 2

    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.get_column("k").to_list() == [540]
    assert_matches_in_memory(q, out)


def test_published_range_holds_when_the_other_side_is_built(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The preferred side ends first and publishes its range; the scan behind the
    # other side then reads two row groups, of which one row survives the filter,
    # so the sample builds that side instead. The range still holds: no key of
    # the other side outside it can match.
    q = fact.filter(pl.col("v") % 500 == 220).join(
        unbounded_dim(*range(200, 400, 20)), on="k"
    )
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: Prefer" in plan
    dim_left = "BUILD SIDE: PreferLeft" in plan
    other = "right" if dim_left else "left"
    lengths = "10 vs. 1" if dim_left else "1 vs. 10"

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    # The dimension's own join logs its choice too, before the publication.
    after = err.split("publishing its ranges", 1)[1]
    assert "Predicate pushdown: reading 2 / 10 row groups" in after
    assert f"sample lengths are: {lengths}" in after
    assert f"build side chosen: {other}" in after
    assert out.get_column("k").to_list() == [220]
    assert_matches_in_memory(q, out)


@pytest.mark.parametrize("dim_left", [False, True])
def test_empty_preferred_side_never_reads_the_other(
    tmp_path: Path,
    dim_left: bool,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Without statistics the range cannot skip anything, so the join itself must
    # finish once the empty side is read.
    path = tmp_path / "fact.parquet"
    pl.DataFrame({"k": range(1000), "v": range(1000)}).write_parquet(
        path, row_group_size=ROWS_PER_GROUP, statistics=False
    )
    fact = pl.scan_parquet(path)
    dim = unbounded_dim(-1)
    q = dim.join(fact, on="k") if dim_left else fact.join(dim, on="k")
    assert "BUILD SIDE: Prefer" in q.explain(engine="streaming")

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    err = capfd.readouterr().err
    assert "preferred build side done with 0 rows" in err
    assert "[ParquetFileReader]" not in err
    assert out.height == 0


def test_a_key_that_may_change_gets_no_range(fact: pl.LazyFrame) -> None:
    # The range would come from one evaluation of the key and the build from
    # another.
    dim = pl.LazyFrame({"ks": [[220, 720], [220, 720]], "d": [1, 2]}).filter(
        pl.col("d") > 0
    )
    key = pl.col("ks").list.sample(1, seed=1).list.first()
    q = fact.join(dim, left_on="k", right_on=key)
    assert "dynamic_predicate" not in q.explain(engine="streaming")
    out = q.collect(engine="streaming")
    assert out.height == 2
    assert set(out.get_column("k").to_list()) <= {220, 720}


def reader_log(
    q: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> tuple[pl.DataFrame, str]:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    return out, capfd.readouterr().err


def test_scan_without_statistics_is_not_eligible(tmp_path: Path) -> None:
    # A range no scan can use gives the join no build side either.
    path = tmp_path / "plain.parquet"
    pl.DataFrame({"k": range(1000), "v": range(1000)}).write_parquet(
        path, row_group_size=ROWS_PER_GROUP
    )
    q = pl.scan_parquet(path, use_statistics=False).join(dim(220, 240), on="k")
    assert "dynamic_predicate" not in q.explain(engine="streaming")
    assert q.collect(engine="streaming").get_column("k").sort().to_list() == [220, 240]


def test_broad_range_filters_rows_by_bloom(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # Shuffled keys: every row group spans the range, so nothing is skipped and
    # the bloom filter over the build keys filters the rows instead.
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    keys = pl.Series("k", range(n)).shuffle(seed=1)
    path = tmp_path / "shuffled.parquet"
    pl.DataFrame({"k": keys, "v": range(n)}).write_parquet(
        path, row_group_size=ROWS_PER_GROUP, statistics="full"
    )
    q = pl.scan_parquet(path).join(dim(220, 240), on="k")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "reading 10 / 10 row groups" in err
    assert "bloom: Some" in err
    assert "Pre-filtered decode enabled (1 live [1 column predicates" in err
    assert out.get_column("k").sort().to_list() == [220, 240]


def test_static_predicate_on_the_key_narrows_the_candidates(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The static predicate keeps three groups; the range keeps one of them.
    q = fact.filter(pl.col("k") < 250).join(tiny(220, 240, 260), on="k")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "reading 1 / 10 row groups" in err
    assert out.get_column("k").sort().to_list() == [220, 240]
    assert_matches_in_memory(q, out)


@pytest.mark.parametrize(
    ("keys", "groups"),
    [((199, 200), "2 / 10"), ((200, 299), "1 / 10"), ((299, 300), "2 / 10")],
)
def test_range_bounds_are_inclusive(
    fact: pl.LazyFrame,
    keys: tuple[int, int],
    groups: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = fact.join(tiny(*keys), on="k")
    out, read = row_groups_read(q, plmonkeypatch, capfd)
    assert read == f"{groups} row groups"
    assert out.get_column("k").sort().to_list() == list(keys)


def test_hive_key_settles_whole_files(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    for part in (0, 5, 10):
        (tmp_path / f"p={part}").mkdir()
        pl.DataFrame(
            {"k": range(part * 100, part * 100 + 100), "v": range(100)}
        ).write_parquet(
            tmp_path / f"p={part}" / "0.parquet", row_group_size=50, statistics="full"
        )
    fact = pl.scan_parquet(tmp_path, hive_partitioning=True)
    q = fact.join(tiny(5, 7, key="p"), on="p")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert err.count("reading 0 / 2 row groups") == 2
    assert err.count("reading 2 / 2 row groups") == 1
    assert out.get_column("k").sort().to_list() == list(range(500, 600))
    assert_matches_in_memory(q, out)


def test_missing_key_column_matches_nothing(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # A file without the key column reads it as null, which no range holds.
    pl.DataFrame({"k": range(100), "v": range(100)}).write_parquet(
        tmp_path / "a.parquet", row_group_size=50, statistics="full"
    )
    pl.DataFrame({"v": range(100, 200)}).write_parquet(
        tmp_path / "b.parquet", row_group_size=50, statistics="full"
    )
    fact = pl.scan_parquet(
        [tmp_path / "a.parquet", tmp_path / "b.parquet"], missing_columns="insert"
    )
    q = fact.join(tiny(60, 70), on="k")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "reading 1 / 2 row groups" in err
    assert "reading 0 / 2 row groups" in err
    assert out.get_column("k").sort().to_list() == [60, 70]


def test_files_of_different_layouts(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    pl.DataFrame({"k": range(n), "v": range(n)}).write_parquet(
        tmp_path / "sorted.parquet", row_group_size=ROWS_PER_GROUP, statistics="full"
    )
    pl.DataFrame(
        {"k": pl.Series(range(n)).shuffle(seed=2), "v": range(n)}
    ).write_parquet(
        tmp_path / "shuffled.parquet", row_group_size=ROWS_PER_GROUP, statistics="full"
    )
    fact = pl.scan_parquet([tmp_path / "sorted.parquet", tmp_path / "shuffled.parquet"])
    q = fact.join(dim(220, 240), on="k")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "reading 1 / 10 row groups" in err
    assert "reading 10 / 10 row groups" in err
    assert out.get_column("k").sort().to_list() == [220, 220, 240, 240]


def test_repeated_and_concurrent_collects(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    q = fact.join(dim(220, 240), on="k")
    for _ in range(3):
        out, read = row_groups_read(q, plmonkeypatch, capfd)
        assert read == "1 / 10 row groups"
        assert out.get_column("k").sort().to_list() == [220, 240]

    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda _: q.collect(engine="streaming"), range(8)))
    for out in outs:
        assert out.get_column("k").sort().to_list() == [220, 240]


def test_saturated_preferred_side_disables_its_range(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The preferred side outgrows the sample and the other side is built; the
    # scan is told there will be no range and keeps every group.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "10")
    q = fact.join(unbounded_dim(*range(0, 1000, 20)), on="k")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "preferred build side reached the sample limit" in err
    assert "reading 10 / 10 row groups" in err
    assert_matches_in_memory(q, out)


@pytest.mark.parametrize("unit", ["ms", "us"])
def test_time_keys_in_file_units(
    tmp_path: Path,
    unit: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The file stores time of day in its own unit; polars compares in nanoseconds.
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = N_ROW_GROUPS * ROWS_PER_GROUP
    values = [time(i // 3600, i // 60 % 60, i % 60) for i in range(n)]
    table = pa.table(
        {
            "k": pa.array(
                values, type=pa.time32(unit) if unit == "ms" else pa.time64(unit)
            ),
            "v": list(range(n)),
        }
    )
    path = tmp_path / f"time_{unit}.parquet"
    pq.write_table(table, path, row_group_size=ROWS_PER_GROUP)
    q = pl.scan_parquet(path).join(tiny(values[220], values[240]), on="k")
    assert "dynamic_predicate" in q.explain(engine="streaming")
    out, read = row_groups_read(q, plmonkeypatch, capfd)
    assert read == "1 / 10 row groups"
    assert out.get_column("v").sort().to_list() == [220, 240]


@pytest.mark.parametrize(
    ("arrow_type", "make"),
    [
        ("duration[s]", lambda i: timedelta(seconds=i)),
        ("timestamp[s]", lambda i: datetime(2020, 1, 1) + timedelta(seconds=i)),
        ("duration[ms]", lambda i: timedelta(milliseconds=i)),
        ("timestamp[ms]", lambda i: datetime(2020, 1, 1) + timedelta(milliseconds=i)),
        ("timestamp[s]", lambda i: datetime(1969, 12, 31) + timedelta(seconds=i)),
        ("duration[s]", lambda i: timedelta(seconds=i - 500)),
        ("date64", lambda i: datetime(1969, 1, 1) + timedelta(days=i)),
    ],
)
def test_second_resolution_keys(
    tmp_path: Path,
    arrow_type: str,
    make: Callable[[int], Any],
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Polars reads second resolution as milliseconds; the bounds must follow.
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = N_ROW_GROUPS * ROWS_PER_GROUP
    if arrow_type == "date64":
        pa_type = pa.date64()
    else:
        kind, unit = arrow_type[:-1].split("[")
        pa_type = pa.duration(unit) if kind == "duration" else pa.timestamp(unit)
    values = [make(i) for i in range(n)]
    table = pa.table({"k": pa.array(values, type=pa_type), "v": list(range(n))})
    path = tmp_path / "temporal.parquet"
    pq.write_table(table, path, row_group_size=ROWS_PER_GROUP)
    fact = pl.scan_parquet(path)
    build = pl.LazyFrame({"k": [values[220], values[240]], "e": [0, 1]}).filter(
        pl.col("e") >= 0
    )
    q = fact.join(build.cast({"k": fact.collect_schema()["k"]}), on="k")
    assert "dynamic_predicate" in q.explain(engine="streaming")
    out, read = row_groups_read(q, plmonkeypatch, capfd)
    assert read == "1 / 10 row groups"
    assert out.get_column("v").sort().to_list() == [220, 240]


def test_decimal_and_string_keys_prune(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    path = tmp_path / "typed.parquet"
    pl.DataFrame(
        {
            "d": pl.Series([D(i) / 100 for i in range(n)], dtype=pl.Decimal(10, 2)),
            "s": [f"{i:04}" for i in range(n)],
            "v": range(n),
        }
    ).write_parquet(path, row_group_size=ROWS_PER_GROUP, statistics="full")
    fact = pl.scan_parquet(path)

    build = pl.LazyFrame(
        {"d": pl.Series([D("2.20"), D("2.40")], dtype=pl.Decimal(10, 2)), "e": [0, 1]}
    ).filter(pl.col("e") >= 0)
    out, read = row_groups_read(fact.join(build, on="d"), plmonkeypatch, capfd)
    assert read == "1 / 10 row groups"
    assert out.get_column("v").sort().to_list() == [220, 240]

    build = pl.LazyFrame({"s": ["0220", "0240"], "e": [0, 1]}).filter(pl.col("e") >= 0)
    out, read = row_groups_read(fact.join(build, on="s"), plmonkeypatch, capfd)
    assert read == "1 / 10 row groups"
    assert out.get_column("v").sort().to_list() == [220, 240]


# A bloom filter over the build keys filters the rows of a scan the range cannot
# prune: the keys of every row group span the range.


@pytest.fixture
def shuffled_fact(tmp_path: Path) -> pl.LazyFrame:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    keys = pl.Series("k", range(n)).shuffle(seed=1)
    df = pl.DataFrame({"k": keys, "k2": keys % 7, "v": keys})
    path = tmp_path / "shuffled.parquet"
    df.write_parquet(path, row_group_size=ROWS_PER_GROUP, statistics="full")
    return pl.scan_parquet(path)


def bloom_run(
    q: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    keys: list[Any],
    column: str = "k",
) -> str:
    """Collect on both engines, check they agree on `keys`, and return the log."""
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert out.get_column(column).sort().to_list() == keys
    assert_matches_in_memory(q, out)
    return err


def test_bloom_with_a_static_predicate_on_the_key(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = shuffled_fact.filter(pl.col("k") > 500).join(tiny(220, 240, 620, 640), on="k")
    err = bloom_run(q, plmonkeypatch, capfd, [620, 640])
    assert "bloom of" in err
    assert "reading 10 / 10 row groups" in err


def test_bloom_set_after_the_scan_opened(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The preferred side reaches the sample limit, so the scan opens before the
    # filter is published and sees it set part way, if at all.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "1")
    build = unbounded_dim(220, 240, 620, 640)
    for q in (
        shuffled_fact.join(build, on="k"),
        shuffled_fact.filter(pl.col("k") > 500).join(build, on="k"),
        shuffled_fact.join(build, on="k").select("v", "d"),
    ):
        assert "BUILD SIDE: Prefer" in q.explain(engine="streaming")
        out, err = reader_log(q, plmonkeypatch, capfd)
        assert "reading 10 / 10 row groups" in err
        assert_matches_in_memory(q, out)


def test_weak_bloom_is_bypassed(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # String keys carry no distinct count, so the plan cannot tell the filter is
    # weak; the reader stops evaluating it once it keeps most rows.
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    path = tmp_path / "strings.parquet"
    pl.DataFrame({"s": [f"s{i % 40}" for i in range(n)], "v": range(n)}).write_parquet(
        path, row_group_size=ROWS_PER_GROUP, statistics="full"
    )
    build = pl.LazyFrame({"s": [f"s{i}" for i in range(50)], "e": range(50)}).filter(
        pl.col("e") >= 0
    )
    q = pl.scan_parquet(path).join(build, on="s")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "bloom of" in err
    assert "Dynamic predicate bypassed" in err
    assert out.height == n
    assert_matches_in_memory(q, out)


def test_bloom_gate_keeps_the_range_only(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The build holds three of the seven values `k2` takes: not worth probing
    # per row.
    q = shuffled_fact.join(tiny(0, 1, 2, key="k2"), on="k2")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "bloom: None" in err
    assert "bloom of" not in err
    assert out.get_column("k2").unique().sort().to_list() == [0, 1, 2]
    assert_matches_in_memory(q, out)


def test_two_blooms_on_one_scan_key(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = shuffled_fact.join(dim(220, 240, 260), on="k").join(tiny(240, 260, 280), on="k")
    assert q.explain(engine="streaming").count("dynamic_predicate") == 2
    err = bloom_run(q, plmonkeypatch, capfd, [240, 260])
    assert err.count("bloom of") == 2
    assert "Pre-filtered decode enabled (1 live [1 column predicates" in err


def test_bloom_on_composite_keys(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    other = pl.LazyFrame(
        {"k": [230, 231, 232], "k2": [230 % 7, 231 % 7, 6], "d": [1, 2, 3]}
    ).filter(pl.col("d") > 0)
    q = shuffled_fact.join(other, on=["k", "k2"])
    err = bloom_run(q, plmonkeypatch, capfd, [230, 231])
    assert err.count("bloom of") >= 1


def test_bloom_never_matches_null_keys(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    keys = pl.Series("k", range(n)).shuffle(seed=3)
    keys_with_nulls = keys.to_frame().select(
        pl.when(keys % 5 == 0).then(None).otherwise(keys).alias("k")
    )
    path = tmp_path / "nulls.parquet"
    keys_with_nulls.with_columns(v=pl.int_range(n)).write_parquet(
        path, row_group_size=ROWS_PER_GROUP, statistics="full"
    )
    build = pl.LazyFrame({"k": [None, 220, 221, 225], "e": [0, 1, 2, 3]}).filter(
        pl.col("e") >= 0
    )
    q = pl.scan_parquet(path).join(build, on="k")
    err = bloom_run(q, plmonkeypatch, capfd, [221])
    assert "bloom of" in err


@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt16, pl.String, pl.Date])
def test_bloom_key_dtypes(
    tmp_path: Path,
    dtype: pl.DataType,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    keys = pl.Series("k", range(n)).shuffle(seed=4)
    path = tmp_path / "typed.parquet"
    pl.DataFrame({"k": keys.cast(dtype), "v": keys}).write_parquet(
        path, row_group_size=ROWS_PER_GROUP, statistics="full"
    )
    build = pl.LazyFrame(
        {"k": pl.Series([220, 240], dtype=pl.Int64).cast(dtype), "e": [0, 1]}
    ).filter(pl.col("e") >= 0)
    q = pl.scan_parquet(path).join(build, on="k")
    err = bloom_run(q, plmonkeypatch, capfd, [220, 240], column="v")
    assert "bloom of" in err


# A semi join publishes from either side, an anti join from its left side only.


def test_semi_join_publishes_from_either_side(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = shuffled_fact.join(tiny(220, 240, 1000), on="k", how="semi")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceRight" in plan
    assert plan.count("dynamic_predicate") == 1
    err = bloom_run(q, plmonkeypatch, capfd, [220, 240])
    assert "bloom of" in err

    q = tiny(220, 240, 1000).join(shuffled_fact, on="k", how="semi")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceLeft" in plan
    assert plan.count("dynamic_predicate") == 1
    err = bloom_run(q, plmonkeypatch, capfd, [220, 240])
    assert "bloom of" in err


def test_anti_join_publishes_from_the_left_only(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = tiny(220, 240, 1000).join(shuffled_fact, on="k", how="anti")
    plan = q.explain(engine="streaming")
    assert "BUILD SIDE: ForceLeft" in plan
    assert plan.count("dynamic_predicate") == 1
    err = bloom_run(q, plmonkeypatch, capfd, [1000])
    assert "bloom of" in err

    q = shuffled_fact.join(tiny(220, 240, 1000), on="k", how="anti")
    plan = q.explain(engine="streaming")
    assert "dynamic_predicate" not in plan
    out = q.collect(engine="streaming")
    assert out.height == N_ROW_GROUPS * ROWS_PER_GROUP - 2
    assert_matches_in_memory(q, out)


def test_preferred_semi_join_gets_no_filter(shuffled_fact: pl.LazyFrame) -> None:
    q = shuffled_fact.join(unbounded_dim(220, 240), on="k", how="semi")
    assert "dynamic_predicate" not in q.explain(engine="streaming")


def test_semi_join_range_prunes_row_groups(
    fact: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    q = fact.join(tiny(220, 240), on="k", how="semi")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "1 / 10 row groups"
    assert out.get_column("k").sort().to_list() == [220, 240]
    assert_matches_in_memory(q, out)


def test_top_k_dynamic_predicate_still_filters_rows(
    shuffled_fact: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # A sort with a slice publishes a per-row dynamic predicate of its own that
    # keeps every row until the sort has a bound. The scan must keep evaluating
    # it and never bypass it for keeping too many rows.
    q = shuffled_fact.sort("k").head(5)
    assert "dynamic_predicate" in q.explain(engine="streaming")
    out, err = reader_log(q, plmonkeypatch, capfd)
    assert "Pre-filtered decode enabled" in err
    assert "Dynamic predicate bypassed" not in err
    assert out.get_column("k").to_list() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_join_above_a_semi_anti_join_traces_its_left_columns_only(
    tmp_path: Path, how: JoinStrategy
) -> None:
    # The semi/anti join outputs left columns only, so `k_right` is the left
    # input's own column, not the right input's `k` under the suffix.
    path = tmp_path / "probe.parquet"
    pl.DataFrame({"k": range(20_000)}).write_parquet(path, row_group_size=1_000)
    left = pl.LazyFrame({"k": range(1_000), "k_right": [15_000] * 1_000})
    dimension = pl.LazyFrame({"k_right": [15_000, 15_001], "e": [0, 1]}).filter(
        pl.col("e") >= 0
    )
    query = left.join(
        pl.scan_parquet(path), on="k", how=how, build_side="force_left"
    ).join(dimension, on="k_right")
    assert_frame_equal(
        query.collect(engine="streaming"),
        query.collect(engine="in-memory"),
        check_row_order=False,
    )


def test_repeated_build_key(tmp_path: Path) -> None:
    # Filters are keyed by position; the same build column may appear twice.
    path = tmp_path / "probe.parquet"
    pl.DataFrame({"a": range(1000), "b": range(1000)}).write_parquet(path)
    build = pl.LazyFrame({"k": [220, 240], "e": [0, 1]}).filter(pl.col("e") >= 0)
    query = pl.scan_parquet(path).join(build, left_on=["a", "b"], right_on=["k", "k"])
    flags = pl.QueryOptFlags(predicate_pushdown=False)
    assert_frame_equal(
        query.collect(engine="streaming", optimizations=flags),
        query.collect(engine="in-memory", optimizations=flags),
        check_row_order=False,
    )

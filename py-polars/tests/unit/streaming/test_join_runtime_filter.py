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
    # The first join's build side has no row bound, so it samples and may read the
    # scan before the second join has built: neither join gets to publish.
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
    q = fact.join(dim(-1), on="k")
    out, groups = row_groups_read(q, plmonkeypatch, capfd)
    assert groups == "0 / 10 row groups"
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


def test_only_parquet_scans_get_a_filter(tmp_path: Path) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    path = tmp_path / "fact.ipc"
    pl.DataFrame({"k": range(n), "v": range(n)}).write_ipc(path)
    q = pl.scan_ipc(path).join(dim(220, 240), on="k")
    assert "dynamic_predicate" not in q.explain(engine="streaming")
    out = q.collect(engine="streaming")
    assert out.get_column("k").sort().to_list() == [220, 240]


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
    # the range is published but skips nothing, except when it is empty.
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
    assert groups == "0 / 4 row groups"
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

"""Statistics-based row group skipping must never change what a query answers.

Parquet column statistics are bounds of a column's values: a writer may store a
truncated bound (a 64-byte prefix of a string, or of a binary blob) that no row
contains. Polars rewrites a row predicate into a predicate on those statistics to
skip row groups, and that rewrite substitutes a statistic for a column value. A
substituted expression is arbitrary — it may be `json_decode`, a slice-and-cast, a
WKB parser — and may fail on a bound that is not a value of the column, or fail on
it in a way the column's own values never would. Such an expression is only
evaluated for a row group whose statistics *are* the value of every row in it;
every other row group is read instead of skipped (issue #29447).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import EngineType
    from tests.conftest import PlMonkeyPatch

ENGINES: list[EngineType] = ["streaming", "in-memory"]


def _write(
    path: Path,
    column: str,
    values: list[Any],
    dtype: pl.DataType | None = None,
    row_group_size: int = 250,
) -> Path:
    pl.DataFrame({column: pl.Series(values, dtype=dtype)}).write_parquet(
        path, row_group_size=row_group_size
    )
    return path


def _assert_statistics_are_truncated(
    path: Path, values: list[Any], row_group: int = 0
) -> None:
    """Assert a test's premise: a stored bound is shorter than the values it bounds."""
    stats = pq.read_metadata(path).row_group(row_group).column(0).statistics
    assert stats is not None
    assert stats.has_min_max
    assert stats.min is not None
    assert stats.max is not None
    assert len(stats.min) < max(len(v) for v in values)
    assert len(stats.max) < max(len(v) for v in values)


def _assert_statistics_bound_their_row_group(
    path: Path, values: list[Any], row_group: int = 0
) -> None:
    """Assert a test's premise: a stored bound bounds the values of its row group.

    The rewrite may only skip a row group by comparing a value against a stored bound
    because the format requires the bound to bound, truncated or not.
    """
    md = pq.read_metadata(path)
    stats = md.row_group(row_group).column(0).statistics
    start = sum(md.row_group(i).num_rows for i in range(row_group))
    group = [
        v
        for v in values[start : start + md.row_group(row_group).num_rows]
        if v is not None
    ]
    assert stats is not None
    assert stats.min is not None
    assert stats.max is not None
    assert stats.min <= min(group)
    assert stats.max >= max(group)


def _assert_matches_eager_and_unoptimized(
    path: Path, expr: pl.Expr, engine: EngineType
) -> pl.DataFrame:
    expected = pl.read_parquet(path).filter(expr)
    assert_frame_equal(
        pl.scan_parquet(path).filter(expr).collect(engine=engine),
        expected,
        check_row_order=False,
    )
    assert_frame_equal(
        pl.scan_parquet(path, use_statistics=False).filter(expr).collect(engine=engine),
        expected,
        check_row_order=False,
    )
    return expected


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_json_decode_29447(tmp_path: Path, engine: EngineType) -> None:
    pad = "p" * 200
    values = [f'{{"x": {i}, "pad": "{pad}"}}' for i in range(1000)]
    path = _write(tmp_path / "t.parquet", "s", values, row_group_size=250)
    _assert_statistics_are_truncated(path, values)
    _assert_statistics_bound_their_row_group(path, values)

    expr = (
        pl.col("s")
        .str.json_decode(pl.Struct({"x": pl.Int64, "pad": pl.String}))
        .struct.field("x")
        > 5
    )
    assert _assert_matches_eager_and_unoptimized(path, expr, engine).height == 994


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_long_common_prefix_slice_cast(
    tmp_path: Path, engine: EngineType
) -> None:
    # Every value shares more than a truncation length's worth of prefix, so the min
    # statistic is a prefix ending in non-digits, which `cast` rejects — the values
    # themselves are numeric.
    values = ["A" * 70 + f"{i:04d}" for i in range(1000)]
    path = _write(tmp_path / "t.parquet", "s", values, row_group_size=250)
    _assert_statistics_are_truncated(path, values)
    _assert_statistics_bound_their_row_group(path, values)

    expr = pl.col("s").str.slice(-4).cast(pl.Int64) < 500
    assert _assert_matches_eager_and_unoptimized(path, expr, engine).height == 500


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_truncated_binary(tmp_path: Path, engine: EngineType) -> None:
    # The values are valid UTF-8; the min statistic cuts a multi-byte character in
    # half, so it is not, and a cast to `String` rejects it — like a WKB parser
    # rejecting a truncated blob.
    values = [("A" * 63 + "\u00e9" + f"{i:04d}").encode() for i in range(100)]
    path = _write(tmp_path / "t.parquet", "b", values, pl.Binary, row_group_size=25)
    _assert_statistics_are_truncated(path, values)
    _assert_statistics_bound_their_row_group(path, values)

    expr = pl.col("b").cast(pl.String).str.slice(-2).cast(pl.Int64) < 50
    assert _assert_matches_eager_and_unoptimized(path, expr, engine).height == 50


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_truncated_value_predicate_never_skips(
    tmp_path: Path, engine: EngineType
) -> None:
    # The substituted term is `False` on the truncated min, not an error: only the guard
    # keeps the row group from being skipped, so every row group is read.
    values = ["A" * 70 + f"{i:04d}" for i in range(1000)]
    path = _write(tmp_path / "t.parquet", "s", values, row_group_size=250)
    _assert_statistics_are_truncated(path, values)
    _assert_statistics_bound_their_row_group(path, values)

    expr = pl.col("s").str.len_chars() > 64
    out = _assert_matches_eager_and_unoptimized(path, expr, engine)

    assert out.height == 1000


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_composed_predicate_keeps_its_pruning(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    engine: EngineType,
) -> None:
    # A conjunction of row conditions becomes a disjunction of skip conditions: the
    # one over statistics prunes the row group it proves cannot match, and the one
    # that is only evaluable on a batch's value reads the other.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    path = tmp_path / "t.parquet"
    pl.DataFrame(
        {
            "s": ["A" * 70 + f"{i:04d}" for i in range(500)],
            "k": list(range(250)) + list(range(5000, 5250)),
        }
    ).write_parquet(path, row_group_size=250)

    expr = (pl.col("k") < 1000) & (pl.col("s").str.slice(-4).cast(pl.Int64) < 500)
    capfd.readouterr()
    out = pl.scan_parquet(path).filter(expr).collect(engine=engine)

    assert "reading 1 / 2 row groups" in capfd.readouterr().err
    assert_frame_equal(out, pl.read_parquet(path).filter(expr), check_row_order=False)


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_exact_min_max_still_skip_row_groups(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    engine: EngineType,
) -> None:
    # Each row group is a single value whose statistics are exactly that value: the
    # rewritten predicate is evaluated on it, and the row group whose value fails it
    # is skipped.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    path = _write(
        tmp_path / "t.parquet",
        "s",
        ["abcdefg"] * 250 + ["abc"] * 250,
        row_group_size=250,
    )

    expr = pl.col("s").str.len_chars() == 7
    capfd.readouterr()
    out = pl.scan_parquet(path).filter(expr).collect(engine=engine)

    assert "reading 1 / 2 row groups" in capfd.readouterr().err
    assert_frame_equal(out, pl.read_parquet(path).filter(expr), check_row_order=False)


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_truncated_bounds_still_skip_row_groups(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    engine: EngineType,
) -> None:
    # Truncated statistics are still bounds of the values they belong to: a
    # comparison against one keeps pruning the row groups it proves cannot match.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    values = ["M" * 10] * 250 + ["A" * 70 + f"{i:04d}" for i in range(250)]
    path = _write(tmp_path / "t.parquet", "s", values, row_group_size=250)
    _assert_statistics_are_truncated(path, values, row_group=1)
    _assert_statistics_bound_their_row_group(path, values, row_group=0)
    _assert_statistics_bound_their_row_group(path, values, row_group=1)

    expr = pl.col("s") == "M" * 10
    capfd.readouterr()
    out = pl.scan_parquet(path).filter(expr).collect(engine=engine)

    assert "reading 1 / 2 row groups" in capfd.readouterr().err
    assert_frame_equal(out, pl.read_parquet(path).filter(expr), check_row_order=False)


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_unprovable_row_groups_are_read(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    engine: EngineType,
) -> None:
    # A row group whose statistics cannot be shown to be the value of a row is read,
    # never skipped: here the statistics are truncated, so the predicate cannot be
    # evaluated on them at all.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    values = ["A" * 70 + f"{i:04d}" for i in range(1000)]
    path = _write(tmp_path / "t.parquet", "s", values, row_group_size=250)
    _assert_statistics_are_truncated(path, values)
    _assert_statistics_bound_their_row_group(path, values)

    expr = pl.col("s").str.slice(-4).cast(pl.Int64) < 500
    capfd.readouterr()
    out = pl.scan_parquet(path).filter(expr).collect(engine=engine)

    assert "reading 4 / 4 row groups" in capfd.readouterr().err
    assert_frame_equal(out, pl.read_parquet(path).filter(expr), check_row_order=False)


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_all_null_row_group_is_read(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    engine: EngineType,
) -> None:
    # A row group with no bounds reports none, and one with nulls does not report
    # `null_count == 0`: neither proves the batch's value, so both are read.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    values: list[str | None] = [None] * 250 + [
        "A" * 70 + f"{i:04d}" for i in range(250)
    ]
    path = _write(tmp_path / "t.parquet", "s", values, row_group_size=250)

    expr = pl.col("s").str.slice(-4).cast(pl.Int64) < 500
    capfd.readouterr()
    out = pl.scan_parquet(path).filter(expr).collect(engine=engine)

    assert "reading 2 / 2 row groups" in capfd.readouterr().err
    assert_frame_equal(out, pl.read_parquet(path).filter(expr), check_row_order=False)
    assert out.height == 250


@pytest.mark.write_disk
@pytest.mark.parametrize("engine", ENGINES)
def test_statistics_hive_partition_constant_is_a_value(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    engine: EngineType,
) -> None:
    # A hive partition column's statistics are its value for every row of its file, so a
    # predicate that mixes it with a column that is a single value per file is evaluated
    # on real values, and the file whose value fails it is skipped.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    for part, value in (("abc", 5), ("abcdefg", 1)):
        (tmp_path / f"part={part}").mkdir()
        pl.DataFrame({"v": [value] * 3}).write_parquet(
            tmp_path / f"part={part}" / "d.parquet"
        )

    expr = pl.col("v") > pl.col("part").str.len_chars()
    capfd.readouterr()
    out = pl.scan_parquet(tmp_path).filter(expr).collect(engine=engine)

    assert "reading 0 / 1 row groups" in capfd.readouterr().err
    assert out.get_column("part").unique().to_list() == ["abc"]
    assert out.height == 3

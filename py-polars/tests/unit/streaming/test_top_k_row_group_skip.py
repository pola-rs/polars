"""A top-k over a parquet scan skips the row groups that cannot reach the top k."""

from __future__ import annotations

import random
from datetime import date, datetime
from decimal import Decimal as D
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")

N_ROW_GROUPS = 200
ROWS_PER_GROUP = 10


def write_shuffled_groups(tmp_path: Path, df: pl.DataFrame) -> pl.LazyFrame:
    """Write `df` sorted, in row groups that are then shuffled."""
    size = ROWS_PER_GROUP
    groups = [df.slice(i, size) for i in range(0, df.height, size)]
    random.Random(0).shuffle(groups)
    path = tmp_path / "data.parquet"
    pl.concat(groups).write_parquet(path, row_group_size=ROWS_PER_GROUP)
    return pl.scan_parquet(path)


def reader_log(
    q: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> tuple[pl.DataFrame, str]:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    return out, capfd.readouterr().err


def assert_matches_in_memory(q: pl.LazyFrame, out: pl.DataFrame) -> None:
    assert_frame_equal(out, q.collect(engine="in-memory"))


@pytest.mark.parametrize("descending", [False, True])
def test_top_k_skips_row_groups(
    descending: bool,
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    lf = write_shuffled_groups(tmp_path, pl.DataFrame({"k": range(n), "v": range(n)}))
    q = lf.sort("k", descending=descending).head(5)

    out, log = reader_log(q, plmonkeypatch, capfd)
    assert_matches_in_memory(q, out)
    order = "max" if descending else "min"
    assert f"Reading row groups by the {order} of 'k'" in log
    assert "Runtime ranges skipped" in log


def test_top_k_multiple_keys_skips_row_groups(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    df = pl.DataFrame({"a": [i // 3 for i in range(n)], "b": [-i for i in range(n)]})
    lf = write_shuffled_groups(tmp_path, df)
    q = lf.sort("a", "b").head(5)
    assert "dynamic_predicate" in q.explain()

    out, log = reader_log(q, plmonkeypatch, capfd)
    assert_matches_in_memory(q, out)
    assert "Reading row groups by the min of 'a'" in log
    assert "Runtime ranges skipped" in log


def test_top_k_maintain_order_keeps_file_order(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    df = pl.DataFrame({"k": [i % 3 for i in range(n)], "v": range(n)})
    lf = write_shuffled_groups(tmp_path, df)
    q = lf.sort("k", maintain_order=True).head(15)

    out, log = reader_log(q, plmonkeypatch, capfd)
    assert_matches_in_memory(q, out)
    assert "Reading row groups by" not in log


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Int32, [5, -3, 7, 0, 2, 9, -8, 4]),
        (pl.UInt8, [5, 3, 7, 0, 2, 9, 8, 4]),
        (pl.Float64, [0.5, float("nan"), -1.5, 3.0, float("inf"), 0.0, -0.0, 2.0]),
        (pl.Date, [date(2020, 1, d) for d in (5, 3, 7, 1, 2, 9, 8, 4)]),
        (
            pl.Datetime("ms"),
            [datetime(2020, 1, 1, h) for h in (5, 3, 7, 1, 2, 9, 8, 4)],
        ),
        (pl.Decimal(10, 2), [D(f"{v}.25") for v in (5, -3, 7, 0, 2, 9, -8, 4)]),
        (pl.String, ["e", "c", "g", "", "b", "zz", "h", "d"]),
        (pl.Binary, [b"e", b"c", b"g", b"", b"b", b"zz", b"h", b"d"]),
        (pl.Enum(["z", "a", "m"]), ["a", "m", "z", "a", "z", "m", "a", "z"]),
    ],
)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("multiple_keys", [False, True])
def test_top_k_row_group_skip_matches_in_memory(
    dtype: pl.DataType,
    values: list[Any],
    descending: bool,
    nulls_last: bool,
    multiple_keys: bool,
    tmp_path: Path,
) -> None:
    rng = random.Random(1)
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    keys = [rng.choice([*values, None]) for _ in range(n)]
    # The last row group holds nulls next to large values, so its statistics alone
    # would place it after the threshold.
    keys[-ROWS_PER_GROUP:] = [None, max(v for v in values if v == v)] * (
        ROWS_PER_GROUP // 2
    )
    df = pl.DataFrame(
        {"k": keys, "t": [i % 5 for i in range(n)], "v": range(n)},
        schema={"k": dtype, "t": pl.Int64, "v": pl.Int64},
    )
    path = tmp_path / "data.parquet"
    df.write_parquet(path, row_group_size=ROWS_PER_GROUP)
    lf = pl.scan_parquet(path)

    by = ["k", "t", "v"] if multiple_keys else ["k"]
    for k in [1, 7, 40, n + 5]:
        q = lf.sort(by, descending=descending, nulls_last=nulls_last).head(k)
        out = q.collect(engine="streaming")
        expected = q.collect(engine="in-memory")
        assert_frame_equal(out.select(by), expected.select(by))

    q = lf.sort(by, descending=descending, nulls_last=nulls_last).slice(3, 4)
    assert_frame_equal(
        q.collect(engine="streaming").select(by),
        q.collect(engine="in-memory").select(by),
    )


def test_top_k_with_filter_matches_in_memory(tmp_path: Path) -> None:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    df = pl.DataFrame(
        {
            "t": [(i * 7919) % n for i in range(n)],
            "s": [f"x{i % 13}" if i % 4 else "" for i in range(n)],
        }
    )
    lf = write_shuffled_groups(tmp_path, df)
    for q in [
        lf.filter(pl.col("s") != "").sort("t").head(10),
        lf.filter(pl.col("s").str.contains("1")).sort("t", descending=True).head(3),
        lf.filter(pl.col("s") != "").sort("t", "s").select("s").head(10),
    ]:
        assert_matches_in_memory(q, q.collect(engine="streaming"))


def test_top_k_threshold_lowers_after_heap_is_full(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # A row index keeps the file order, so the best key comes after the heap is full.
    # The long tail keeps the reader from decoding all of it before the threshold drops.
    plmonkeypatch.setenv("POLARS_ROW_GROUP_PREFETCH_SIZE", "1")
    df = pl.DataFrame({"k": [1000] * 1000 + [0] * 10 + [500] * 30000})
    path = tmp_path / "data.parquet"
    df.write_parquet(path, row_group_size=ROWS_PER_GROUP)
    q = pl.scan_parquet(path, row_index_name="idx").sort("k").head(1)

    out, log = reader_log(q, plmonkeypatch, capfd)
    assert out["k"].to_list() == [0]
    assert "Reading row groups by" not in log
    assert "Runtime ranges skipped" in log

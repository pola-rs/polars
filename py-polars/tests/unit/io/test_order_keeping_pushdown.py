"""Predicates on a column computed by an order-keeping expression reach the scan."""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

N_ROW_GROUPS = 200
ROWS_PER_GROUP = 10


@pytest.fixture
def scan(tmp_path: Path) -> pl.LazyFrame:
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    df = pl.DataFrame(
        {
            "t": range(n),
            "d": [i // ROWS_PER_GROUP for i in range(n)],
            "f": [float(i) for i in range(n)],
            "s": [str(i) for i in range(n)],
        },
        schema={"t": pl.Int32, "d": pl.UInt16, "f": pl.Float64, "s": pl.String},
    )
    groups = [df.slice(i, ROWS_PER_GROUP) for i in range(0, n, ROWS_PER_GROUP)]
    random.Random(0).shuffle(groups)
    path = tmp_path / "data.parquet"
    pl.concat(groups).write_parquet(path, row_group_size=ROWS_PER_GROUP)
    return pl.scan_parquet(path)


def scan_selection(q: pl.LazyFrame) -> str:
    plan = q.explain()
    assert "FILTER" not in plan
    return next(line for line in plan.splitlines() if "SELECTION" in line)


def reader_log(
    q: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> tuple[pl.DataFrame, str]:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    log = capfd.readouterr().err
    plmonkeypatch.setenv("POLARS_VERBOSE", "0")
    return out, log


ORDER_KEEPING = [
    (pl.col("t").cast(pl.Int64) * 1_000_000).cast(pl.Datetime("us")),
    pl.col("t") + 5,
    100 - pl.col("t"),
    -pl.col("t"),
    pl.col("t") * -2,
    pl.col("t") // 3,
    pl.col("d").cast(pl.Date),
    pl.col("d").cast(pl.Date).cast(pl.Datetime("ms")),
]


@pytest.mark.parametrize("expr", ORDER_KEEPING)
@pytest.mark.parametrize("descending", [False, True])
def test_top_k_through_order_keeping_projection(
    expr: pl.Expr,
    descending: bool,
    scan: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    q = scan.with_columns(k=expr).sort("k", descending=descending).head(5)
    assert "dynamic_predicate" in scan_selection(q)

    out, log = reader_log(q, plmonkeypatch, capfd)
    assert "Runtime ranges skipped" in log
    assert_frame_equal(out.select("k"), q.collect(engine="in-memory").select("k"))


@pytest.mark.parametrize("expr", ORDER_KEEPING)
def test_filter_through_order_keeping_projection(
    expr: pl.Expr,
    scan: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    middle = scan.with_columns(k=expr).select(pl.col("k").sort().gather(1000)).collect()
    q = scan.with_columns(k=expr).filter(pl.col("k") < middle.item())
    assert "SELECTION" in scan_selection(q)

    out, log = reader_log(q, plmonkeypatch, capfd)
    assert f"reading {N_ROW_GROUPS} / {N_ROW_GROUPS} row groups" not in log
    assert "row groups" in log
    assert_frame_equal(out, q.collect(engine="in-memory"), check_row_order=False)


def test_filter_through_date_cast(scan: pl.LazyFrame) -> None:
    lf = scan.with_columns(pl.col("d").cast(pl.Date))
    q = lf.filter(
        (pl.col("d") >= date(1970, 3, 1)) & (pl.col("d") <= date(1970, 3, 31))
    )
    assert "cast(Date)" in scan_selection(q)
    assert_frame_equal(q.collect(), q.collect(engine="in-memory"))
    assert q.select(pl.len()).collect().item() == 31 * ROWS_PER_GROUP


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("f") * 2,
        pl.col("s").cast(pl.Int64),
        pl.col("t") * 0,
        pl.col("t").cast(pl.Int8, strict=False),
        pl.col("t") + pl.col("d"),
        pl.col("t").cast(pl.String),
    ],
)
def test_not_order_keeping_stays_above_projection(
    expr: pl.Expr, scan: pl.LazyFrame
) -> None:
    q = scan.with_columns(k=expr).filter(pl.col("k").is_not_null()).head(3)
    assert "FILTER" in q.explain()
    assert_frame_equal(q.collect(), q.collect(engine="in-memory"))


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_top_k_through_order_keeping_projection_with_nulls(
    descending: bool, nulls_last: bool, tmp_path: Path
) -> None:
    rng = random.Random(1)
    n = N_ROW_GROUPS * ROWS_PER_GROUP
    t = [rng.choice([None, *range(50)]) for _ in range(n)]
    path = tmp_path / "data.parquet"
    pl.DataFrame({"t": t}).write_parquet(path, row_group_size=ROWS_PER_GROUP)
    lf = pl.scan_parquet(path).with_columns(k=100 - pl.col("t"))
    for k in [1, 7, 40]:
        q = lf.sort("k", descending=descending, nulls_last=nulls_last).head(k)
        assert_frame_equal(
            q.collect(engine="streaming").select("k"),
            q.collect(engine="in-memory").select("k"),
        )


def test_top_k_through_select_and_sql(
    scan: pl.LazyFrame,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    lf = scan.select(t=pl.col("t") * 2, s=pl.col("s"))
    q = lf.sort("t").head(5)
    assert "dynamic_predicate" in scan_selection(q)
    out, log = reader_log(q, plmonkeypatch, capfd)
    assert "Runtime ranges skipped" in log
    assert out["t"].to_list() == [0, 2, 4, 6, 8]

    q = pl.SQLContext(hits=lf).execute("SELECT s FROM hits ORDER BY t DESC LIMIT 3")
    out, log = reader_log(q, plmonkeypatch, capfd)
    assert "Runtime ranges skipped" in log
    assert out["s"].to_list() == ["1999", "1998", "1997"]


def test_top_k_through_two_projections(scan: pl.LazyFrame) -> None:
    q = (
        scan.with_columns(t=pl.col("t") * 2)
        .with_columns(t=1000 - pl.col("t"))
        .sort("t")
        .head(3)
    )
    assert "dynamic_predicate" in scan_selection(q)
    assert q.collect()["t"].to_list() == [1000 - 2 * 1999, 1000 - 2 * 1998, -2994]


def test_top_k_on_order_keeping_hive_column(tmp_path: Path) -> None:
    for p in range(5):
        (tmp_path / f"p={p}").mkdir()
        pl.DataFrame({"v": [p * 10 + i for i in range(3)]}).write_parquet(
            tmp_path / f"p={p}" / "data.parquet"
        )
    lf = pl.scan_parquet(tmp_path, hive_partitioning=True).with_columns(
        p=10 - pl.col("p")
    )
    q = lf.sort("p", "v").head(4)
    assert_frame_equal(q.collect(), q.collect(engine="in-memory"))
    assert q.collect()["v"].to_list() == [40, 41, 42, 30]


def test_sort_by_order_keeping_expression_keeps_filter() -> None:
    lf = pl.LazyFrame({"x": [1, 2, 3]}).sort(pl.col("x") + 1)
    assert lf.filter(pl.col("x") == 2).collect()["x"].to_list() == [2]


def test_top_k_by_order_keeping_second_key(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    df = pl.DataFrame({"x": [100] * 1000 + [80] * 1000 + [60] * 1000})
    df.write_parquet(path, row_group_size=ROWS_PER_GROUP)
    q = pl.scan_parquet(path, row_index_name="i").sort(["x", pl.col("x") * 2]).head(1)
    assert q.collect()["x"].to_list() == [60]


HUGE_DATES = pl.Series([2**30] * 10).cast(pl.Date)


@pytest.mark.parametrize(
    ("values", "expr"),
    [
        (
            pl.Series([date(2500, 1, 1)] * 10),
            pl.col("t").cast(pl.Datetime("ns"), strict=False),
        ),
        (HUGE_DATES, pl.col("t").cast(pl.Datetime("us", "UTC"), strict=False)),
        (
            pl.Series([2**62] * 10).cast(pl.Datetime("ms")),
            pl.col("t").cast(pl.Datetime("us"), strict=False),
        ),
        (
            pl.Series([2**62] * 10).cast(pl.Duration("us")),
            pl.col("t").cast(pl.Duration("ns"), strict=False),
        ),
        (HUGE_DATES, pl.col("t") + timedelta(days=1)),
        (HUGE_DATES, pl.col("t") - date(1970, 1, 1)),
        (
            pl.Series(range(10)).cast(pl.Duration("us")),
            pl.col("t") * pl.lit(2**63, pl.UInt64),
        ),
    ],
)
def test_expression_that_gives_null_keeps_null_filter(
    values: pl.Series, expr: pl.Expr, tmp_path: Path
) -> None:
    path = tmp_path / "data.parquet"
    pl.DataFrame({"t": values}).write_parquet(path)
    lf = pl.scan_parquet(path).with_columns(expr)
    for predicate in [pl.col("t").is_null(), pl.col("t").is_not_null()]:
        q = lf.filter(predicate)
        assert_frame_equal(
            q.collect(), lf.collect(engine="in-memory").filter(predicate)
        )


@pytest.mark.parametrize(
    ("dtype", "expr"),
    [
        (pl.UInt32, -pl.col("t")),
        (pl.Date, -pl.col("t")),
        (pl.Int128, -pl.col("t")),
        (pl.Duration("us"), pl.col("t") - datetime(1970, 1, 1)),
        (pl.Datetime("us"), pl.col("t") - datetime(1970, 1, 1, tzinfo=UTC)),
    ],
)
def test_expression_that_raises_is_evaluated(
    dtype: pl.DataType, expr: pl.Expr, tmp_path: Path
) -> None:
    path = tmp_path / "data.parquet"
    pl.DataFrame({"t": [1, 2, 3]}).cast(dtype).write_parquet(path)
    q = pl.scan_parquet(path).with_columns(k=expr).filter(pl.col("k").is_null())
    with pytest.raises(pl.exceptions.PolarsError):
        q.collect()

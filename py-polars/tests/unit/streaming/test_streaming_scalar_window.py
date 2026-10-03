from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.fixture(autouse=True)
def small_morsels(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "100")


def _frame(n: int = 3_000) -> pl.DataFrame:
    i = pl.col("id")
    return pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        g=pl.when(i % 101 == 0).then(None).otherwise(i % 97),
        h=(i % 13).cast(pl.String),
        k=i % 5,
        x=pl.when(i % 7 == 0).then(None).otherwise((i * 104729) % 1_000).cast(pl.Int64),
        y=((i * 7919) % 500).cast(pl.Float64),
        z=pl.when(i % 3 == 0).then(None).otherwise(i % 11).cast(pl.Int8),
        n=pl.lit(None, dtype=pl.Int32),
    )


def _scan(df: pl.DataFrame, path: Path, parts: int = 6) -> pl.LazyFrame:
    size = df.height // parts + 1
    for i in range(parts):
        df.slice(i * size, size).write_parquet(path / f"{i:02}.parquet")
    return pl.scan_parquet(path / "*.parquet")


def _scalar_flags(q: pl.LazyFrame) -> list[bool]:
    """Whether each window node of the physical plan is a scalar window."""
    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    return ["scalar-window[" in line for line in dot.splitlines() if "window[" in line]


def _assert_same(q: pl.LazyFrame, *, check_row_order: bool = True) -> None:
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=check_row_order,
        check_exact=False,
    )


def _assert_same_on_path(
    q: pl.LazyFrame, path: str, plmonkeypatch: PlMonkeyPatch, capfd: Any
) -> list[str]:
    """Like `_assert_same`, and checks that every scalar window reduces on `path`.

    Returns the verbose lines of the scalar windows.
    """
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect(engine="streaming")
    plmonkeypatch.delenv("POLARS_VERBOSE")
    lines = [
        line
        for line in capfd.readouterr().err.splitlines()
        if line.startswith("[scalar-window]")
    ]
    paths = {
        "local" if "per pipeline" in line else "partitioned"
        for line in lines
        if line.startswith("[scalar-window]: reduce ")
    }
    assert paths == {path}
    assert_frame_equal(out, q.collect(engine="in-memory"), check_exact=False)
    return lines


@pytest.mark.parametrize(
    "exprs",
    [
        [pl.col("x").sum().over("g").alias("w")],
        [pl.col("x").mean().over("g").alias("w")],
        [pl.col("y").min().over("h").alias("w")],
        [pl.col("y").max().over("h").alias("w")],
        [pl.col("x").count().over("g").alias("w")],
        [pl.len().over("g").alias("w")],
        [pl.col("z").sum().over("k").alias("w")],
        [pl.col("n").sum().over("g").alias("w")],
        [pl.col("n").mean().over("g").alias("w")],
        [pl.col("x").sum().over("g", "h").alias("w")],
        [pl.col("x").sum().over("g", "h", "k").alias("w")],
        [
            pl.col("x").sum().over("g").alias("a"),
            pl.col("y").mean().over("g").alias("b"),
            pl.col("z").min().over("g").alias("c"),
            pl.col("x").max().over("g").alias("d"),
            pl.col("x").count().over("g").alias("e"),
            pl.len().over("g").alias("f"),
        ],
        [(pl.col("y") - pl.col("y").mean().over("k")).alias("w")],
        [(pl.col("x").sum().over("g") / pl.col("x").count().over("g")).alias("w")],
    ],
)
def test_scalar_window_matches_in_memory(tmp_path: Path, exprs: list[pl.Expr]) -> None:
    q = _scan(_frame(), tmp_path).with_columns(exprs)
    assert _scalar_flags(q) == [True]
    _assert_same(q)


def test_scalar_window_from_in_memory_source() -> None:
    q = _frame().lazy().with_columns(w=pl.col("x").sum().over("g"))
    assert _scalar_flags(q) == [True]
    _assert_same(q)


def test_scalar_window_two_specs(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(
        a=pl.col("x").sum().over("g"),
        b=pl.col("x").sum().over("h"),
        c=pl.col("y").mean().over("g"),
    )
    assert _scalar_flags(q) == [True, True]
    _assert_same(q)


@pytest.mark.parametrize(
    "consumer",
    [
        lambda lf: lf.select(pl.col("w").sum(), pl.col("id").sum()),
        lambda lf: lf.group_by("k").agg(pl.col("w").first(), pl.col("id").sum()),
        lambda lf: lf.sort("id"),
    ],
)
def test_scalar_window_unordered_consumer(
    tmp_path: Path, consumer: Callable[[pl.LazyFrame], pl.LazyFrame]
) -> None:
    q = _scan(_frame(), tmp_path).with_columns(w=pl.col("x").sum().over("k"))
    q = consumer(q)
    assert _scalar_flags(q) == [True]
    _assert_same(q, check_row_order=False)


def test_scalar_window_single_group(tmp_path: Path) -> None:
    q = (
        _scan(_frame(), tmp_path)
        .with_columns(o=pl.lit(1))
        .with_columns(w=pl.col("y").sum().over("o"))
    )
    assert _scalar_flags(q) == [True]
    _assert_same(q)


def test_scalar_window_empty() -> None:
    q = (
        _frame()
        .lazy()
        .filter(pl.col("id") < 0)
        .with_columns(w=pl.col("x").sum().over("g"), v=pl.col("y").mean().over("g"))
    )
    out = q.collect(engine="streaming")
    assert out.height == 0
    assert_frame_equal(out, q.collect(engine="in-memory"))


def test_scalar_window_stops_early(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(w=pl.col("x").sum().over("k")).head(5)
    _assert_same(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").median().over("g"),
        pl.col("x").first().over("g"),
        pl.col("x").sum().over("g", order_by="id"),
        pl.col("x").filter(pl.col("y") > 100).sum().over("g"),
        (pl.col("x") - pl.col("x").mean()).over("g"),
        pl.col("x").cum_sum().over("g"),
        pl.col("x").sum().over("g", mapping_strategy="join"),
        pl.col("h").count().over("g"),
        (pl.col("x") + 1).sum().over("g"),
    ],
)
def test_scalar_window_fallbacks(tmp_path: Path, expr: pl.Expr) -> None:
    q = _scan(_frame(), tmp_path).with_columns(w=expr)
    assert not any(_scalar_flags(q))
    _assert_same(q)


def test_scalar_window_mixed_spec_falls_back(tmp_path: Path) -> None:
    q = _scan(_frame(), tmp_path).with_columns(
        a=pl.col("x").sum().over("g"), b=pl.col("x").cum_sum().over("g")
    )
    assert _scalar_flags(q) == [False]
    _assert_same(q)


@pytest.mark.parametrize(("groups", "path"), [(10, "local"), (100_000, "partitioned")])
def test_scalar_window_many_groups(
    tmp_path: Path, groups: int, path: str, plmonkeypatch: PlMonkeyPatch, capfd: Any
) -> None:
    n = 100_000
    df = pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        k=(pl.col("id") * 7919) % groups,
        j=pl.col("id") % 3,
        x=pl.when(pl.col("id") % 5 == 0).then(None).otherwise(pl.col("id") % 1_000),
    )
    lf = _scan(df, tmp_path)
    q = lf.with_columns(
        a=pl.col("x").sum().over("k"),
        b=pl.col("x").mean().over("k"),
        c=pl.len().over("k"),
    )
    assert _scalar_flags(q) == [True]
    _assert_same_on_path(q, path, plmonkeypatch, capfd)
    _assert_same(lf.with_columns(a=pl.col("x").max().over("k", "j")))


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ],
)
def test_scalar_window_value_dtypes(dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        {"k": [1, 2, 1, 2, 3, 1], "x": [1, None, 3, 4, None, 6]},
        schema={"k": pl.Int64, "x": dtype},
    )
    q = df.lazy().with_columns(
        s=pl.col("x").sum().over("k"),
        m=pl.col("x").mean().over("k"),
        lo=pl.col("x").min().over("k"),
        hi=pl.col("x").max().over("k"),
        c=pl.col("x").count().over("k"),
    )
    assert _scalar_flags(q) == [True]
    _assert_same(q)


@pytest.mark.parametrize("groups", [7, 100_000])
@pytest.mark.parametrize(
    "key",
    [
        pl.col("i").cast(pl.String),
        pl.col("i").cast(pl.String).cast(pl.Categorical),
        pl.col("i").cast(pl.Decimal(10, 2)),
        pl.col("i").cast(pl.Float64) / 2 - 1,
        pl.col("i").cast(pl.Date),
        pl.col("i") % 2 == 0,
        pl.struct(a=pl.col("i"), b=pl.col("i").cast(pl.String)),
        pl.concat_list(pl.col("i"), pl.col("i") % 3),
    ],
)
def test_scalar_window_key_dtypes(
    tmp_path: Path,
    key: pl.Expr,
    groups: int,
    plmonkeypatch: PlMonkeyPatch,
    capfd: Any,
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "5000")
    n = 100_000
    df = pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        i=pl.when(pl.col("id") % 11 == 0).then(None).otherwise(pl.col("id") % groups),
        x=pl.col("id") % 1_000,
    )
    q = (
        _scan(df, tmp_path)
        .with_columns(k=key)
        .with_columns(s=pl.col("x").sum().over("k"), c=pl.len().over("k"))
    )
    assert _scalar_flags(q) == [True]
    n_keys = df.select(key.n_unique()).item()
    path = "local" if n_keys < 1_000 else "partitioned"
    _assert_same_on_path(q, path, plmonkeypatch, capfd)


# Spill everything the memory manager can: no budget, and every allocation is counted.
SPILL_ENV = {
    "POLARS_OOC_MEMORY_BUDGET_MB": "0",
    "POLARS_OOC_SPILL_MIN_BYTES": "1",
    "POLARS_OOC_DRIFT_THRESHOLD": "0",
}


@pytest.mark.write_disk
@pytest.mark.parametrize(("groups", "path"), [(7, "local"), (100_000, "partitioned")])
def test_scalar_window_spilled(
    tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: Any, groups: int, path: str
) -> None:
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    for name, value in SPILL_ENV.items():
        plmonkeypatch.setenv(name, value)
    plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(spill_dir))
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "5000")

    n = 100_000
    i = pl.col("id")
    df = pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
        k=pl.when(i % 13 == 0).then(None).otherwise((i * 7919) % groups),
        j=i % 3,
        x=pl.when(i % 7 == 0).then(None).otherwise(i % 1_000),
        y=(i % 17).cast(pl.Float64),
    )
    lf = _scan(df, tmp_path)
    q = lf.with_columns(
        a=pl.col("x").sum().over("k"),
        b=pl.col("y").mean().over("k"),
        c=pl.len().over("k"),
        d=pl.col("x").max().over("k", "j"),
    )
    assert _scalar_flags(q) == [True, True]
    lines = _assert_same_on_path(q, path, plmonkeypatch, capfd)
    if path == "partitioned":
        # With a memory budget of zero, every block holds one morsel.
        counts = re.findall(r"reduced (\d+) morsels in (\d+) blocks", "\n".join(lines))
        assert len(counts) == 2
        assert all(int(m) > 1 and m == b for m, b in counts)
    _assert_same(q.slice(1_000, 500))
    _assert_same(q.with_columns(e=pl.col("a").mean().over("j")))
    _assert_same(pl.concat([q.filter(i % 2 == 0), q.filter(i % 2 == 1)]))
    other = lf.select("id", z=i * 2)
    _assert_same(q.join(other, on="id", maintain_order="left"))
    _assert_same(other.join(q, on="id", how="left", maintain_order="left"))


@pytest.mark.write_disk
def test_scalar_window_spill_files_removed(tmp_path: Path) -> None:
    # The spill directory is fixed per process, so use a fresh one.
    script = textwrap.dedent(
        """
        import os, pathlib, time
        import polars as pl

        n = 100_000
        i = pl.col("id")
        df = pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(
            k=(i * 7919) % n, l=i % 7, x=i % 1_000, s=pl.lit("a")
        )
        for key in ["k", "l"]:
            q = df.lazy().with_columns(w=pl.col("x").sum().over(key))
            q.collect(engine="streaming")
            assert q.head(5).collect(engine="streaming").height == 5
            try:
                q.with_columns(pl.col("s").cast(pl.Int64)).collect(engine="streaming")
                raise AssertionError("expected an error")
            except pl.exceptions.InvalidOperationError:
                pass

        process_dir = pathlib.Path(os.environ["POLARS_OOC_SPILL_DIR"]) / str(os.getpid())
        assert process_dir.exists(), "nothing was spilled"
        for _ in range(200):
            if not any(process_dir.iterdir()):
                break
            time.sleep(0.05)
        assert not any(process_dir.iterdir()), list(process_dir.iterdir())
        """
    )
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    env = {
        **os.environ,
        **SPILL_ENV,
        "POLARS_OOC_SPILL_DIR": str(spill_dir),
        "POLARS_IDEAL_MORSEL_SIZE": "5000",
    }
    subprocess.run([sys.executable, "-c", script], env=env, check=True)

"""
Benchmarks of struct-field projection pushdown (not collected by pytest).

    python struct_field_pruning.py generate DIR [--rows N]
    python struct_field_pruning.py read DIR [--repeat N]
    python struct_field_pruning.py plan [--fields N] [--repeat N]

`generate` writes `DIR/wide.parquet`: a column `s`, a struct of 50 fields (25 numeric,
25 strings), and an `id` column. `read` reads 1, 2, 10 and all 50 fields, each with
projection pushdown (the struct pruned) and without it, and reports the time and the
bytes read from the file. `plan` reports the time to optimize a query over a struct of
10K fields, selected whole, by a few fields and for its validity.

Bytes read are the process's `rchar` (Linux `/proc/self/io`) with
`POLARS_FORCE_ASYNC=1`, so a local file is read with `pread` rather than memory-mapped;
the bytes the selected leaves take in the file (compressed) are given beside them.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("POLARS_FORCE_ASYNC", "1")

import pyarrow.parquet as pq

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable

N_FIELDS = 50
NO_PUSHDOWN = pl.QueryOptFlags(projection_pushdown=False)


def field_name(i: int) -> str:
    return f"n{i:02}" if i % 2 == 0 else f"s{i:02}"


def generate(out_dir: Path, rows: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "wide.parquet"

    i = pl.int_range(pl.len(), dtype=pl.Int64)
    fields = []
    for k in range(N_FIELDS):
        h = (i + k * 1_000_003).hash(k)
        if k % 2 == 0:
            expr = (h % 1_000_000).cast(pl.Int64) if k % 4 == 0 else h.cast(pl.Float64)
        else:
            # ~24 bytes, low repetition: dictionary encoding does not hide the size.
            expr = pl.format("v{}-{}", h % 10_000_019, h % 97)
        fields.append(expr.alias(field_name(k)))

    (
        pl.LazyFrame()
        .select(pl.repeat(0, rows))
        .select(id=i, s=pl.struct(fields))
        .sink_parquet(path, row_group_size=256 * 1024)
    )
    return path


def read_bytes() -> int:
    with Path("/proc/self/io").open() as f:
        for line in f:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    msg = "no rchar in /proc/self/io"
    raise RuntimeError(msg)


def leaf_bytes(path: Path, fields: list[str] | None) -> int:
    """Compressed bytes of the leaves of `s` for `fields` (all if `None`)."""
    md = pq.ParquetFile(path).metadata
    total = 0
    for rg in range(md.num_row_groups):
        for c in range(md.num_columns):
            col = md.row_group(rg).column(c)
            parts = col.path_in_schema.split(".")
            if parts[0] == "s" and (fields is None or parts[1] in fields):
                total += col.total_compressed_size
    return total


def measure(f: Callable[[], object], repeat: int) -> tuple[float, float, int]:
    f()  # Warm up (and the page cache).
    times = []
    nbytes = []
    for _ in range(repeat):
        b = read_bytes()
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
        nbytes.append(read_bytes() - b)
    return min(times), statistics.median(times), int(statistics.median(nbytes))


def bench_read(data_dir: Path, repeat: int, engine: str) -> None:
    path = data_dir / "wide.parquet"
    lf = pl.scan_parquet(path)
    n_rows = lf.select(pl.len()).collect().item()
    print(
        f"file: {path} ({path.stat().st_size / 1e6:.1f} MB, {n_rows} rows), "
        f"engine: {engine}, polars {pl.__version__}, repeat: {repeat}"
    )
    print(
        f"{'fields':>8} {'mode':>12} {'min s':>8} {'median s':>9} "
        f"{'read MB':>9} {'leaves MB':>10}"
    )

    selections: list[list[str] | None] = [
        [field_name(1)],
        [field_name(0), field_name(1)],
        [field_name(k) for k in range(10)],
        None,
    ]
    for fields in selections:
        exprs = (
            [pl.col("s")]
            if fields is None
            else [pl.col("s").struct.field(f) for f in fields]
        )
        q = lf.select(exprs)
        for mode, opts in [
            ("pushdown", pl.QueryOptFlags()),
            ("no pushdown", NO_PUSHDOWN),
        ]:
            t_min, t_med, nbytes = measure(
                lambda q=q, opts=opts: q.collect(engine=engine, optimizations=opts),  # type: ignore[misc]
                repeat,
            )
            print(
                f"{N_FIELDS if fields is None else len(fields):>8} {mode:>12} "
                f"{t_min:>8.3f} {t_med:>9.3f} {nbytes / 1e6:>9.1f} "
                f"{leaf_bytes(path, fields) / 1e6:>10.1f}"
            )


def bench_plan(n_fields: int, repeat: int, tmp_dir: Path) -> None:
    schema = {"s": pl.Struct({f"f{i}": pl.Int64 for i in range(n_fields)})}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"plan_{n_fields}.parquet"
    pl.DataFrame(schema=schema).write_parquet(path)

    s = pl.col("s")
    queries = {
        "whole": lambda lf: lf.select(s),
        "3 fields": lambda lf: lf.select(
            s.struct.field("f0", "f17", f"f{n_fields - 1}")
        ),
        "all fields": lambda lf: lf.select(s.struct.unnest()),
        "validity": lambda lf: lf.select(s.is_null()),
        "with_fields": lambda lf: lf.select(
            s.struct.with_fields(x=pl.field("f1") * 2).struct.field("x", "f2")
        ),
    }

    print(f"plan time, struct of {n_fields} fields, repeat: {repeat}")
    print(f"{'query':>12} {'source':>10} {'pushdown s':>11} {'no pushdown s':>14}")
    for source, make in [
        ("parquet", lambda: pl.scan_parquet(path)),
        ("in-memory", lambda: pl.DataFrame(schema=schema).lazy()),
    ]:
        for name, query in queries.items():
            q = query(make())
            row = []
            for opts in [pl.QueryOptFlags(), NO_PUSHDOWN]:
                t_min, _, _ = measure(
                    lambda q=q, opts=opts: q.explain(optimizations=opts), repeat
                )  # type: ignore[misc]
                row.append(t_min)
            print(f"{name:>12} {source:>10} {row[0]:>11.4f} {row[1]:>14.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("dir", type=Path)
    g.add_argument("--rows", type=int, default=2_000_000)
    r = sub.add_parser("read")
    r.add_argument("dir", type=Path)
    r.add_argument("--repeat", type=int, default=5)
    r.add_argument("--engine", default="streaming")
    p = sub.add_parser("plan")
    p.add_argument("--fields", type=int, default=10_000)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--dir", type=Path, default=Path("/tmp/struct_field_pruning"))
    args = parser.parse_args()

    if args.cmd == "generate":
        path = generate(args.dir, args.rows)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    elif args.cmd == "read":
        bench_read(args.dir, args.repeat, args.engine)
    else:
        bench_plan(args.fields, args.repeat, args.dir)


if __name__ == "__main__":
    main()

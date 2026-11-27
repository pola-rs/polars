"""Bounded same-binary nested cost-gate study; run via offload, timings exclusive.

Generate once, then run each thread stratum. Each mode owns a persistent process;
mode order is shuffled per case/round and all timed results are checked against
the same query with predicate pushdown disabled. Startup/checking are untimed.
"""

# This probe deliberately passes heterogeneous JSON records across process pipes.
# ruff: noqa: ANN001, ANN201, ANN202, ANN204

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODES = {
    "root_off": ("root", "off"),
    "leaf_off": ("leaf", "off"),
    "leaf_on": ("leaf", "on"),
    "auto": ("auto", "auto"),
}


def cases():
    out = []

    def add(
        n,
        k,
        pct,
        *,
        variant="core",
        threads=5,
        selection="whole",
        payload="int",
        clustering="random",
        encoding="plain",
        cardinality="high",
        independent=False,
    ):
        case = {
            "n": n,
            "k": k,
            "pct": pct,
            "variant": variant,
            "threads": threads,
            "selection": selection,
            "payload": payload,
            "clustering": clustering,
            "encoding": encoding,
            "cardinality": cardinality,
            "independent": independent,
        }
        case["source_n"] = n * (2 if selection == "pruned" else 1)
        case["id"] = f"c{len(out):03d}"
        case["fixture"] = "-".join(
            str(case[x])
            for x in (
                "source_n",
                "payload",
                "clustering",
                "encoding",
                "cardinality",
                "independent",
            )
        )
        out.append(case)

    for n in (4, 8, 16, 32):
        for k in sorted({1, n // 4, n // 2}):
            for pct in (0.1, 10, 50, 90):
                add(n, k, pct)
    for n in (8, 32):
        for k in (1, n // 2):
            for pct in (0.1, 90):
                add(n, k, pct, variant="wide", payload="mixed128")
                add(n, k, pct, variant="clustered", clustering="clustered")
                add(
                    n,
                    k,
                    pct,
                    variant="dictionary",
                    encoding="dictionary",
                    cardinality="low",
                )
                add(n, k, pct, variant="thread1", threads=1)
                add(n, k, pct, variant="low_plain", cardinality="low")
    for n in (8, 32):
        for k in (1, n // 2):
            for pct in (0.1, 10, 50, 90):
                add(n, k, pct, variant="pruned", selection="pruned")
    for n in (8, 32):
        for k in (n // 4, n // 2):
            for pct in (0.1, 50):
                add(n, k, pct, variant="independent", independent=True)
    assert len(out) == 108, len(out)
    return out


def fixture_path(directory, case):
    return directory / "data" / (case["fixture"] + ".parquet")


def generate(directory, rows, row_group_size, only=None):
    import pyarrow.parquet as pq

    import polars as pl

    (directory / "data").mkdir(parents=True, exist_ok=True)
    fixtures = {}
    for case in cases():
        if only and case["id"] not in only:
            continue
        if case["fixture"] in fixtures:
            continue
        path = fixture_path(directory, case)
        domain = 1024 if case["cardinality"] == "low" else rows
        i = pl.int_range(rows, dtype=pl.Int64)
        rank = i if case["clustering"] == "clustered" else (i * 2654435761) % rows
        fields = []
        for j in range(case["source_n"]):
            r = (i.hash(j + 17) % domain) if case["independent"] else rank % domain
            value = r + j * rows
            if case["payload"] == "mixed128" and j >= case["source_n"] // 2:
                # Hash bits avoid a constant, trivially compressed 128-byte tail.
                h = i.hash(j).cast(pl.String).str.pad_start(20, "0")
                value = pl.concat_str([h] * 7).str.slice(0, 128)
            fields.append(value.alias(f"v{j}"))
        s = pl.struct(fields)
        for _ in range(3):
            s = pl.struct(s.alias("a"))
        table = pl.select(s=s).to_arrow()
        pq.write_table(
            table,
            path,
            row_group_size=row_group_size,
            compression="zstd",
            use_dictionary=case["encoding"] == "dictionary",
            write_statistics=True,
        )
        del table
        meta = pq.ParquetFile(path).metadata
        enc = sorted(
            {
                e
                for rg in range(meta.num_row_groups)
                for col in range(meta.num_columns)
                for e in meta.row_group(rg).column(col).encodings
            }
        )
        assert ("RLE_DICTIONARY" in enc) == (case["encoding"] == "dictionary"), enc
        fixtures[case["fixture"]] = {
            "bytes": path.stat().st_size,
            "encodings": enc,
            "row_groups": meta.num_row_groups,
            "physical_columns": meta.num_columns,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        }
        assert sum(x["bytes"] for x in fixtures.values()) < 3_000_000_000
        print(
            json.dumps({"generated": case["fixture"], **fixtures[case["fixture"]]}),
            flush=True,
        )
    (directory / "fixtures.json").write_text(
        json.dumps(
            {"rows": rows, "row_group_size": row_group_size, "fixtures": fixtures},
            indent=2,
        )
        + "\n"
    )
    (directory / "cases.json").write_text(json.dumps(cases(), indent=2) + "\n")


def query(directory, case, rows):
    import polars as pl

    leaf = pl.col("s").struct.field("a").struct.field("a").struct.field("a")
    domain = 1024 if case["cardinality"] == "low" else rows
    fraction = case["pct"] / 100
    if case["independent"]:
        fraction **= 1 / case["k"]
    threshold = math.floor(domain * fraction)
    predicate = pl.all_horizontal(
        [leaf.struct.field(f"v{j}") < threshold + j * rows for j in range(case["k"])]
    )
    q = pl.scan_parquet(
        fixture_path(directory, case), use_statistics=False, parallel="auto"
    ).filter(predicate)
    if case["selection"] == "whole":
        q = q.select("s")
    else:
        q = q.select([leaf.struct.field(f"v{j}") for j in range(case["n"])])
    return q


def worker(directory, rows):
    import polars as pl

    q = expected = None
    for line in sys.stdin:
        command = json.loads(line)
        if command["op"] == "release":
            q = expected = None
            response = {"released": True}
        elif command["op"] == "prepare":
            q = expected = None
            if os.environ.get("POLARS_NESTED_PROBE_DIAGNOSTICS") == "1":
                print(
                    "NESTED_PROBE_CASE " + json.dumps(command["case"]),
                    file=sys.stderr,
                    flush=True,
                )
            q = query(directory, command["case"], rows)
            expected = q.collect(
                engine="streaming",
                optimizations=pl.QueryOptFlags(
                    predicate_pushdown=False, projection_pushdown=False
                ),
            )
            actual = q.collect(engine="streaming")
            assert actual.equals(expected), command["case"]
            del actual
            response = {
                "kept": expected.height,
                "threads": pl.thread_pool_size(),
                "explain": q.explain(),
                "correctness": True,
            }
        else:
            start = time.perf_counter_ns()
            actual = q.collect(engine="streaming")
            elapsed = (time.perf_counter_ns() - start) / 1e9
            assert actual.equals(expected), command
            response = {"seconds": elapsed, "kept": actual.height, "correctness": True}
            del actual
        print(json.dumps(response), flush=True)


class Worker:
    """One fixed runtime strategy and thread pool, driven through JSON pipes."""

    def __init__(self, directory, rows, threads, name, label, diagnostics=False):
        mode, mask = MODES[name]
        env = dict(
            os.environ,
            POLARS_MAX_THREADS=str(threads),
            POLARS_NESTED_PROBE_MODE=mode,
            POLARS_NESTED_PROBE_MASK=mask,
            POLARS_NESTED_PROBE_DIAGNOSTICS=str(int(diagnostics)),
        )
        env.pop("POLARS_NESTED_PROBE_SKIP_ROOT", None)
        self.stderr = (
            directory / f"{label}-worker-{name}-t{threads}-diag{int(diagnostics)}.log"
        ).open("w")
        self.process = subprocess.Popen(
            [sys.executable, __file__, str(directory), "--worker", "--rows", str(rows)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )

    def call(self, command):
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            message = f"worker exited {self.process.poll()}; see worker stderr"
            raise RuntimeError(message)
        return json.loads(line)

    def close(self, *, check=True):
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.stderr.close()
        if check:
            assert self.process.returncode == 0, self.process.returncode


def run(directory, rows, repeat, label, only, diagnostics, seed):
    chosen = [c for c in cases() if not only or c["id"] in only]
    with Path(__file__).open("rb") as source:
        driver_hash = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "machine": platform.node(),
        "platform": platform.platform(),
        "rows": rows,
        "warmup": 1,
        "repeat": repeat,
        "statistics": False,
        "label": label,
        "command": sys.argv,
        "modes": MODES,
        "seed": seed,
        "binary_sha256": os.environ.get("NESTED_PROBE_BINARY_SHA256"),
        "revision": os.environ.get("NESTED_PROBE_REVISION"),
        "driver_sha256": driver_hash,
        "python": sys.executable,
    }
    assert json.loads((directory / "fixtures.json").read_text())["rows"] == rows
    (directory / f"{label}-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    rng = random.Random(seed)
    with (directory / f"{label}.jsonl").open("w") as log:
        for threads in sorted({c["threads"] for c in chosen}):
            workers = {
                m: Worker(directory, rows, threads, m, label, diagnostics)
                for m in MODES
            }
            try:
                for case in [c for c in chosen if c["threads"] == threads]:
                    order = list(workers)
                    rng.shuffle(order)
                    preparations = {
                        m: workers[m].call({"op": "prepare", "case": case})
                        for m in order
                    }
                    assert len({x["kept"] for x in preparations.values()}) == 1
                    samples = {m: [] for m in workers}
                    orders = []
                    for _ in range(repeat):
                        rng.shuffle(order)
                        orders.append(order.copy())
                        for m in order:
                            samples[m].append(workers[m].call({"op": "run"})["seconds"])
                    record = {
                        "case": case,
                        "kept": next(iter(preparations.values()))["kept"],
                        "actual_retention": next(iter(preparations.values()))["kept"]
                        / rows,
                        "samples": samples,
                        "median": {
                            m: statistics.median(t) if t else None
                            for m, t in samples.items()
                        },
                        "orders": orders,
                        "preparations": preparations,
                    }
                    line = json.dumps(record)
                    log.write(line + "\n")
                    log.flush()
                    print(
                        json.dumps(
                            {k: v for k, v in record.items() if k != "preparations"}
                        ),
                        flush=True,
                    )
                    for w in workers.values():
                        w.call({"op": "release"})
            finally:
                failed = sys.exc_info()[0] is not None
                for w in workers.values():
                    w.close(check=not failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rows", type=int, default=524288)
    parser.add_argument("--row-group-size", type=int, default=131072)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--label", default="cost-gates")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.directory, args.rows)
    elif args.generate:
        generate(args.directory, args.rows, args.row_group_size, args.only)
    else:
        run(
            args.directory,
            args.rows,
            args.repeat,
            args.label,
            args.only,
            args.diagnostics,
            args.seed,
        )

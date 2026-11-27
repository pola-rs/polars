"""Decoder-only root rematerialization counterfactual; offload timings exclusive.

Generate once, then run each thread stratum. Each mode owns a persistent process;
mode order is shuffled per case/round and all timed results are checked against
the same query with predicate pushdown disabled. Startup/checking are untimed.
"""

# This probe passes heterogeneous JSON records through subprocess pipes.
# ruff: noqa: ANN001, ANN201, ANN204

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
    "leaf_auto": ("leaf", "auto"),
    "leaf_skip": ("leaf", "auto"),
    "auto": ("auto", "auto"),
}


def cases():
    out = []
    for threads in (1, 5):
        for depth in (1, 4):
            for payload in ("int", "string"):
                for k in (1, 2):
                    for pct in (1, 50, 90):
                        out.append(
                            {
                                "id": f"r{len(out):03d}",
                                "threads": threads,
                                "depth": depth,
                                "payload": payload,
                                "k": k,
                                "pct": pct,
                                "source_n": 8,
                                "fixture": f"d{depth}-{payload}",
                            }
                        )
    assert len(out) == 48
    return out


def fixture_path(directory, case):
    return directory / "data" / (case["fixture"] + ".parquet")


def generate(directory, rows, row_group_size):
    import pyarrow.parquet as pq

    import polars as pl

    (directory / "data").mkdir(parents=True, exist_ok=True)
    fixtures = {}
    for case in cases():
        if case["fixture"] in fixtures:
            continue
        path = fixture_path(directory, case)
        i = pl.int_range(rows, dtype=pl.Int64)
        rank = (i * 2654435761) % rows
        fields = []
        for j in range(8):
            value = pl.when(i % (1009 + j) == 0).then(None).otherwise(rank + j * rows)
            if case["payload"] == "string":
                value = value.cast(pl.String).str.pad_start(16, "0")
            fields.append(value.alias(f"v{j}"))
        s = pl.struct(fields)
        for level in range(case["depth"]):
            s = pl.when(i % (997 + level * 2) == 0).then(None).otherwise(s)
            if level + 1 < case["depth"]:
                s = pl.struct(s.alias("a"))
        table = pl.select(s=s, other=i).to_arrow()
        pq.write_table(
            table,
            path,
            row_group_size=row_group_size,
            compression="zstd",
            use_dictionary=False,
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
        fixtures[case["fixture"]] = {
            "bytes": path.stat().st_size,
            "encodings": enc,
            "row_groups": meta.num_row_groups,
            "physical_columns": meta.num_columns,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        }
        assert meta.num_row_groups == 4
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

    leaf = pl.col("s")
    for _ in range(case["depth"] - 1):
        leaf = leaf.struct.field("a")
    threshold = math.floor(rows * case["pct"] / 100)
    comparisons = []
    for j in range(case["k"]):
        value = threshold + j * rows
        if case["payload"] == "string":
            value = str(value).zfill(16)
        comparisons.append(leaf.struct.field(f"v{j}") < value)
    q = pl.scan_parquet(
        fixture_path(directory, case), use_statistics=False, parallel="auto"
    ).filter(pl.all_horizontal(comparisons))
    if case.get("guard") == "residual":
        q = q.filter(
            (leaf.struct.field("v0").cast(pl.Int64) + pl.col("other")) % 3 == 0
        )
    elif case.get("guard") == "whole":
        q = q.filter(pl.col("s").is_not_null())
    q = q.select("other")
    if case.get("guard") == "dynamic_join":
        q = q.join(pl.LazyFrame({"other": list(range(1000))}), on="other")
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
            q = query(directory, command["case"], rows)
            plan = q.explain()
            scan = plan.find("Parquet SCAN")
            assert scan >= 0, plan
            above_scan = plan[:scan]
            assert "FILTER" not in above_scan, plan
            assert 'col("s")' not in above_scan, plan
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

    def __init__(self, directory, rows, threads, name, diagnostics=False):
        mode, mask = MODES[name]
        env = dict(
            os.environ,
            POLARS_MAX_THREADS=str(threads),
            POLARS_NESTED_PROBE_MODE=mode,
            POLARS_NESTED_PROBE_MASK=mask,
            POLARS_NESTED_PROBE_DIAGNOSTICS=str(int(diagnostics)),
        )
        env.pop("POLARS_NESTED_PROBE_SKIP_ROOT", None)
        if name == "leaf_skip":
            env["POLARS_NESTED_PROBE_SKIP_ROOT"] = "s"
        self.stderr = (
            directory / f"worker-{name}-t{threads}-diag{int(diagnostics)}.log"
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

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.stderr.close()
        assert self.process.returncode == 0, self.process.returncode


def run(directory, rows, repeat, label, only, diagnostics, seed):
    chosen = [c for c in cases() if not only or c["id"] in only]
    with Path(__file__).open("rb") as driver_file:
        driver_hash = hashlib.file_digest(driver_file, "sha256").hexdigest()
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
    }
    assert json.loads((directory / "fixtures.json").read_text())["rows"] == rows
    (directory / f"{label}-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    rng = random.Random(seed)
    with (directory / f"{label}.jsonl").open("w") as log:
        for threads in sorted({c["threads"] for c in chosen}):
            workers = {
                m: Worker(directory, rows, threads, m, diagnostics) for m in MODES
            }
            try:
                stratum_cases = [c for c in chosen if c["threads"] == threads]
                rng.shuffle(stratum_cases)
                for case in stratum_cases:
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
                            m: (statistics.median(t) if t else None)
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
                for w in workers.values():
                    w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rows", type=int, default=524288)
    parser.add_argument("--row-group-size", type=int, default=131072)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--label", default="root-redecode")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--guards", action="store_true")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.worker:
        worker(args.directory, args.rows)
    elif args.guards:
        import polars as pl

        for guard in ("residual", "whole", "dynamic_join"):
            case = dict(cases()[18], guard=guard)
            q = query(args.directory, case, args.rows)
            expected = q.collect(
                engine="streaming",
                optimizations=pl.QueryOptFlags(
                    predicate_pushdown=False, projection_pushdown=False
                ),
            )
            actual = q.collect(engine="streaming")
            matches = actual.sort("other").equals(expected.sort("other"))
            assert matches == (guard != "residual"), (guard, matches)
            print(
                json.dumps(
                    {
                        "guard": guard,
                        "kept": actual.height,
                        "expected_kept": expected.height,
                        "correctness": matches,
                        "expected_negative": (guard == "residual"),
                        "explain": q.explain(),
                    }
                ),
                flush=True,
            )
    elif args.generate:
        generate(args.directory, args.rows, args.row_group_size)
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

"""Repeatable warm-cache benchmark for struct-leaf parquet prefiltering.

POLARS_MAX_THREADS=5 python nested_predicate_pushdown.py DIR --label baseline
Run each revision with the same profile and machine, under offload --exclusive.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import polars as pl


def run(
    directory: Path,
    label: str,
    rows: int,
    repeat: int,
    *,
    widths: list[int] | None = None,
    percentages: list[float] | None = None,
    selections: list[str] | None = None,
    include_lists: bool = True,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{label}.jsonl"
    with output.open("w") as log:
        log.write(
            json.dumps(
                {
                    "machine": platform.node(),
                    "platform": platform.platform(),
                    "polars": pl.__version__,
                    "threads": pl.thread_pool_size(),
                    "rows": rows,
                    "repeat": repeat,
                    "warmup": 1,
                    "statistics": False,
                    "label": label,
                }
            )
            + "\n"
        )
        for lists in (False, True) if include_lists else (False,):
            for width in widths or ((2, 50) if lists else (2, 10, 50)):
                path = directory / f"w{width}-list{int(lists)}.parquet"
                if not path.exists():
                    i = pl.int_range(rows, dtype=pl.Int64)
                    # Hash-distributed predicates prevent row-group skipping. Nulls at
                    # several ancestor levels exercise definition-level handling.
                    fields = [
                        (i.hash(k) % 100_000).alias(f"v{k}") for k in range(width)
                    ]
                    if lists:
                        fields.append(pl.concat_list(i, i + 1).alias("list"))
                    s = pl.struct(fields)
                    for depth in range(3):
                        s = pl.struct(
                            pl.when(i % (31 + depth) == 0)
                            .then(None)
                            .otherwise(s)
                            .alias("a")
                        )
                    pl.select(s=s, other=i).write_parquet(path, row_group_size=250_000)
                leaf = pl.col("s").struct.field("a").struct.field("a").struct.field("a")
                for pct in percentages or ((1, 50) if lists else (0.1, 1, 10, 50, 100)):
                    for selection in selections or (
                        ("sibling", "whole") if lists else ("sibling", "other", "whole")
                    ):
                        q = pl.scan_parquet(path, use_statistics=False).filter(
                            leaf.struct.field("v0") < int(pct * 1000)
                        )
                        q = q.select(
                            {
                                "sibling": leaf.struct.field("v1"),
                                "other": pl.col("other"),
                                "whole": pl.col("s"),
                            }[selection]
                        )
                        expected = q.collect(engine="streaming")
                        times = []
                        for _ in range(repeat):
                            start = time.perf_counter()
                            result = q.collect(engine="streaming")
                            times.append(time.perf_counter() - start)
                            assert result.equals(expected)
                        record = {
                            "width": width,
                            "list": lists,
                            "percent": pct,
                            "selection": selection,
                            "kept": expected.height,
                            "seconds": times,
                            "median": statistics.median(times),
                        }
                        line = json.dumps(record)
                        log.write(line + "\n")
                        log.flush()
                        print(line, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--widths", nargs="+", type=int)
    parser.add_argument("--percentages", nargs="+", type=float)
    parser.add_argument(
        "--selections", nargs="+", choices=["sibling", "other", "whole"]
    )
    parser.add_argument("--no-lists", action="store_true")
    args = parser.parse_args()
    run(
        args.directory,
        args.label,
        args.rows,
        args.repeat,
        widths=args.widths,
        percentages=args.percentages,
        selections=args.selections,
        include_lists=not args.no_lists,
    )

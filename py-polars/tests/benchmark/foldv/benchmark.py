"""Compare complete Polars queries, including JIT compilation on every call."""

from __future__ import annotations

import argparse
import statistics
from time import perf_counter
from typing import TYPE_CHECKING

import battery
import inventory
import trading

from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from types import ModuleType

    import polars as pl


def query(example: ModuleType, df: pl.DataFrame, method: str) -> pl.DataFrame:
    return df.lazy().select(getattr(example, method)()).collect(engine="in-memory")


def benchmark(
    example: ModuleType, rows: int, repeat: int, variant: str = ""
) -> tuple[float, float]:
    print(f"  generating {row_label(rows)} rows", flush=True)
    df = example.generate(rows)
    # Correctness and initial setup are outside timing. JIT compilation is repeated
    # during every timed query; this warm-up does not create a compiled-kernel cache.
    print("  checking that foldv and Polars give the same result", flush=True)
    assert_frame_equal(
        query(example, df, "foldv" + variant),
        query(example, df, "best_effort" + variant),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    timings: dict[str, list[float]] = {"foldv": [], "best_effort": []}
    for iteration in range(repeat):
        print(f"  timing run {iteration + 1}/{repeat}", flush=True)
        methods = list(timings)
        if iteration % 2:
            methods.reverse()
        for method in methods:
            start = perf_counter()
            result = query(example, df, method + variant)
            elapsed = perf_counter() - start
            assert result.height == rows
            timings[method].append(elapsed)

    return statistics.median(timings["foldv"]), statistics.median(
        timings["best_effort"]
    )


def row_label(rows: int) -> str:
    return f"{rows // 1_000_000}M" if rows and rows % 1_000_000 == 0 else f"{rows:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5_000_000)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if args.rows < 0 or args.repeat < 1:
        parser.error("rows must be nonnegative and repeat must be positive")

    rows = row_label(args.rows)
    cases = (
        ("Trading", trading, ""),
        ("Inventory allocation", inventory, ""),
        ("Battery sizing", battery, "_sizing"),
    )
    summaries = []
    for name, example, variant in cases:
        print(f"{name}:", flush=True)
        native, baseline = benchmark(example, args.rows, args.repeat, variant)
        summaries.append((name, baseline / native, native * 1000, baseline * 1000))

    print(
        f"\n{rows} rows, in-memory engine, median of {args.repeat} runs, "
        "including JIT compilation\n"
    )
    print(f"{'Use case':<23} {'Speedup':>8} {'foldv':>12} {'Polars':>12}")
    for name, speedup, native_ms, baseline_ms in summaries:
        print(
            f"{name:<23} {speedup:>7.1f}x {native_ms:>9.2f} ms {baseline_ms:>9.2f} ms"
        )
    print("\nBaseline: Polars batch UDFs.")


if __name__ == "__main__":
    main()

"""Reproducible release-build benchmark for Polars' native UUID data type."""

from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from io import BytesIO
from typing import Any

import polars as pl

N = 1_000_000
REPEATS = 7


def measure(function: Callable[[], Any], *, repeats: int = REPEATS) -> dict[str, float]:
    function()
    function()
    timings: list[float] = []
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = function()
            timings.append((time.perf_counter_ns() - start) / 1_000_000)
            del result
    finally:
        gc.enable()
    return {
        "median_ms": round(statistics.median(timings), 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
    }


native = pl.uuid4(N, eager=True).rename("id")
strings = native.cast(pl.String)
needle_native = native[N // 2]
needle_string = strings[N // 2]

base = pl.uuid4(N // 10, eager=True).rename("id")
native_groups = pl.concat([base] * 10).sample(fraction=1.0, shuffle=True, seed=42)
string_groups = native_groups.cast(pl.String)


def parquet_bytes(series: pl.Series) -> bytes:
    output = BytesIO()
    series.to_frame().write_parquet(output, compression="uncompressed")
    return output.getvalue()


operations: dict[str, Callable[[], Any]] = {
    "sort/native": native.sort,
    "sort/string": strings.sort,
    "n_unique/native": native.n_unique,
    "n_unique/string": strings.n_unique,
    "filter/native": lambda: native.filter(native == needle_native),
    "filter/string": lambda: strings.filter(strings == needle_string),
    "group_by/native": lambda: native_groups.to_frame().group_by("id").len(),
    "group_by/string": lambda: string_groups.to_frame().group_by("id").len(),
    "parse/string_to_uuid": lambda: strings.cast(pl.UUID),
    "format/uuid_to_string": lambda: native.cast(pl.String),
    "generate/v4": lambda: pl.uuid4(N, eager=True),
    "generate/v7": lambda: pl.uuid7(N, eager=True),
    "parquet_write/native": lambda: parquet_bytes(native),
    "parquet_write/string": lambda: parquet_bytes(strings),
}

results = {name: measure(function) for name, function in operations.items()}
native_parquet = parquet_bytes(native)
string_parquet = parquet_bytes(strings)

report = {
    "environment": {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "polars": pl.__version__,
        "rows": N,
        "repeats": REPEATS,
        "rust_profile": os.environ.get(
            "POLARS_NATIVE_UUID_BENCHMARK_PROFILE", "unspecified"
        ),
    },
    "storage_bytes": {
        "memory/native": native.estimated_size(),
        "memory/string": strings.estimated_size(),
        "parquet_uncompressed/native": len(native_parquet),
        "parquet_uncompressed/string": len(string_parquet),
    },
    "timings": results,
}

print(json.dumps(report, indent=2, sort_keys=True))

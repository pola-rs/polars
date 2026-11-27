from __future__ import annotations

import os
import subprocess
import sys


def test_timeout_prints_traceback_and_exits() -> None:
    script = """\
import time
import polars as pl

def slow(s):
    time.sleep(30)
    return s

def run_stuck_query():
    pl.LazyFrame({"a": [1]}).select(pl.col("a").map_batches(slow)).collect()

run_stuck_query()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "POLARS_TIMEOUT_MS": "200"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "POLARS_TIMEOUT_MS exceeded" in result.stderr
    assert "in run_stuck_query" in result.stderr

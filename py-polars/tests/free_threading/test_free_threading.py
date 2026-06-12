from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest


def is_free_threaded_python() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


pytestmark = pytest.mark.skipif(
    not is_free_threaded_python(),
    reason="requires free-threaded CPython",
)


def assert_gil_disabled() -> None:
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    assert callable(is_gil_enabled)
    assert not is_gil_enabled()


def test_import_does_not_enable_gil() -> None:
    code = """
    import sys

    import polars as pl

    assert not sys._is_gil_enabled()
    assert pl.DataFrame({"x": [1, 2, 3]}).select(pl.col("x").sum()).item() == 6
    """
    env = os.environ.copy()
    env.pop("PYTHON_GIL", None)
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True, env=env)


def test_independent_dataframes_from_multiple_threads() -> None:
    import polars as pl

    assert_gil_disabled()

    def worker(seed: int) -> tuple[int, int]:
        df = pl.DataFrame(
            {
                "g": [seed % 4, seed % 4, (seed + 1) % 4, (seed + 1) % 4],
                "x": [seed, seed + 1, seed + 2, seed + 3],
            }
        )
        out = (
            df.with_columns((pl.col("x") * 2).alias("y"))
            .group_by("g")
            .agg(pl.col("y").sum().alias("sum_y"))
            .sort("g")
        )
        return out.height, int(out["sum_y"].sum())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(64)))

    assert all(height == 2 for height, _ in results)
    assert sum(total for _, total in results) == sum(
        8 * seed + 12 for seed in range(64)
    )


def test_independent_lazyframes_from_multiple_threads() -> None:
    import polars as pl

    assert_gil_disabled()

    def worker(seed: int) -> int:
        left = pl.DataFrame(
            {"k": [1, 2, 3, 4], "x": [seed + i for i in range(4)]}
        ).lazy()
        right = pl.DataFrame({"k": [1, 2, 3, 4], "z": [10, 20, 30, 40]}).lazy()
        out = (
            left.join(right, on="k")
            .filter(pl.col("x") >= seed + 1)
            .with_columns((pl.col("x") + pl.col("z")).alias("v"))
            .select(pl.col("v").sum())
            .collect()
        )
        return int(out.item())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(64)))

    assert results == [3 * seed + 96 for seed in range(64)]


def test_concat_df_from_multiple_threads() -> None:
    import polars as pl

    assert_gil_disabled()

    def worker(seed: int) -> int:
        frames = [pl.DataFrame({"x": [seed + i]}) for i in range(16)]
        return int(pl.concat(frames)["x"].sum())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(64)))

    assert results == [16 * seed + 120 for seed in range(64)]

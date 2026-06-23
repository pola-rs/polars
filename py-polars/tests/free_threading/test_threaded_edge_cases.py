from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.free_threading._free_threading import assert_gil_disabled


def test_shared_read_only_dataframe_from_multiple_threads() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "g": [idx % 4 for idx in range(128)],
            "x": list(range(128)),
        }
    )

    def worker(group: int) -> int:
        assert_gil_disabled()
        out = df.filter(pl.col("g") == group).select(pl.col("x").sum())
        return int(out.item())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, [idx % 4 for idx in range(64)]))

    expected_by_group = [
        sum(idx for idx in range(128) if idx % 4 == group) for group in range(4)
    ]
    assert results == [expected_by_group[idx % 4] for idx in range(64)]


def test_schema_inference_from_multiple_threads() -> None:
    import polars as pl

    def worker(seed: int) -> tuple[str, int | None]:
        assert_gil_disabled()
        df = pl.DataFrame(
            [
                {"name": f"v{seed}", "value": seed},
                {"name": f"v{seed + 1}", "value": None},
            ]
        )
        return str(df.schema["value"]), df["value"].null_count()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(64)))

    assert results == [("Int64", 1)] * 64


def test_concurrent_exception_propagation() -> None:
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3]})

    def worker(_: int) -> str:
        assert_gil_disabled()
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.select(pl.col("missing"))
        return "raised"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(64)))

    assert results == ["raised"] * 64

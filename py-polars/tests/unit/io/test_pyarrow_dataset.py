from __future__ import annotations

import os

import pyarrow.dataset as ds

import polars as pl


def test_dataset(io_test_dir: str) -> None:
    # windows path does not seem to work
    if os.name != "nt":
        file = os.path.join(io_test_dir, "small.ipc")
        dset = ds.dataset(file, format="ipc")

        expected = (
            pl.scan_ipc(file)
            .filter("bools")
            .select(["bools", "floats", "date"])
            .collect()
        )
        out = (
            pl.scan_ds(dset)
            .filter("bools")
            .select(["bools", "floats", "date"])
            .collect()
        )

        assert out.frame_equal(expected)

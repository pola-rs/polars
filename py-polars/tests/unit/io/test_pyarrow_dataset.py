from __future__ import annotations

import os
import typing

import pyarrow.dataset as ds

import polars as pl


@typing.no_type_check
def helper_dataset_test(io_test_dir: str, query) -> None:
    file = os.path.join(io_test_dir, "small.ipc")
    dset = ds.dataset(file, format="ipc")

    expected = query(pl.scan_ipc(file))
    out = query(pl.scan_ds(dset))
    assert out.frame_equal(expected)


def test_dataset(io_test_dir: str) -> None:
    # windows path does not seem to work
    if os.name != "nt":
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter("bools").select(["bools", "floats", "date"]).collect(),
        )
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter(~pl.col("bools"))
            .select(["bools", "floats", "date"])
            .collect(),
        )
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter(pl.col("int_nulls").is_null())
            .select(["bools", "floats", "date"])
            .collect(),
        )
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter(pl.col("int_nulls").is_not_null())
            .select(["bools", "floats", "date"])
            .collect(),
        )
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter(pl.col("int_nulls").is_not_null() == pl.col("bools"))
            .select(["bools", "floats", "date"])
            .collect(),
        )
        # this equality on a column with nulls fails as pyarrow has different
        # handling kleene logic. We leave it for now and document it in the function.
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter(pl.col("int") == 10)
            .select(["bools", "floats", "int_nulls"])
            .collect(),
        )
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter(pl.col("int") != 10)
            .select(["bools", "floats", "int_nulls"])
            .collect(),
        )
        # this predicate is not supported by pyarrow
        # check if we still do it on our side
        helper_dataset_test(
            io_test_dir,
            lambda lf: lf.filter(pl.col("floats").sum().over("date") == 10)
            .select(["bools", "floats", "date"])
            .collect(),
        )

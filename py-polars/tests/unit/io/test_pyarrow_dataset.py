from __future__ import annotations

import sys
import tempfile
import typing
from datetime import date, datetime, time
from pathlib import Path

import pyarrow.dataset as ds
import pytest

import polars as pl
from polars.testing import assert_frame_equal


@typing.no_type_check
def helper_dataset_test(file_path: Path, query) -> None:
    dset = ds.dataset(file_path, format="ipc")

    expected = query(pl.scan_ipc(file_path))
    out = query(pl.scan_pyarrow_dataset(dset))
    assert_frame_equal(out, expected)


@pytest.mark.write_disk()
@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_dataset(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.ipc"
        df.write_ipc(file_path)

        helper_dataset_test(
            file_path,
            lambda lf: lf.filter("bools").select(["bools", "floats", "date"]).collect(),
        )
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(~pl.col("bools"))
            .select(["bools", "floats", "date"])
            .collect(),
        )
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("int_nulls").is_null())
            .select(["bools", "floats", "date"])
            .collect(),
        )
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("int_nulls").is_not_null())
            .select(["bools", "floats", "date"])
            .collect(),
        )
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("int_nulls").is_not_null() == pl.col("bools"))
            .select(["bools", "floats", "date"])
            .collect(),
        )
        # this equality on a column with nulls fails as pyarrow has different
        # handling kleene logic. We leave it for now and document it in the function.
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("int") == 10)
            .select(["bools", "floats", "int_nulls"])
            .collect(),
        )
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("int") != 10)
            .select(["bools", "floats", "int_nulls"])
            .collect(),
        )
        # this predicate is not supported by pyarrow
        # check if we still do it on our side
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("floats").sum().over("date") == 10)
            .select(["bools", "floats", "date"])
            .collect(),
        )

        # temporal types
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("date") < date(1972, 1, 1))
            .select(["bools", "floats", "date"])
            .collect(),
        )
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("datetime") > datetime(1970, 1, 1, second=13))
            .select(["bools", "floats", "date"])
            .collect(),
        )
        # not yet supported in pyarrow
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("time") >= time(microsecond=100))
            .select(["bools", "time", "date"])
            .collect(),
        )

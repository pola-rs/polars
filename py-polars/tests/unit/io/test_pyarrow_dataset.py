from __future__ import annotations

import typing
from datetime import date, datetime, time
from typing import TYPE_CHECKING

import pyarrow.dataset as ds
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@typing.no_type_check
def helper_dataset_test(file_path: Path, query) -> None:
    dset = ds.dataset(file_path, format="ipc")

    expected = query(pl.scan_ipc(file_path))
    out = query(pl.scan_pyarrow_dataset(dset))
    assert_frame_equal(out, expected)


@pytest.mark.write_disk()
def test_dataset(df: pl.DataFrame, tmp_path: Path) -> None:
    file_path = tmp_path / "small.ipc"
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

    # pushdown is_in
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int").is_in([1, 3, 20]))
        .select(["bools", "floats", "date"])
        .collect(),
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int").is_in(list(range(120))))
        .select(["bools", "floats", "date"])
        .collect(),
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("cat").is_in([]))
        .select(["bools", "floats", "date"])
        .collect(),
    )
    # direct filter
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.Series([True, False, True]))
        .select(["bools", "floats", "date"])
        .collect(),
    )

from __future__ import annotations

from datetime import date, datetime, time
from typing import TYPE_CHECKING, Callable

import pyarrow.dataset as ds
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


def helper_dataset_test(
    file_path: Path,
    query: Callable[[pl.LazyFrame], pl.DataFrame],
    batch_size: int | None = None,
    n_expected: int | None = None,
) -> None:
    dset = ds.dataset(file_path, format="ipc")
    expected = query(pl.scan_ipc(file_path))
    out = query(
        pl.scan_pyarrow_dataset(dset, batch_size=batch_size),
    )
    assert_frame_equal(out, expected)
    if n_expected is not None:
        assert len(out) == n_expected


@pytest.mark.write_disk()
def test_dataset_foo(df: pl.DataFrame, tmp_path: Path) -> None:
    file_path = tmp_path / "small.ipc"
    df.write_ipc(file_path)

    helper_dataset_test(
        file_path,
        lambda lf: lf.filter("bools").select(["bools", "floats", "date"]).collect(),
        n_expected=1,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(~pl.col("bools"))
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=2,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int_nulls").is_null())
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=1,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int_nulls").is_not_null())
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=2,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int_nulls").is_not_null() == pl.col("bools"))
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=0,
    )
    # this equality on a column with nulls fails as pyarrow has different
    # handling kleene logic. We leave it for now and document it in the function.
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int") == 10)
        .select(["bools", "floats", "int_nulls"])
        .collect(),
        n_expected=0,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int") != 10)
        .select(["bools", "floats", "int_nulls"])
        .collect(),
        n_expected=3,
    )
    # this predicate is not supported by pyarrow
    # check if we still do it on our side
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("floats").sum().over("date") == 10)
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=0,
    )
    # temporal types
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("date") < date(1972, 1, 1))
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=1,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("datetime") > datetime(1970, 1, 1, second=13))
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=1,
    )
    # not yet supported in pyarrow
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("time") >= time(microsecond=100))
        .select(["bools", "time", "date"])
        .collect(),
        n_expected=3,
    )
    # pushdown is_in
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int").is_in([1, 3, 20]))
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=2,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(
            pl.col("date").is_in([date(1973, 8, 17), date(1973, 5, 19)])
        )
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=2,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(
            pl.col("datetime").is_in(
                [
                    datetime(1970, 1, 1, 0, 0, 12, 341234),
                    datetime(1970, 1, 1, 0, 0, 13, 241324),
                ]
            )
        )
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=2,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int").is_in(list(range(120))))
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=3,
    )
    # TODO: remove string cache
    with pl.StringCache():
        helper_dataset_test(
            file_path,
            lambda lf: lf.filter(pl.col("cat").is_in([]))
            .select(["bools", "floats", "date"])
            .collect(),
            n_expected=0,
        )
        helper_dataset_test(
            file_path,
            lambda lf: lf.collect(),
            batch_size=2,
            n_expected=3,
        )

    # direct filter
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.Series([True, False, True]))
        .select(["bools", "floats", "date"])
        .collect(),
        n_expected=2,
    )


def test_pyarrow_dataset_comm_subplan_elim(tmp_path: Path) -> None:
    df0 = pl.DataFrame({"a": [1, 2, 3]})

    df1 = pl.DataFrame({"a": [1, 2]})

    file_path_0 = tmp_path / "0.parquet"
    file_path_1 = tmp_path / "1.parquet"

    df0.write_parquet(file_path_0)
    df1.write_parquet(file_path_1)

    ds0 = ds.dataset(file_path_0, format="parquet")
    ds1 = ds.dataset(file_path_1, format="parquet")

    lf0 = pl.scan_pyarrow_dataset(ds0)
    lf1 = pl.scan_pyarrow_dataset(ds1)

    assert lf0.join(lf1, on="a", how="inner").collect().to_dict(as_series=False) == {
        "a": [1, 2]
    }

import sys
import typing
from pathlib import Path

import pyarrow.dataset as ds
import pytest

import polars as pl
from polars.testing import assert_frame_equal


@typing.no_type_check
@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_struct_pyarrow_dataset_5796() -> None:
    num_rows = 2**17 + 1

    df = pl.from_records(
        [
            dict(  # noqa: C408
                id=i,
                nested=dict(  # noqa: C408
                    a=i,
                ),
            )
            for i in range(num_rows)
        ]
    )

    df.write_parquet("/tmp/out.parquet", use_pyarrow=True)
    tbl = ds.dataset("/tmp/out.parquet").to_table()
    result = pl.from_arrow(tbl)
    assert_frame_equal(result, df)


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_sink_parquet_ipc(io_files_path: Path) -> None:
    file = io_files_path / "small.parquet"

    dst = "/tmp/test_sink.parquet"
    pl.scan_parquet(file).sink_parquet(dst)
    with pl.StringCache():
        assert pl.read_parquet(dst).frame_equal(pl.read_parquet(file))

    dst = "/tmp/test_sink.ipc"
    pl.scan_parquet(file).sink_ipc(dst)
    with pl.StringCache():
        assert pl.read_ipc(dst).frame_equal(pl.read_parquet(file))


@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_fetch_union() -> None:
    pl.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]}).write_parquet(
        "/tmp/df_fetch_1.parquet"
    )
    pl.DataFrame({"a": [3, 4, 5], "b": [4, 5, 6]}).write_parquet(
        "/tmp/df_fetch_2.parquet"
    )

    assert pl.scan_parquet("/tmp/df_fetch_1.parquet").fetch(1).to_dict(False) == {
        "a": [0],
        "b": [1],
    }
    assert pl.scan_parquet("/tmp/df_fetch_*.parquet").fetch(1).to_dict(False) == {
        "a": [0, 3],
        "b": [1, 4],
    }

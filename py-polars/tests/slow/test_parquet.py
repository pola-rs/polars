import io
import sys
import tempfile
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

import polars as pl
from polars.testing import assert_frame_equal


@typing.no_type_check
@pytest.mark.xfail(sys.platform == "win32", reason="Does not work on Windows")
def test_struct_pyarrow_dataset_5796() -> None:
    num_rows = 2**17 + 1

    df = pl.from_records(
        [dict(id=i, nested=dict(a=i)) for i in range(num_rows)]  # noqa: C408
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "out.parquet"
        df.write_parquet(file_path, use_pyarrow=True)
        tbl = ds.dataset(file_path).to_table()
        result = pl.from_arrow(tbl)

    assert_frame_equal(result, df)


def test_parquet_chunks_545() -> None:
    cases = [1048576, 1048577]

    for case in cases:
        f = io.BytesIO()
        # repeat until it has case instances
        df = pd.DataFrame(
            np.tile([1.0, pd.to_datetime("2010-10-10")], [case, 1]),
            columns=["floats", "dates"],
        )

        # write as parquet
        df.to_parquet(f)
        f.seek(0)

        # read it with polars
        polars_df = pl.read_parquet(f)
        assert pl.DataFrame(df).frame_equal(polars_df)

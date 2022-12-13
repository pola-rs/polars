import os
import typing

import pyarrow.dataset as ds

import polars as pl


@typing.no_type_check
def test_struct_pyarrow_dataset_5796() -> None:
    if os.name != "nt":
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
        assert pl.from_arrow(tbl).frame_equal(df)

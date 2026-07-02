"""
from datetime import datetime

import numpy as np
import polars as pl
import polars_cloud as pc

# --8<-- [start:set-context]
ctx = pc.ComputeContext(
    workspace="your-workspace",
    cpus=12,
    memory=12,
    requirements="requirements.txt",
)
# --8<-- [end:set-context]


# --8<-- [start:run-plugin]
import polars_xdt as xdt

lf = pl.LazyFrame(
    {
        "local_dt": [
            datetime(2020, 10, 10, 1),
            datetime(2020, 10, 10, 2),
            datetime(2020, 10, 9, 20),
        ],
        "timezone": [
            "Europe/London",
            "Africa/Kigali",
            "America/New_York",
        ],
    }
)

query = lf.with_columns(
    xdt.from_local_datetime("local_dt", pl.col("timezone"), "UTC").alias("date")
)

query.remote(ctx).show()
# --8<-- [end:run-plugin]


# --8<-- [start:run-udf]
import numpy as numpy

lf = pl.LazyFrame(
    {
        "keys": ["a", "a", "b", "b"],
        "values": [10, 7, 1, 23],
    }
)

q = lf.select(pl.col("values").map_batches(np.log, return_dtype=pl.Float64))

q.remote(ctx).show()
# --8<-- [end:run-udf]
"""

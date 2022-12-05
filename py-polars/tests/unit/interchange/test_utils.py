from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.internals.interchange.dataframe_protocol import Dtype, DtypeKind
from polars.internals.interchange.utils import NATIVE_ENDIANNESS as NE
from polars.internals.interchange.utils import polars_dtype_to_dtype


@pytest.mark.parametrize(
    "polars_dtype, dtype",
    [
        (pl.Int8, (DtypeKind.INT, 8, "c", NE)),
        (pl.Categorical, (DtypeKind.CATEGORICAL, 32, "I", NE)),
        (pl.Duration, (DtypeKind.DATETIME, 64, "tDu", NE)),
        (pl.Duration(time_unit="ns"), (DtypeKind.DATETIME, 64, "tDn", NE)),
        (pl.Datetime, (DtypeKind.DATETIME, 64, "tsu:", NE)),
        (pl.Datetime(time_unit="ms"), (DtypeKind.DATETIME, 64, "tsm:", NE)),
        (
            pl.Datetime(time_zone="Amsterdam/Europe"),
            (DtypeKind.DATETIME, 64, "tsu:Amsterdam/Europe", NE),
        ),
        (
            pl.Datetime(time_unit="ns", time_zone="Asia/Seoul"),
            (DtypeKind.DATETIME, 64, "tsn:Asia/Seoul", NE),
        ),
    ],
)
def test_polars_dtype_to_dtype(polars_dtype: pl.DataType, dtype: Dtype) -> None:
    assert polars_dtype_to_dtype(polars_dtype) == dtype

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.interchange.protocol import DtypeKind, Endianness
from polars.interchange.utils import polars_dtype_to_dtype

if TYPE_CHECKING:
    from polars.interchange.protocol import Dtype

NE = Endianness.NATIVE


@pytest.mark.parametrize(
    ("polars_dtype", "dtype"),
    [
        (pl.Int8, (DtypeKind.INT, 8, "c", NE)),
        (pl.Int16, (DtypeKind.INT, 16, "s", NE)),
        (pl.Int32, (DtypeKind.INT, 32, "i", NE)),
        (pl.Int64, (DtypeKind.INT, 64, "l", NE)),
        (pl.UInt8, (DtypeKind.UINT, 8, "C", NE)),
        (pl.UInt16, (DtypeKind.UINT, 16, "S", NE)),
        (pl.UInt32, (DtypeKind.UINT, 32, "I", NE)),
        (pl.UInt64, (DtypeKind.UINT, 64, "L", NE)),
        (pl.Float32, (DtypeKind.FLOAT, 32, "f", NE)),
        (pl.Float64, (DtypeKind.FLOAT, 64, "g", NE)),
        (pl.Boolean, (DtypeKind.BOOL, 1, "b", NE)),
        (pl.String, (DtypeKind.STRING, 8, "U", NE)),
        (pl.Date, (DtypeKind.DATETIME, 32, "tdD", NE)),
        (pl.Time, (DtypeKind.DATETIME, 64, "ttu", NE)),
        (pl.Categorical, (DtypeKind.CATEGORICAL, 32, "I", NE)),
        (pl.Enum, (DtypeKind.CATEGORICAL, 32, "I", NE)),
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


def test_polars_dtype_to_dtype_unsupported_type() -> None:
    with pytest.raises(ValueError, match="not supported"):
        polars_dtype_to_dtype(pl.List)

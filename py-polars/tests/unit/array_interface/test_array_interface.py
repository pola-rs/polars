from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import polars as pl


def assert_interface_equal(left: Any, right: Any) -> None:
    def set_defaults(interface: dict[str, Any]) -> dict[str, Any]:
        if "mask" not in interface:
            interface["mask"] = None
        if "offset" not in interface:
            interface["offset"] = 0
        return interface

    left_i = set_defaults(left.__array_interface__)
    right_i = set_defaults(right.__array_interface__)
    assert left_i == right_i


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ],
)
def test_series_numeric_no_copy(dtype: pl.PolarsDataType) -> None:
    s = pl.Series([1, 2, 3], dtype=dtype, strict=False)
    result = np.array(s, copy=False)
    assert_interface_equal(s, result)
    assert s.to_list() == result.tolist()

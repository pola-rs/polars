from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl

if TYPE_CHECKING:
    from polars._typing import TimeUnit


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_from_numpy_timedelta(time_unit: TimeUnit) -> None:
    s = pl.Series(
        "name",
        np.array(
            [timedelta(days=1), timedelta(seconds=1)], dtype=f"timedelta64[{time_unit}]"
        ),
    )
    assert s.dtype == pl.Duration(time_unit)
    assert s.name == "name"
    assert s.dt[0] == timedelta(days=1)
    assert s.dt[1] == timedelta(seconds=1)

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
import pytest

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import TimeUnit


@pytest.mark.parametrize("time_unit", ["ns", "ms", "us"])
@pytest.mark.parametrize("sequence", [np.array, list])
def test_from_numpy_datetime(
    time_unit: TimeUnit, sequence: Callable[[Iterable[Any]], Iterable[Any]]
) -> None:
    dt = datetime(2042, 1, 1, 11, 11, 11)
    s = pl.Series(
        "name",
        sequence(
            [
                np.datetime64(dt, time_unit),
            ]
        ),
    )
    assert s.dtype == pl.Datetime(time_unit)
    assert s.name == "name"
    assert s.dt[0] == dt


@pytest.mark.parametrize("time_unit", ["ns", "ms", "us"])
@pytest.mark.parametrize("sequence", [np.array, list])
def test_from_numpy_timedelta(
    time_unit: TimeUnit, sequence: Callable[[Iterable[Any]], Iterable[Any]]
) -> None:
    one_day_td = timedelta(days=1)
    one_second_td = timedelta(seconds=1)
    s = pl.Series(
        "name",
        sequence(
            [
                np.timedelta64(one_day_td, time_unit),
                np.timedelta64(one_second_td, time_unit),
            ]
        ),
    )
    assert s.dtype == pl.Duration(time_unit)
    assert s.name == "name"
    assert s.dt[0] == one_day_td
    assert s.dt[1] == one_second_td

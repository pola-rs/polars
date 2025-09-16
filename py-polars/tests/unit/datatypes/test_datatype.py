import copy
from datetime import datetime, time, timedelta

import polars as pl
from polars.datatypes import (
    is_array,
    is_categorical,
    is_datetime,
    is_decimal,
    is_duration,
    is_enum,
    is_list,
    is_struct,
    is_time,
)


# https://github.com/pola-rs/polars/issues/14771
def test_datatype_copy() -> None:
    dtype = pl.Int64()
    result = copy.deepcopy(dtype)
    assert dtype == result
    assert isinstance(result, pl.Int64)


def test_typeguards() -> None:
    df = pl.DataFrame(
        [
            pl.Series("array", [[1], [2], [3]], dtype=pl.Array(pl.Int8, 1)),
            pl.Series("cat", (abc := ["a", "b", "c"]), dtype=pl.Categorical),
            pl.Series("datetime", [datetime(2022, 1, 1)] * 3),
            pl.Series("decimal", [1.1] * 3, dtype=pl.Decimal()),
            pl.Series("duration", [timedelta(seconds=1)] * 3),
            pl.Series("enum", abc, dtype=pl.Enum(abc)),
            pl.Series("list", [[2], [3], [4]], dtype=pl.List),
            pl.Series("struct", [{"a": 1}] * 3),
            pl.Series("time", [time(8, 1, 0)] * 3),
        ]
    )

    for d in df.dtypes:
        if is_array(d):
            _ = d.width
        if is_categorical(d):
            _ = d.ordering
        if is_datetime(d):
            _ = d.time_zone
        if is_decimal(d):
            _ = d.scale
        if is_duration(d):
            _ = d.time_unit
        if is_enum(d):
            _ = d.categories
        if is_list(d):
            _ = d.inner
        if is_struct(d):
            _ = d.fields
        if is_time(d):
            _ = d.max()

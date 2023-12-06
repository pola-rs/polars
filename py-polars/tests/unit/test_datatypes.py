from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars import datatypes
from polars.datatypes import (
    DTYPE_TEMPORAL_UNITS,
    DataTypeGroup,
    Field,
    Int64,
    List,
    Struct,
    py_type_to_dtype,
)

if TYPE_CHECKING:
    from polars.datatypes import DataTypeClass

SIMPLE_DTYPES: list[DataTypeClass] = list(
    pl.INTEGER_DTYPES  # type: ignore[arg-type]
    | pl.FLOAT_DTYPES
    | {
        pl.Boolean,
        pl.Utf8,
        pl.Binary,
        pl.Time,
        pl.Date,
        pl.Object,
        pl.Null,
        pl.Unknown,
    }
)


def test_simple_dtype_init_takes_no_args() -> None:
    for dtype in SIMPLE_DTYPES:
        with pytest.raises(TypeError):
            dtype(10)


def test_simple_dtype_init_returns_instance() -> None:
    dtype = pl.Int8()
    assert isinstance(dtype, pl.Int8)


def test_complex_dtype_init_returns_instance() -> None:
    dtype = pl.Datetime()
    assert isinstance(dtype, pl.Datetime)
    assert dtype.time_unit == "us"


def test_dtype_temporal_units() -> None:
    # check (in)equality behaviour of temporal types that take units
    for time_unit in DTYPE_TEMPORAL_UNITS:
        assert pl.Datetime == pl.Datetime(time_unit)
        assert pl.Duration == pl.Duration(time_unit)

        assert pl.Datetime(time_unit) == pl.Datetime
        assert pl.Duration(time_unit) == pl.Duration

    assert pl.Datetime("ms") != pl.Datetime("ns")
    assert pl.Duration("ns") != pl.Duration("us")

    # check timeunit from pytype
    for inferred_dtype, expected_dtype in (
        (py_type_to_dtype(datetime), pl.Datetime),
        (py_type_to_dtype(timedelta), pl.Duration),
    ):
        assert inferred_dtype == expected_dtype
        assert inferred_dtype.time_unit == "us"  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="invalid `time_unit`"):
        pl.Datetime("?")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid `time_unit`"):
        pl.Duration("?")  # type: ignore[arg-type]


def test_dtype_base_type() -> None:
    assert pl.Date.base_type() is pl.Date
    assert pl.List(pl.Int32).base_type() is pl.List
    assert (
        pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)]).base_type()
        is pl.Struct
    )
    for dtype in pl.DATETIME_DTYPES:
        assert dtype.base_type() is pl.Datetime


def test_dtype_groups() -> None:
    grp = DataTypeGroup([pl.Datetime], match_base_type=False)
    assert pl.Datetime("ms", "Asia/Tokyo") not in grp

    grp = DataTypeGroup([pl.Datetime])
    assert pl.Datetime("ms", "Asia/Tokyo") in grp


def test_get_index_type() -> None:
    assert pl.get_index_type() == pl.UInt32


def test_dtypes_picklable() -> None:
    parametric_type = pl.Datetime("ns")
    singleton_type = pl.Float64
    assert pickle.loads(pickle.dumps(parametric_type)) == parametric_type
    assert pickle.loads(pickle.dumps(singleton_type)) == singleton_type


def test_dtypes_hashable() -> None:
    # ensure that all the types can be hashed, and that their hashes
    # are sufficient to ensure distinct entries in a dictionary/set

    all_dtypes = [
        getattr(datatypes, d)
        for d in dir(datatypes)
        if isinstance(getattr(datatypes, d), datatypes.DataType)
    ]
    assert len(set(all_dtypes + all_dtypes)) == len(all_dtypes)
    assert len({pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")}) == 3
    assert len({pl.List, pl.List(pl.Int16), pl.List(pl.Int32), pl.List(pl.Int64)}) == 4


@pytest.mark.parametrize(
    ("dtype", "representation"),
    [
        (pl.Boolean, "Boolean"),
        (pl.Datetime, "Datetime"),
        (
            pl.Datetime(time_zone="Europe/Amsterdam"),
            "Datetime(time_unit='us', time_zone='Europe/Amsterdam')",
        ),
        (pl.List(pl.Int8), "List(Int8)"),
        (pl.List(pl.Duration(time_unit="ns")), "List(Duration(time_unit='ns'))"),
        (pl.Struct, "Struct"),
        (
            pl.Struct({"name": pl.Utf8, "ids": pl.List(pl.UInt32)}),
            "Struct([Field('name', Utf8), Field('ids', List(UInt32))])",
        ),
    ],
)
def test_repr(dtype: pl.PolarsDataType, representation: str) -> None:
    assert repr(dtype) == representation


def test_conversion_dtype() -> None:
    df = (
        pl.DataFrame(
            {
                "id_column": [1, 2, 3, 4],
                "some_column": ["a", "b", "c", "d"],
                "some_partition_column": [
                    "partition_1",
                    "partition_2",
                    "partition_1",
                    "partition_2",
                ],
            }
        )
        .select(
            pl.struct(
                pl.col("id_column"), pl.col("some_column").cast(pl.Categorical)
            ).alias("struct"),
            pl.col("some_partition_column"),
        )
        .group_by("some_partition_column", maintain_order=True)
        .agg("struct")
    )

    result: pl.DataFrame = pl.from_arrow(df.to_arrow())  # type: ignore[assignment]
    # the assertion is not the real test
    # this tests if dtype has bubbled up correctly in conversion
    # if not we would UB
    expected = {
        "some_partition_column": ["partition_1", "partition_2"],
        "struct": [
            [
                {"id_column": 1, "some_column": "a"},
                {"id_column": 3, "some_column": "c"},
            ],
            [
                {"id_column": 2, "some_column": "b"},
                {"id_column": 4, "some_column": "d"},
            ],
        ],
    }
    assert result.to_dict(as_series=False) == expected


def test_struct_field_iter() -> None:
    s = Struct(
        [Field("a", List(List(Int64))), Field("b", List(Int64)), Field("c", Int64)]
    )
    assert list(s) == [
        ("a", List(List(Int64))),
        ("b", List(Int64)),
        ("c", Int64),
    ]
    assert list(reversed(s)) == [
        ("c", Int64),
        ("b", List(Int64)),
        ("a", List(List(Int64))),
    ]

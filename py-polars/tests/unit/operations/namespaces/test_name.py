from __future__ import annotations

from collections import OrderedDict

import polars as pl
from polars.testing import assert_frame_equal


def test_name_change_case() -> None:
    df = pl.DataFrame(
        schema={"ColX": pl.Int32, "ColY": pl.String},
    ).with_columns(
        pl.all().name.to_uppercase(),
        pl.all().name.to_lowercase(),
    )
    assert df.schema == OrderedDict(
        [
            ("ColX", pl.Int32),
            ("ColY", pl.String),
            ("COLX", pl.Int32),
            ("COLY", pl.String),
            ("colx", pl.Int32),
            ("coly", pl.String),
        ]
    )


def test_name_prefix_suffix() -> None:
    df = pl.DataFrame(
        schema={"ColX": pl.Int32, "ColY": pl.String},
    ).with_columns(
        pl.all().name.prefix("#"),
        pl.all().name.suffix("!!"),
    )
    assert df.schema == OrderedDict(
        [
            ("ColX", pl.Int32),
            ("ColY", pl.String),
            ("#ColX", pl.Int32),
            ("#ColY", pl.String),
            ("ColX!!", pl.Int32),
            ("ColY!!", pl.String),
        ]
    )


def test_name_update_all() -> None:
    df = pl.DataFrame(
        schema={
            "col1": pl.UInt32,
            "col2": pl.Float64,
            "other": pl.UInt64,
        }
    )
    assert (
        df.select(
            pl.col("col2").append(pl.col("other")),
            pl.col("col1").append(pl.col("other")).name.keep(),
            pl.col("col1").append(pl.col("other")).name.prefix("prefix_"),
            pl.col("col1").append(pl.col("other")).name.suffix("_suffix"),
        )
    ).schema == OrderedDict(
        [
            ("col2", pl.Float64),
            ("col1", pl.UInt64),
            ("prefix_col1", pl.UInt64),
            ("col1_suffix", pl.UInt64),
        ]
    )


def test_name_map_chain_21164() -> None:
    df = pl.DataFrame({"MyCol": [0, 1, 2]})
    assert_frame_equal(
        df.select(pl.all().name.to_lowercase().name.suffix("_suffix")),
        df.select(mycol_suffix=pl.col("MyCol")),
    )


def test_when_then_keep_map_13858() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    assert_frame_equal(
        df.with_columns(
            pl.when(True)
            .then(pl.int_range(3))
            .otherwise(pl.all())
            .name.keep()
            .name.suffix("_other")
        ),
        df.with_columns(a_other=pl.int_range(3), b_other=pl.int_range(3)),
    )


def test_keep_name_struct_field_23669() -> None:
    df = pl.DataFrame(
        [
            pl.Series("foo", [{"x": 1}], pl.Struct({"x": pl.Int64})),
            pl.Series("bar", [{"x": 2}], pl.Struct({"x": pl.Int64})),
        ]
    )
    assert_frame_equal(
        df.select(pl.all().struct.field("x").name.keep()),
        pl.DataFrame(
            [
                pl.Series("foo", [1], pl.Int64),
                pl.Series("bar", [2], pl.Int64),
            ]
        ),
    )

from __future__ import annotations

from collections import OrderedDict

import pytest

import polars as pl
from polars.exceptions import ComputeError


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


def test_name_replace() -> None:
    df = pl.DataFrame(
        schema={"n_foo": pl.Int32, "n_bar": pl.String, "misc?": pl.Float64},
    )

    assert df.select(
        pl.all().name.replace("^n_", "col_"),
    ).schema == {
        "col_foo": pl.Int32,
        "col_bar": pl.String,
        "misc?": pl.Float64,
    }

    assert df.select(
        pl.all().name.replace("(a|e|i|o|u)", "#"),
    ).schema == {
        "n_f##": pl.Int32,
        "n_b#r": pl.String,
        "m#sc?": pl.Float64,
    }

    with pytest.raises(ComputeError, match="repetition operator missing expression"):
        df.select(
            pl.all().name.replace("?", "!!"),
        )

    assert df.select(
        pl.all().name.replace("?", "!!", literal=True),
    ).schema == {
        "n_foo": pl.Int32,
        "n_bar": pl.String,
        "misc!!": pl.Float64,
    }


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

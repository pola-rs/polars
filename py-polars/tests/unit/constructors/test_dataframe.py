from __future__ import annotations

import pytest

import polars as pl


def test_df_init_strict() -> None:
    data = {"a": [1, 2, 3.0]}
    schema = {"a": pl.Int8}
    with pytest.raises(TypeError):
        pl.DataFrame(data, schema=schema, strict=True)

    df = pl.DataFrame(data, schema=schema, strict=False)

    # TODO: This should result in a Float Series without nulls
    # https://github.com/pola-rs/polars/issues/14427
    assert df["a"].to_list() == [1, 2, None]

    assert df["a"].dtype == pl.Int8


def test_df_init_from_series_strict() -> None:
    s = pl.Series("a", [-1, 0, 1])
    schema = {"a": pl.UInt8}
    with pytest.raises(pl.ComputeError):
        pl.DataFrame(s, schema=schema, strict=True)

    df = pl.DataFrame(s, schema=schema, strict=False)

    assert df["a"].to_list() == [None, 0, 1]
    assert df["a"].dtype == pl.UInt8

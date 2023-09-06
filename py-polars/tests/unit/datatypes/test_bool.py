from __future__ import annotations

import numpy as np
import pytest

import polars as pl


@pytest.mark.slow()
def test_bool_arg_min_max() -> None:
    # masks that ensures we take more than u64 chunks
    # and slicing and dicing to ensure the offsets work
    for _ in range(100):
        offset = np.random.randint(0, 100)
        sample = np.random.rand(1000)
        a = sample > 0.99
        idx = a[offset:].argmax()
        assert idx == pl.Series(a)[offset:].arg_max()
        idx = a[offset:].argmin()
        assert idx == pl.Series(a)[offset:].arg_min()

        a = sample > 0.01
        idx = a[offset:].argmax()
        assert idx == pl.Series(a)[offset:].arg_max()
        idx = a[offset:].argmin()
        assert idx == pl.Series(a)[offset:].arg_min()


def test_bool_sum_empty() -> None:
    assert pl.Series([], dtype=pl.Boolean).sum() == 0


def test_bool_min_max() -> None:
    assert pl.Series([None, True]).min()
    assert not pl.Series([None, True, False]).min()
    assert not pl.Series([False, True]).min()
    assert pl.Series([True, True]).min()
    assert not pl.Series([False, False]).min()
    assert pl.Series([None, True]).max()
    assert pl.Series([None, True, False]).max()
    assert pl.Series([False, True]).max()
    assert pl.Series([True, True]).max()
    assert not pl.Series([False, False]).max()


def test_bool_literal_expressions() -> None:
    df = pl.DataFrame({"x": [False, True]})

    def val(expr: pl.Expr) -> dict[str, list[bool]]:
        return df.select(expr).to_dict(as_series=False)

    assert val(pl.col("x") & False) == {"x": [False, False]}
    assert val(pl.col("x") & True) == {"x": [False, True]}
    assert val(pl.col("x") | False) == {"x": [False, True]}
    assert val(pl.col("x") | True) == {"x": [True, True]}
    assert val(pl.col("x") ^ False) == {"x": [False, True]}
    assert val(pl.col("x") ^ True) == {"x": [True, False]}
    assert val(False & pl.col("x")) == {"literal": [False, False]}
    assert val(True & pl.col("x")) == {"literal": [False, True]}
    assert val(False | pl.col("x")) == {"literal": [False, True]}
    assert val(True | pl.col("x")) == {"literal": [True, True]}
    assert val(False ^ pl.col("x")) == {"literal": [False, True]}
    assert val(True ^ pl.col("x")) == {"literal": [True, False]}

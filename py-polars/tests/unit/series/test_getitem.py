from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_series_equal
from polars.testing.parametric import series


@given(
    srs=series(max_size=10, dtype=pl.Int64),
    start=st.sampled_from([-5, -4, -3, -2, -1, None, 0, 1, 2, 3, 4, 5]),
    stop=st.sampled_from([-5, -4, -3, -2, -1, None, 0, 1, 2, 3, 4, 5]),
    step=st.sampled_from([-5, -4, -3, -2, -1, None, 1, 2, 3, 4, 5]),
)
def test_series_getitem(
    srs: pl.Series,
    start: int | None,
    stop: int | None,
    step: int | None,
) -> None:
    py_data = srs.to_list()

    s = slice(start, stop, step)
    sliced_py_data = py_data[s]
    sliced_pl_data = srs[s].to_list()

    assert sliced_py_data == sliced_pl_data, f"slice [{start}:{stop}:{step}] failed"
    assert_series_equal(srs, srs, check_exact=True)


@pytest.mark.parametrize(
    ("rng", "expected_values"),
    [
        (range(2), [1, 2]),
        (range(1, 4), [2, 3, 4]),
        (range(3, 0, -2), [4, 2]),
    ],
)
def test_series_getitem_range(rng: range, expected_values: list[int]) -> None:
    s = pl.Series([1, 2, 3, 4])
    result = s[rng]
    expected = pl.Series(expected_values)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "mask",
    [
        [True, False, True],
        pl.Series([True, False, True]),
        np.array([True, False, True]),
    ],
)
def test_series_getitem_boolean_mask(mask: Any) -> None:
    s = pl.Series([1, 2, 3])
    print(mask)
    with pytest.raises(
        TypeError,
        match="selecting rows by passing a boolean mask to `__getitem__` is not supported",
    ):
        s[mask]


@pytest.mark.parametrize(
    "input", [[], (), pl.Series(dtype=pl.Int64), np.array([], dtype=np.uint32)]
)
def test_series_getitem_empty_inputs(input: Any) -> None:
    s = pl.Series("a", ["x", "y", "z"], dtype=pl.String)
    result = s[input]
    expected = pl.Series("a", dtype=pl.String)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("indices", [[0, 2], pl.Series([0, 2]), np.array([0, 2])])
def test_series_getitem_multiple_indices(indices: Any) -> None:
    s = pl.Series(["x", "y", "z"])
    result = s[indices]
    expected = pl.Series(["x", "z"])
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "match"),
    [
        (
            [0.0, 1.0],
            "cannot select elements using Sequence with elements of type 'float'",
        ),
        (
            "foobar",
            "cannot select elements using Sequence with elements of type 'str'",
        ),
        (
            pl.Series([[1, 2], [3, 4]]),
            "cannot treat Series of type List\\(Int64\\) as indices",
        ),
        (np.array([0.0, 1.0]), "cannot treat NumPy array of type float64 as indices"),
        (object(), "cannot select elements using key of type 'object'"),
    ],
)
def test_series_getitem_col_invalid_inputs(input: Any, match: str) -> None:
    s = pl.Series([1, 2, 3])
    with pytest.raises(TypeError, match=match):
        s[input]

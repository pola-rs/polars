from typing import Any

import pytest

import polars as pl
from polars.utils._parse_expr_input import _inputs_to_list


@pytest.mark.parametrize("input", [None, []])
def test_inputs_to_list_empty(input: Any) -> None:
    assert _inputs_to_list(input) == []


@pytest.mark.parametrize(
    "input",
    [5, 2.0, "a", pl.Series([1, 2, 3]), pl.lit(4)],
)
def test_inputs_to_list_single(input: Any) -> None:
    assert _inputs_to_list(input) == [input]


@pytest.mark.parametrize(
    "input",
    [[5], ["a", "b"], (1, 2, 3), ["a", 5, 3.2]],
)
def test_inputs_to_list_multiple(input: Any) -> None:
    assert _inputs_to_list(input) == list(input)

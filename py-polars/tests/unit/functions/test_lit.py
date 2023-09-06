from __future__ import annotations

from typing import Any, Sequence

import pytest

import polars as pl


@pytest.mark.parametrize(
    "sequence",
    [
        [[1, 2], [3, 4, 5]],
        (1, 2, 3),
    ],
)
def test_lit_deprecated_sequence_input(sequence: Sequence[Any]) -> None:
    with pytest.deprecated_call():
        pl.lit(sequence)

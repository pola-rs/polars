from __future__ import annotations

import random

import numpy as np
import pytest
from polars.utils.various import parse_random_seed


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (42, 42),
        (np.random.default_rng(42), 892),
        (None, 2201),
    ],
)
def test_parse_random_seed(
    seed: int | np.random.Generator | None, expected: int
) -> None:
    """Check that the random seed is parsed correctly."""
    random.seed(1)
    assert parse_random_seed(seed) == expected

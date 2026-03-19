from typing import Any

import pytest

from polars.testing.parametric.strategies._utils import flexhash


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, 2),
        (1.0, 2.0),
        ("x", "y"),
        ([1, 2], [3, 4]),
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": [1.0]}, {"a": 1, "b": [1.5]}),
    ],
)
def test_flexhash_flat(left: Any, right: Any) -> None:
    assert flexhash(left) != flexhash(right)
    assert flexhash(left) == flexhash(left)
    assert flexhash(right) == flexhash(right)

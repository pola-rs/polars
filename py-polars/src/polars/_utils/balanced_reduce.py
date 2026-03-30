from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def balanced_reduce(function: Callable[[T, T], T], values: list[T]) -> T | None:
    """Applies a reduction in a balanced tree pattern."""
    if not values:
        return None

    last = [values.pop()] if len(values) > 1 and len(values) % 2 != 0 else []

    while len(values) > 1:
        for i in range(0, len(values), 2):
            v = function(values[i], values[i + 1])
            values[i // 2] = v

        values = values[: len(values) // 2]

    return function(values[0], last[0]) if last else values[0]

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def tree_reduce(function: Callable[[T, T], T], values: list[T]) -> T | None:
    """Applies a reduction in a tree pattern."""
    if not values:
        return None

    while len(values) > 1:
        for i in range(0, len(values), 2):
            v = values[i]

            if i + 1 < len(values):
                v = function(v, values[i + 1])

            values[i // 2] = v

        values = values[: -(len(values) // -2)]

    return values[0]

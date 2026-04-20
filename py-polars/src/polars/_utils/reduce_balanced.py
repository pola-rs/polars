from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def reduce_balanced(function: Callable[[T, T], T], iterable: Iterable[T]) -> T:
    """Applies a reduction in a balanced tree pattern."""
    values = list(iterable)

    if not values:
        msg = "reduce_balanced() of empty iterable"
        raise TypeError(msg)

    while len(values) > 2:
        half_floor = len(values) // 2

        for i in range(0, half_floor, 2):
            values[i // 2] = (
                values[i] if i + 1 == half_floor else function(values[i], values[i + 1])
            )

        rtree_offset = half_floor % 2

        for i in range(half_floor, len(values), 2):
            values[i // 2 + rtree_offset] = (
                values[i]
                if i + 1 == len(values)
                else function(values[i], values[i + 1])
            )

        n_ltree = -(half_floor // -2)
        n_rtree = -((len(values) - half_floor) // -2)

        del values[n_ltree + n_rtree :]

    return values.pop() if len(values) == 1 else function(values[0], values[1])

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def reduce_balanced(function: Callable[[T, T], T], iterable: Iterable[T]) -> T:
    """Applies a reduction in a balanced tree pattern."""
    values = list(iterable)

    if not values:
        msg = "reduce_balanced() of empty iterable"
        raise TypeError(msg)

    while len(values) > 1:
        rem = len(values) % 2

        for i in range(0, len(values) - rem, 2):
            values[i // 2] = function(values[i], values[i + 1])

        new_len = (len(values) // 2) + rem

        if rem:
            values[new_len - 1] = values[-1]

        del values[new_len:]

    return values.pop()

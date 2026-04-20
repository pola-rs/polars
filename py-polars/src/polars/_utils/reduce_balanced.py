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
        split_at = len(values) if len(values) <= 3 else len(values) // 2

        for i in range(0, split_at, 2):
            if i + 1 == split_at:
                prev = i // 2 - 1
                values[prev] = function(values[prev], values[i])
            else:
                values[i // 2] = function(values[i], values[i + 1])

        for i in range(split_at, len(values), 2):
            if i + 1 == len(values):
                prev = i // 2 - 1
                values[prev] = function(values[prev], values[i])
            else:
                values[i // 2] = function(values[i], values[i + 1])

        del values[split_at // 2 + (len(values) - split_at) // 2 :]

    return values.pop()

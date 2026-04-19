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
        last = [values.pop()] if len(values) % 2 != 0 else []

        for i in range(0, len(values), 2):
            v = function(values[i], values[i + 1])
            values[i // 2] = v

        del values[len(values) // 2 :]

        if last:
            values.append(last[0])

    return values[0]

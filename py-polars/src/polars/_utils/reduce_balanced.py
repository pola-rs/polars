from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def reduce_balanced(function: Callable[[T, T], T], iterable: Iterable[T]) -> T:
    """Applies a reduction in a balanced tree pattern."""
    values = list(iterable)

    if not values:
        msg = "reduce_balanced() of empty iterable"
        raise TypeError(msg)

    if len(values) == 1:
        return values.pop()

    stack = [(0, len(values))]

    i = 0

    while i < len(stack):
        offset, length = stack[i]
        half = -(length // -2)

        if length > 3:
            stack.append((offset + half, length - half))

        if length > 2:
            stack.append((offset, half))

        stack[i] = (offset, offset + half)

        i += 1

    for idx_l, idx_r in reversed(stack):
        values[idx_l] = function(values[idx_l], values[idx_r])

    return values[0]

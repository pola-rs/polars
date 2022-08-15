from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable

import polars.internals as pli

if TYPE_CHECKING:
    from polars.internals.type_aliases import Namespace

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    P = ParamSpec("P")


def expr(
    namespace: Namespace | None = None,
) -> Callable[[Callable[P, pli.Series]], Callable[P, pli.Series]]:
    """
    Dispatch Series method to an expression implementation.

    Decorator.

    Parameters
    ----------
    namespace : {'arr', 'cat', 'dt', 'str', 'struct'}
        Expression namespace.

    """

    def decorator(func: Callable[P, pli.Series]) -> Callable[P, pli.Series]:
        def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> pli.Series:
            s = pli.wrap_s(self._s)
            expr = pli.col(s.name)
            if namespace is not None:
                expr = getattr(expr, namespace)
            f = getattr(expr, func.__name__)
            return s.to_frame().select(f(*args, **kwargs)).to_series()

        return wrapper  # type: ignore[return-value]

    return decorator

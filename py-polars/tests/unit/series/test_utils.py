from collections.abc import Callable
from typing import cast

from polars import Series
from polars.series.utils import _is_empty_method


def test_is_empty_method_python_315_constants() -> None:
    def empty_method() -> None:
        """An empty method with a docstring."""

    # Python 3.15 omits the implicit None from co_consts for this function.
    empty_method.__code__ = empty_method.__code__.replace(
        co_consts=(empty_method.__doc__,)
    )

    assert _is_empty_method(cast("Callable[..., Series]", empty_method))


def test_is_empty_method_without_docstring() -> None:
    def empty_method() -> None:
        pass

    assert _is_empty_method(cast("Callable[..., Series]", empty_method))


def test_is_empty_method_rejects_implementation() -> None:
    def implemented_method() -> int:
        return 1

    assert not _is_empty_method(cast("Callable[..., Series]", implemented_method))

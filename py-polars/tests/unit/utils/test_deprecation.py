from __future__ import annotations

import inspect
from typing import Any

import pytest

from polars.utils.deprecation import (
    deprecate_function,
    deprecate_nonkeyword_arguments,
    deprecate_renamed_function,
    deprecate_renamed_parameter,
    issue_deprecation_warning,
    warn_closed_future_change,
)


def test_issue_deprecation_warning() -> None:
    with pytest.deprecated_call():
        issue_deprecation_warning("deprecated", version="0.1.2")


def test_deprecate_function() -> None:
    @deprecate_function("This is deprecated.", version="1.0.0")
    def hello() -> None:
        ...

    with pytest.deprecated_call():
        hello()


def test_deprecate_renamed_function() -> None:
    @deprecate_renamed_function("new_hello", version="1.0.0")
    def hello() -> None:
        ...

    with pytest.deprecated_call(match="new_hello"):
        hello()


def test_deprecate_renamed_parameter(recwarn: Any) -> None:
    @deprecate_renamed_parameter("foo", "oof", version="1.0.0")
    @deprecate_renamed_parameter("bar", "rab", version="2.0.0")
    def hello(oof: str, rab: str, ham: str) -> None:
        ...

    hello(foo="x", bar="y", ham="z")  # type: ignore[call-arg]

    assert len(recwarn) == 2
    assert "oof" in str(recwarn[0].message)
    assert "rab" in str(recwarn[1].message)


class Foo:  # noqa: D101
    @deprecate_nonkeyword_arguments(allowed_args=["self", "baz"], version="0.1.2")
    def bar(  # noqa: D102
        self, baz: str, ham: str | None = None, foobar: str | None = None
    ) -> None:
        ...


def test_deprecate_nonkeyword_arguments_method_signature() -> None:
    # Note the added star indicating keyword-only arguments after 'baz'
    expected = "(self, baz: 'str', *, ham: 'str | None' = None, foobar: 'str | None' = None) -> 'None'"
    assert str(inspect.signature(Foo.bar)) == expected


def test_deprecate_nonkeyword_arguments_method_warning() -> None:
    msg = (
        r"All arguments of Foo\.bar except for \'baz\' will be keyword-only in the next breaking release."
        r" Use keyword arguments to silence this warning."
    )
    with pytest.deprecated_call(match=msg):
        Foo().bar("qux", "quox")


def test_warn_closed_future_change() -> None:
    @warn_closed_future_change()
    def rolling_test(by: str | None = None, closed: str = "left") -> None:
        ...

    # If we pass `by` but not `closed`, we expect a warning
    with pytest.deprecated_call():
        rolling_test(by="b")


def test_warn_closed_future_change_no_warning(recwarn: Any) -> None:
    @warn_closed_future_change()
    def rolling_test(by: str | None = None, closed: str = "left") -> None:
        ...

    # No warning if we do not pass `by`
    rolling_test()

    # If we pass `by` and `closed`, we expect no warning...
    rolling_test(by="b", closed="left")

    # ... regardless of the value
    rolling_test(by="b", closed="right")

    assert len(recwarn) == 0

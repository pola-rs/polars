from __future__ import annotations

import inspect
from typing import Any

import pytest

from polars._utils.deprecation import (
    deprecate_function,
    deprecate_nonkeyword_arguments,
    deprecate_parameter_as_multi_positional,
    deprecate_renamed_function,
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)


def test_issue_deprecation_warning() -> None:
    with pytest.deprecated_call():
        issue_deprecation_warning("deprecated", version="0.1.2")


def test_deprecate_function() -> None:
    @deprecate_function("This is deprecated.", version="1.0.0")
    def hello() -> None: ...

    with pytest.deprecated_call():
        hello()


def test_deprecate_renamed_function() -> None:
    @deprecate_renamed_function("new_hello", version="1.0.0")
    def hello() -> None: ...

    with pytest.deprecated_call(match="new_hello"):
        hello()


def test_deprecate_renamed_parameter(recwarn: Any) -> None:
    @deprecate_renamed_parameter("foo", "oof", version="1.0.0")
    @deprecate_renamed_parameter("bar", "rab", version="2.0.0")
    def hello(oof: str, rab: str, ham: str) -> None: ...

    hello(foo="x", bar="y", ham="z")  # type: ignore[call-arg]

    assert len(recwarn) == 2
    assert "oof" in str(recwarn[0].message)
    assert "rab" in str(recwarn[1].message)


class Foo:  # noqa: D101
    @deprecate_nonkeyword_arguments(allowed_args=["self", "baz"], version="0.1.2")
    def bar(
        self, baz: str, ham: str | None = None, foobar: str | None = None
    ) -> None: ...


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


def test_deprecate_parameter_as_multi_positional(recwarn: Any) -> None:
    @deprecate_parameter_as_multi_positional("foo", version="1.0.0")
    def hello(*foo: str) -> tuple[str, ...]:
        return foo

    with pytest.deprecated_call():
        result = hello(foo="x")
    assert result == hello("x")

    with pytest.deprecated_call():
        result = hello(foo=["x", "y"])  # type: ignore[arg-type]
    assert result == hello("x", "y")


def test_deprecate_parameter_as_multi_positional_existing_arg(recwarn: Any) -> None:
    @deprecate_parameter_as_multi_positional("foo", version="1.0.0")
    def hello(bar: int, *foo: str) -> tuple[int, tuple[str, ...]]:
        return bar, foo

    with pytest.deprecated_call():
        result = hello(5, foo="x")
    assert result == hello(5, "x")

    with pytest.deprecated_call():
        result = hello(5, foo=["x", "y"])  # type: ignore[arg-type]
    assert result == hello(5, "x", "y")

from __future__ import annotations

import inspect
import warnings

import pytest

from polars.utils.deprecation import (
    deprecate_nonkeyword_arguments,
    deprecated,
    deprecated_name,
    issue_deprecation_warning,
    redirect,
)


def test_issue_deprecation_warning() -> None:
    with pytest.deprecated_call():
        issue_deprecation_warning("deprecated", version="0.1.2")


def test_deprecated_decorator() -> None:
    @deprecated("This is deprecated.", version="3.2.1")
    def hello() -> None:
        ...

    with pytest.deprecated_call():
        hello()


def test_deprecated_name_decorator() -> None:
    @deprecated_name("new_hello", version="3.2.1")
    def hello() -> None:
        ...

    with pytest.deprecated_call(match="new_hello"):
        hello()


def test_redirect() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # one-to-one redirection
        @redirect({"foo": "bar"})
        class DemoClass1:
            def bar(self, upper: bool = False) -> str:
                return "BAZ" if upper else "baz"

        assert DemoClass1().foo() == "baz"  # type: ignore[attr-defined]

        # redirection with **kwargs
        @redirect({"foo": ("bar", {"upper": True})})
        class DemoClass2:
            def bar(self, upper: bool = False) -> str:
                return "BAZ" if upper else "baz"

        assert DemoClass2().foo() == "BAZ"  # type: ignore[attr-defined]


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

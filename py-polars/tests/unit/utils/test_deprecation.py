from __future__ import annotations

import inspect
from typing import Any, get_args

import pytest

from polars._typing import DeprecationType
from polars._utils.deprecation import (
    deprecate_nonkeyword_arguments,
    deprecate_parameter_as_multi_positional,
    deprecate_renamed_parameter,
    deprecated,
    identify_deprecations,
    issue_deprecation_warning,
)


def test_issue_deprecation_warning() -> None:
    with pytest.deprecated_call(match=r"\(Deprecated in version 0\.1\.2\)"):
        issue_deprecation_warning("deprecated function", version="0.1.2")


def test_deprecate_function() -> None:
    @deprecated("`hello` is deprecated.")
    def hello() -> None: ...

    with pytest.deprecated_call():
        hello()


def test_deprecate_renamed_parameter(recwarn: Any) -> None:
    @deprecate_renamed_parameter("foo", "oof", version="1.2.3")
    @deprecate_renamed_parameter("bar", "rab", version="4.5.6")
    def hello(oof: str, rab: str, ham: str) -> None: ...

    hello(foo="x", bar="y", ham="z")  # type: ignore[call-arg]

    assert len(recwarn) == 2
    assert "oof" in str(recwarn[0].message)
    assert "rab" in str(recwarn[1].message)


class Foo:  # noqa: D101
    @deprecate_nonkeyword_arguments(allowed_args=["self", "baz"], version="1.0.0")
    def bar(
        self, baz: str, ham: str | None = None, foobar: str | None = None
    ) -> None: ...


def test_deprecate_nonkeyword_arguments_method_signature() -> None:
    # Note the added star indicating keyword-only arguments after 'baz'
    expected = "(self, baz: 'str', *, ham: 'str | None' = None, foobar: 'str | None' = None) -> 'None'"
    assert str(inspect.signature(Foo.bar)) == expected


def test_deprecate_nonkeyword_arguments_method_warning() -> None:
    msg = (
        r"all arguments of Foo\.bar except for \'baz\' will be keyword-only in the next breaking release."
        r" Use keyword arguments to silence this warning."
    )
    with pytest.deprecated_call(match=msg):
        Foo().bar("qux", "quox")


def test_deprecate_parameter_as_multi_positional(recwarn: Any) -> None:
    @deprecate_parameter_as_multi_positional("foo")
    def hello(*foo: str) -> tuple[str, ...]:
        return foo

    with pytest.deprecated_call():
        result = hello(foo="x")  # type: ignore[call-arg]
    assert result == hello("x")

    with pytest.deprecated_call():
        result = hello(foo=["x", "y"])  # type: ignore[call-arg, arg-type]
    assert result == hello("x", "y")


def test_deprecate_parameter_as_multi_positional_existing_arg(recwarn: Any) -> None:
    @deprecate_parameter_as_multi_positional("foo")
    def hello(bar: int, *foo: str) -> tuple[int, tuple[str, ...]]:
        return bar, foo

    with pytest.deprecated_call():
        result = hello(5, foo="x")  # type: ignore[call-arg]
    assert result == hello(5, "x")

    with pytest.deprecated_call():
        result = hello(5, foo=["x", "y"])  # type: ignore[call-arg, arg-type]
    assert result == hello(5, "x", "y")


def test_deprecated_decorator_ordering_26536() -> None:
    """Test that @deprecated works regardless of decorator ordering."""
    # @deprecated on top (outermost) — always worked
    @deprecated("`func_a` is deprecated.")
    @deprecate_renamed_parameter("old", "new", version="1.0.0")
    def func_a(new: str) -> str:
        return new

    with pytest.deprecated_call(match="`func_a` is deprecated."):
        assert func_a(old="x") == "x"  # type: ignore[call-arg]

    # @deprecated on bottom (innermost) — previously broken
    @deprecate_renamed_parameter("old", "new", version="1.0.0")
    @deprecated("`func_b` is deprecated.")
    def func_b(new: str) -> str:
        return new

    with pytest.deprecated_call(match="`func_b` is deprecated."):
        assert func_b(old="x") == "x"  # type: ignore[call-arg]

    # @deprecated sandwiched between multiple decorators
    @deprecate_renamed_parameter("foo", "bar", version="1.0.0")
    @deprecate_renamed_parameter("baz", "qux", version="1.0.0")
    @deprecated("`func_c` is deprecated.")
    def func_c(bar: str, qux: str) -> str:
        return bar + qux

    with pytest.deprecated_call(match="`func_c` is deprecated."):
        assert func_c(foo="a", baz="b") == "ab"  # type: ignore[call-arg]


@pytest.mark.slow
def test_identify_deprecations() -> None:
    dep = identify_deprecations()
    assert isinstance(dep, dict)

    valid_args = get_args(DeprecationType)
    assert all(key in valid_args for key in dep)

    with pytest.raises(
        ValueError,
        match="unrecognised deprecation type 'bitterballen'",
    ):
        identify_deprecations("bitterballen")  # type: ignore[arg-type]

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from polars._utils.deprecation import (
    deprecate_nonkeyword_arguments,
    deprecate_renamed_function,
)
from polars._utils.various import normalize_filepath
from polars._utils.wrap import wrap_expr
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from io import IOBase

    from polars import Expr


class ExprMetaNameSpace:
    """Namespace for expressions on a meta level."""

    _accessor = "meta"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def __eq__(self, other: ExprMetaNameSpace | Expr) -> bool:  # type: ignore[override]
        return self._pyexpr.meta_eq(other._pyexpr)

    def __ne__(self, other: ExprMetaNameSpace | Expr) -> bool:  # type: ignore[override]
        return not self == other

    def eq(self, other: ExprMetaNameSpace | Expr) -> bool:
        """
        Indicate if this expression is the same as another expression.

        Examples
        --------
        >>> foo_bar = pl.col("foo").alias("bar")
        >>> foo = pl.col("foo")
        >>> foo_bar.meta.eq(foo)
        False
        >>> foo_bar2 = pl.col("foo").alias("bar")
        >>> foo_bar.meta.eq(foo_bar2)
        True
        """
        return self._pyexpr.meta_eq(other._pyexpr)

    def ne(self, other: ExprMetaNameSpace | Expr) -> bool:
        """
        Indicate if this expression is NOT the same as another expression.

        Examples
        --------
        >>> foo_bar = pl.col("foo").alias("bar")
        >>> foo = pl.col("foo")
        >>> foo_bar.meta.ne(foo)
        True
        >>> foo_bar2 = pl.col("foo").alias("bar")
        >>> foo_bar.meta.ne(foo_bar2)
        False
        """
        return not self.eq(other)

    def has_multiple_outputs(self) -> bool:
        """
        Indicate if this expression expands into multiple expressions.

        Examples
        --------
        >>> e = pl.col(["a", "b"]).name.suffix("_foo")
        >>> e.meta.has_multiple_outputs()
        True
        """
        return self._pyexpr.meta_has_multiple_outputs()

    def is_column(self) -> bool:
        r"""
        Indicate if this expression is a basic (non-regex) unaliased column.

        Examples
        --------
        >>> e = pl.col("foo")
        >>> e.meta.is_column()
        True
        >>> e = pl.col("foo") * pl.col("bar")
        >>> e.meta.is_column()
        False
        >>> e = pl.col(r"^col.*\d+$")
        >>> e.meta.is_column()
        False
        """
        return self._pyexpr.meta_is_column()

    def is_regex_projection(self) -> bool:
        """
        Indicate if this expression expands to columns that match a regex pattern.

        Examples
        --------
        >>> e = pl.col("^.*$").name.prefix("foo_")
        >>> e.meta.is_regex_projection()
        True
        """
        return self._pyexpr.meta_is_regex_projection()

    def is_column_selection(self, *, allow_aliasing: bool = False) -> bool:
        """
        Indicate if this expression only selects columns (optionally with aliasing).

        This can include bare columns, column matches by regex or dtype, selectors
        and exclude ops, and (optionally) column/expression aliasing.

        .. versionadded:: 0.20.30

        Parameters
        ----------
        allow_aliasing
            If False (default), any aliasing is not considered pure column selection.
            Set True to allow for column selection that also includes aliasing.

        Examples
        --------
        >>> import polars.selectors as cs
        >>> e = pl.col("foo")
        >>> e.meta.is_column_selection()
        True
        >>> e = pl.col("foo").alias("bar")
        >>> e.meta.is_column_selection()
        False
        >>> e.meta.is_column_selection(allow_aliasing=True)
        True
        >>> e = pl.col("foo") * pl.col("bar")
        >>> e.meta.is_column_selection()
        False
        >>> e = cs.starts_with("foo")
        >>> e.meta.is_column_selection()
        True
        >>> e = cs.starts_with("foo").exclude("foo!")
        >>> e.meta.is_column_selection()
        True
        """
        return self._pyexpr.meta_is_column_selection(allow_aliasing)

    @overload
    def output_name(self, *, raise_if_undetermined: Literal[True] = True) -> str: ...

    @overload
    def output_name(self, *, raise_if_undetermined: Literal[False]) -> str | None: ...

    def output_name(self, *, raise_if_undetermined: bool = True) -> str | None:
        """
        Get the column name that this expression would produce.

        It may not always be possible to determine the output name as that can depend
        on the schema of the context; in that case this will raise `ComputeError` if
        `raise_if_undetermined` is True (the default), or `None` otherwise.

        Examples
        --------
        >>> e = pl.col("foo") * pl.col("bar")
        >>> e.meta.output_name()
        'foo'
        >>> e_filter = pl.col("foo").filter(pl.col("bar") == 13)
        >>> e_filter.meta.output_name()
        'foo'
        >>> e_sum_over = pl.sum("foo").over("groups")
        >>> e_sum_over.meta.output_name()
        'foo'
        >>> e_sum_slice = pl.sum("foo").slice(pl.len() - 10, pl.col("bar"))
        >>> e_sum_slice.meta.output_name()
        'foo'
        >>> pl.len().meta.output_name()
        'len'
        """
        try:
            return self._pyexpr.meta_output_name()
        except ComputeError:
            if not raise_if_undetermined:
                return None
            raise

    def pop(self) -> list[Expr]:
        """
        Pop the latest expression and return the input(s) of the popped expression.

        Returns
        -------
        list of Expr
            A list of expressions which in most cases will have a unit length.
            This is not the case when an expression has multiple inputs.
            For instance in a `fold` expression.

        Examples
        --------
        >>> e = pl.col("foo").alias("bar")
        >>> first = e.meta.pop()[0]
        >>> first.meta == pl.col("foo")
        True
        >>> first.meta == pl.col("bar")
        False
        """
        return [wrap_expr(e) for e in self._pyexpr.meta_pop()]

    def root_names(self) -> list[str]:
        """
        Get a list with the root column name.

        Examples
        --------
        >>> e = pl.col("foo") * pl.col("bar")
        >>> e.meta.root_names()
        ['foo', 'bar']
        >>> e_filter = pl.col("foo").filter(pl.col("bar") == 13)
        >>> e_filter.meta.root_names()
        ['foo', 'bar']
        >>> e_sum_over = pl.sum("foo").over("groups")
        >>> e_sum_over.meta.root_names()
        ['foo', 'groups']
        >>> e_sum_slice = pl.sum("foo").slice(pl.len() - 10, pl.col("bar"))
        >>> e_sum_slice.meta.root_names()
        ['foo', 'bar']
        """
        return self._pyexpr.meta_root_names()

    def undo_aliases(self) -> Expr:
        """
        Undo any renaming operation like `alias` or `name.keep`.

        Examples
        --------
        >>> e = pl.col("foo").alias("bar")
        >>> e.meta.undo_aliases().meta == pl.col("foo")
        True
        >>> e = pl.col("foo").sum().over("bar")
        >>> e.name.keep().meta.undo_aliases().meta == e
        True
        """
        return wrap_expr(self._pyexpr.meta_undo_aliases())

    def _as_selector(self) -> Expr:
        """Turn this expression in a selector."""
        return wrap_expr(self._pyexpr._meta_as_selector())

    def _selector_add(self, other: Expr) -> Expr:
        """Add selectors."""
        return wrap_expr(self._pyexpr._meta_selector_add(other._pyexpr))

    def _selector_sub(self, other: Expr) -> Expr:
        """Subtract selectors."""
        return wrap_expr(self._pyexpr._meta_selector_sub(other._pyexpr))

    def _selector_and(self, other: Expr) -> Expr:
        """& selectors."""
        return wrap_expr(self._pyexpr._meta_selector_and(other._pyexpr))

    @overload
    def serialize(self, file: None = ...) -> str: ...

    @overload
    def serialize(self, file: IOBase | str | Path) -> None: ...

    def serialize(self, file: IOBase | str | Path | None = None) -> str | None:
        """
        Serialize this expression to a file or string in JSON format.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to `None`
            (default), the output is returned as a string instead.

        See Also
        --------
        Expr.deserialize

        Examples
        --------
        Serialize the expression into a JSON string.

        >>> expr = pl.col("foo").sum().over("bar")
        >>> json = expr.meta.serialize()
        >>> json
        '{"Window":{"function":{"Agg":{"Sum":{"Column":"foo"}}},"partition_by":[{"Column":"bar"}],"options":{"Over":"GroupsToRows"}}}'

        The expression can later be deserialized back into an `Expr` object.

        >>> from io import StringIO
        >>> pl.Expr.deserialize(StringIO(json))  # doctest: +ELLIPSIS
        <Expr ['col("foo").sum().over([col("ba…'] at ...>
        """

        def serialize_to_string() -> str:
            with BytesIO() as buf:
                self._pyexpr.serialize(buf)
                json_bytes = buf.getvalue()
            return json_bytes.decode("utf8")

        if file is None:
            return serialize_to_string()
        elif isinstance(file, StringIO):
            json_str = serialize_to_string()
            file.write(json_str)
            return None
        elif isinstance(file, (str, Path)):
            file = normalize_filepath(file)
            self._pyexpr.serialize(file)
            return None
        else:
            self._pyexpr.serialize(file)
            return None

    @overload
    def write_json(self, file: None = ...) -> str: ...

    @overload
    def write_json(self, file: IOBase | str | Path) -> None: ...

    @deprecate_renamed_function("Expr.meta.serialize", version="0.20.11")
    def write_json(self, file: IOBase | str | Path | None = None) -> str | None:
        """
        Write expression to json.

        .. deprecated:: 0.20.11
            This method has been renamed to :meth:`serialize`.
        """
        return self.serialize(file)

    @overload
    def tree_format(self, *, return_as_string: Literal[False]) -> None: ...

    @overload
    def tree_format(self, *, return_as_string: Literal[True]) -> str: ...

    @deprecate_nonkeyword_arguments(version="0.19.3")
    def tree_format(self, return_as_string: bool = False) -> str | None:  # noqa: FBT001
        """
        Format the expression as a tree.

        Parameters
        ----------
        return_as_string:
            If True, return as string rather than printing to stdout.

        Examples
        --------
        >>> e = (pl.col("foo") * pl.col("bar")).sum().over(pl.col("ham")) / 2
        >>> e.meta.tree_format(return_as_string=True)  # doctest: +SKIP
        """
        s = self._pyexpr.meta_tree_format()
        if return_as_string:
            return s
        else:
            print(s)
            return None

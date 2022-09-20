from __future__ import annotations

import polars.internals as pli


class ExprMetaNameSpace:
    """Namespace for expressions on a meta level."""

    _accessor = "meta"

    def __init__(self, expr: pli.Expr):
        self._pyexpr = expr._pyexpr

    def __eq__(self, other: ExprMetaNameSpace | pli.Expr) -> bool:  # type: ignore[override] # noqa: #501
        return self._pyexpr.meta_eq(other._pyexpr)

    def __ne__(self, other: ExprMetaNameSpace | pli.Expr) -> bool:  # type: ignore[override] # noqa: #501
        return not self == other  # noqa: SIM201

    def pop(self) -> list[pli.Expr]:
        """
        Pop the latest expression and return the input(s) of the popped expression.

        Returns
        -------
        A list of expressions which in most cases will have a unit length.
        This is not the case when an expression has multiple inputs.
        For instance in a ``fold`` expression.

        """
        return [pli.wrap_expr(e) for e in self._pyexpr.meta_pop()]

    def root_names(self) -> list[str]:
        """Get a list with the root column name."""
        return self._pyexpr.meta_roots()

    def output_name(self) -> str:
        """
        Get the column name that this expression would produce.

        It might not always be possible to determine the output name
        as it might depend on the schema of the context. In that case
        this will raise a ``pl.ComputeError``.

        """
        return self._pyexpr.meta_output_name()

    def undo_aliases(self) -> pli.Expr:
        """Undo any renaming operation like ``alias`` or ``keep_name``."""
        return pli.wrap_expr(self._pyexpr.meta_undo_aliases())

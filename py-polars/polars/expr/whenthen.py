from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import polars.functions as F
from polars.expr.expr import Expr
from polars.utils._parse_expr_input import (
    parse_as_expression,
    parse_when_constraint_expressions,
)
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import (
    deprecate_renamed_parameter,
    issue_deprecation_warning,
)

if TYPE_CHECKING:
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr


class When:
    """
    Utility class for the `when-then-otherwise` expression.

    Represents the initial state of the expression after `pl.when(...)` is called.

    In this state, `then` must be called to continue to finish the expression.

    """

    def __init__(self, when: Any):
        self._when = when

    @deprecate_renamed_parameter("expr", "statement", version="0.18.9")
    def then(self, statement: IntoExpr) -> Then:
        """
        Attach a statement to the corresponding condition.

        Parameters
        ----------
        statement
            The statement to apply if the corresponding condition is true.
            Accepts expression input. Non-expression inputs are parsed as literals.

        """
        if isinstance(statement, str):
            _warn_for_deprecated_string_input_behavior(statement)
        statement_pyexpr = parse_as_expression(statement, str_as_lit=True)
        return Then(self._when.then(statement_pyexpr))


class Then(Expr):
    """
    Utility class for the `when-then-otherwise` expression.

    Represents the state of the expression after `pl.when(...).then(...)` is called.

    """

    def __init__(self, then: Any):
        self._then = then

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Expr:  # type: ignore[override]
        return wrap_expr(pyexpr)

    @property
    def _pyexpr(self) -> PyExpr:
        return self._then.otherwise(F.lit(None)._pyexpr)

    @deprecate_renamed_parameter("predicate", "condition", version="0.18.9")
    def when(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
        **constraints: Any,
    ) -> ChainedWhen:
        """
        Add a condition to the `when-then-otherwise` expression.

        Parameters
        ----------
        predicates
            Condition(s) that must be met in order to apply the subsequent statement.
            Accepts one or more boolean expressions, which are implicitly combined with
            `&`. String input is parsed as a column name.
        constraints
            Apply conditions as `colname = value` keyword arguments that are treated as
            equality matches, such as `x = 123`. As with the predicates parameter,
            multiple conditions are implicitly combined using `&`.

        """
        condition_pyexpr = parse_when_constraint_expressions(*predicates, **constraints)
        return ChainedWhen(self._then.when(condition_pyexpr))

    @deprecate_renamed_parameter("expr", "statement", version="0.18.9")
    def otherwise(self, statement: IntoExpr) -> Expr:
        """
        Define a default for the `when-then-otherwise` expression.

        Parameters
        ----------
        statement
            The statement to apply if all conditions are false.
            Accepts expression input. Non-expression inputs are parsed as literals.

        """
        if isinstance(statement, str):
            _warn_for_deprecated_string_input_behavior(statement)
        statement_pyexpr = parse_as_expression(statement, str_as_lit=True)
        return wrap_expr(self._then.otherwise(statement_pyexpr))


class ChainedWhen(Expr):
    """
    Utility class for the `when-then-otherwise` expression.

    Represents the state of the expression after an additional `when` is called.

    In this state, `then` must be called to continue to finish the expression.

    """

    def __init__(self, chained_when: Any):
        self._chained_when = chained_when

    @deprecate_renamed_parameter("expr", "statement", version="0.18.9")
    def then(self, statement: IntoExpr) -> ChainedThen:
        """
        Attach a statement to the corresponding condition.

        Parameters
        ----------
        statement
            The statement to apply if the corresponding condition is true.
            Accepts expression input. Non-expression inputs are parsed as literals.

        """
        if isinstance(statement, str):
            _warn_for_deprecated_string_input_behavior(statement)
        statement_pyexpr = parse_as_expression(statement, str_as_lit=True)
        return ChainedThen(self._chained_when.then(statement_pyexpr))


class ChainedThen(Expr):
    """
    Utility class for the `when-then-otherwise` expression.

    Represents the state of the expression after an additional `then` is called.

    """

    def __init__(self, chained_then: Any):
        self._chained_then = chained_then

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Expr:  # type: ignore[override]
        return wrap_expr(pyexpr)

    @property
    def _pyexpr(self) -> PyExpr:
        return self._chained_then.otherwise(F.lit(None)._pyexpr)

    @deprecate_renamed_parameter("predicate", "condition", version="0.18.9")
    def when(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
        **constraints: Any,
    ) -> ChainedWhen:
        """
        Add another condition to the `when-then-otherwise` expression.

        Parameters
        ----------
        predicates
            Condition(s) that must be met in order to apply the subsequent statement.
            Accepts one or more boolean expressions, which are implicitly combined with
            `&`. String input is parsed as a column name.
        constraints
            Apply conditions as `colname = value` keyword arguments that are treated as
            equality matches, such as `x = 123`. As with the predicates parameter,
            multiple conditions are implicitly combined using `&`.

        """
        condition_pyexpr = parse_when_constraint_expressions(*predicates, **constraints)
        return ChainedWhen(self._chained_then.when(condition_pyexpr))

    @deprecate_renamed_parameter("expr", "statement", version="0.18.9")
    def otherwise(self, statement: IntoExpr) -> Expr:
        """
        Define a default for the `when-then-otherwise` expression.

        Parameters
        ----------
        statement
            The statement to apply if all conditions are false.
            Accepts expression input. Non-expression inputs are parsed as literals.

        """
        if isinstance(statement, str):
            _warn_for_deprecated_string_input_behavior(statement)
        statement_pyexpr = parse_as_expression(statement, str_as_lit=True)
        return wrap_expr(self._chained_then.otherwise(statement_pyexpr))


def _warn_for_deprecated_string_input_behavior(input: str) -> None:
    issue_deprecation_warning(
        "in a future version, string input will be parsed as a column name rather than a string literal."
        f" To silence this warning, pass the input as an expression instead: `pl.lit({input!r})`",
        version="0.18.9",
    )

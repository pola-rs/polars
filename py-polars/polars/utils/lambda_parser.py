"""Try to suggest faster alternatives to simple lambda expressions used in `apply`."""
from __future__ import annotations

import ast
import sys
import warnings
from typing import Any, Callable

from polars.exceptions import PolarsInefficientApplyWarning
from polars.utils.various import find_stacklevel

AST_OPERATOR_TO_STR = {
    ast.Add: "+",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Sub: "-",
    ast.Mod: "%",
    ast.FloorDiv: "//",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
}
AST_CMPOP_TO_STR = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


def _ast_expr_to_str(
    operand: ast.AST, arg: str, names: list[str], *, level: int, is_expr: bool
) -> str | None:
    """
    Convert an operand of a binary operation to str.

    Parameters
    ----------
    operand
        The operand to process.
    arg
        The argument name (e.g. if the lambda is `lambda x: x + 1`, then `arg` is `x`).
    names
        The column names (or root names) of the original DataFrame (or Expr).
    level
        How deep we are in the AST.
    is_expr
        Whether the original object was a DataFrame or an Expr.


    Returns
    -------
    str | None
        Return a replacement expression for the `apply` (if found).
    """
    import ast

    if isinstance(operand, ast.Subscript) and not is_expr:
        return _ast_subscript_to_str(operand, arg, names)
    elif isinstance(operand, ast.BinOp):
        ret = _ast_binop_to_str(operand, arg, names, level=level, is_expr=is_expr)
        if ret is None:
            return None
        return f"({ret})"
    elif isinstance(operand, ast.Compare):
        ret = _ast_compare_to_str(operand, arg, names, level=level, is_expr=is_expr)
        if ret is None:
            return None
        return f"({ret})"
    elif isinstance(operand, ast.Constant):
        return operand.value
    elif isinstance(operand, ast.Name):
        if len(names) == 1 and operand.id == arg and is_expr:
            return f'pl.col("{names[0]}")'
        return None
    return None


def _ast_operator_to_str(op: ast.operator) -> str | None:
    """Return the string representation of some simple operators."""
    for ast_op, str_op in AST_OPERATOR_TO_STR.items():
        if isinstance(op, ast_op):
            return str_op
    return None


def _ast_cmpop_to_str(op: ast.cmpop) -> str | None:
    """Return the string representation of some simple operators."""
    for ast_op, str_op in AST_CMPOP_TO_STR.items():
        if isinstance(op, ast_op):
            return str_op
    return None


def _ast_binop_to_str(
    binop: ast.BinOp, arg: str, columns: list[str], level: int, is_expr: bool
) -> str | None:
    """
    Convert a binary operation, such as `x + 1`, to str.

    Only return if the left-hand-side, the right-hand-side, and the operator are
    simple.
    """
    if level > 5:
        # We don't want to go too deep in the AST. Note that it's incredibly unlikely
        # that we'll ever reach this point, as only single-line lambda expressions
        # are allowed.
        return None
    op = _ast_operator_to_str(binop.op)
    if op is None:
        return None
    left = _ast_expr_to_str(binop.left, arg, columns, level=level + 1, is_expr=is_expr)
    if left is None:
        return None
    right = _ast_expr_to_str(
        binop.right, arg, columns, level=level + 1, is_expr=is_expr
    )
    if right is None:
        return None
    return f"{left} {op} {right}"


def _ast_compare_to_str(
    binop: ast.Compare, arg: str, columns: list[str], level: int, is_expr: bool
) -> str | None:
    """
    Convert a binary comparison, such as `x == 1`, to str.

    Only return if the left-hand-side, the right-hand-side, and the operator are
    simple.
    """
    if level > 5:
        # We don't want to go too deep in the AST. Note that it's incredibly unlikely
        # that we'll ever reach this point, as only single-line lambda expressions
        # are allowed.
        return None
    if len(binop.ops) != 1:
        return None
    op = _ast_cmpop_to_str(binop.ops[0])
    if op is None:
        return None
    left = _ast_expr_to_str(binop.left, arg, columns, level=level + 1, is_expr=is_expr)
    if left is None:
        return None
    if len(binop.comparators) != 1:
        return None
    right = _ast_expr_to_str(
        binop.comparators[0], arg, columns, level=level + 1, is_expr=is_expr
    )
    if right is None:
        return None
    return f"{left} {op} {right}"


def _ast_subscript_to_str(
    subscript: ast.Subscript, arg: str, columns: list[str]
) -> str | None:
    """
    Convert a subscript expression, such as `x[0]`, to str.

    Only handle the simple case where the slice is a constant integer,
    and replace it with the corresponding column name.
    """
    import ast

    if not isinstance(subscript.value, ast.Name):
        return None
    if subscript.value.id != arg:
        return None
    if sys.version_info < (3, 9):
        # ast.Index was deprecated in Python 3.9.
        slice_ = subscript.slice.value
    else:
        slice_ = subscript.slice
    if not isinstance(slice_, ast.Constant):
        return None
    if not isinstance(slice_.value, int):
        return None
    return f'pl.col("{columns[slice_.value]}")'


def _parse_lambda_from_function(function: Callable[[Any], Any]) -> str | None:
    """
    Extract the lambda expression from a function.

    We use `insect.getsource`, which returns the entire line of code which the user
    wrote. We then extract the lambda expression from that line of code.
    """
    import inspect

    try:
        src = inspect.getsource(function)
    except Exception:
        return None
    src = src[src.find("lambda") :]
    parens = 1
    chars = []
    for character in src:
        if character == "(":
            parens += 1
        elif character == ")":
            parens -= 1
        if parens == 0:
            break
        chars.append(character)
    else:
        return None
    src = "".join(chars)
    return src


def _str_lambda_to_ast_lambda(src: str) -> ast.Lambda | None:
    """
    Parse a lambda expression into an AST tree.

    Only supports simple lambda expressions with a single argument
    and where the return value is a binary operation.

    Examples
    --------
    - `lambda x: x + 1`
    - `lambda x: x[0] + 1`
    - `lambda x: x**2`
    """
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    if len(tree.body) != 1:
        return None
    if not isinstance(tree.body[0], ast.Expr):
        return None
    lambda_ = tree.body[0].value
    if not isinstance(lambda_, ast.Lambda):
        return None
    args = lambda_.args.args
    if len(args) != 1:
        return None
    if not isinstance(lambda_.body, (ast.BinOp, ast.Compare)):
        return None
    return lambda_


def maybe_warn_about_dataframe_apply_function(
    function: Callable[[Any], Any], columns: list[str]
) -> None:
    str_lambda = _parse_lambda_from_function(function)
    if str_lambda is None:
        return None
    ast_lambda = _str_lambda_to_ast_lambda(str_lambda)
    if ast_lambda is None:
        return
    suggestion = _ast_expr_to_str(
        ast_lambda.body, ast_lambda.args.args[0].arg, columns, level=0, is_expr=False
    )

    if suggestion:
        warnings.warn(
            "DataFrame.apply is much slower than the native expressions API. Only use "
            "it if you cannot implement your logic otherwise.\n"
            "In this simple case, you can replace:\n"
            f"\033[31m-    .apply({str_lambda})\033[0m\n"
            "\n"
            "with:\n"
            f'\033[32m+    .select({suggestion}.alias("apply"))\033[0m\n',
            PolarsInefficientApplyWarning,
            stacklevel=find_stacklevel(),
        )


def maybe_warn_about_expr_apply_function(
    function: Callable[[Any], Any], columns: list[str]
) -> None:
    str_lambda = _parse_lambda_from_function(function)
    if str_lambda is None:
        return None
    ast_lambda = _str_lambda_to_ast_lambda(str_lambda)
    if ast_lambda is None:
        return
    suggestion = _ast_expr_to_str(
        ast_lambda.body, ast_lambda.args.args[0].arg, columns, level=0, is_expr=True
    )

    if suggestion:
        warnings.warn(
            "Expr.apply is much slower than the native expressions API. Only use "
            "it if you cannot implement your logic otherwise.\n"
            "In this simple case, you can replace:\n"
            f'\033[31m-    pl.col("{columns[0]}").apply({str_lambda})\033[0m\n'
            "\n"
            "with:\n"
            f"\033[32m+    {suggestion}\033[0m\n",
            PolarsInefficientApplyWarning,
            stacklevel=find_stacklevel(),
        )

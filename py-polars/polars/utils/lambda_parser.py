"""Try to suggest faster alternatives to simple lambda expressions used in `apply`."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable

from polars.exceptions import PolarsInefficientApplyWarning
from polars.utils.various import find_stacklevel

if TYPE_CHECKING:
    import ast


def _process_operand(
    operand: ast.AST, arg: str, names: list[str], level: int, is_expr: bool
) -> str | None:
    """
    Process an operand of a binary operation.

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
        return _process_subscript(operand, arg, names)
    elif isinstance(operand, ast.BinOp):
        ret = _process_binop(operand, arg, names, level=level, is_expr=is_expr)
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


def _process_operator(op: ast.operator) -> str | None:
    """Return the string representation of some simple operators."""
    import ast
    if isinstance(op, ast.Add):
        return "+"
    elif isinstance(op, ast.Mult):
        return "*"
    elif isinstance(op, ast.Div):
        return "/"
    elif isinstance(op, ast.Sub):
        return "-"
    elif isinstance(op, ast.Mod):
        return "%"
    elif isinstance(op, ast.FloorDiv):
        return "//"
    elif isinstance(op, ast.Pow):
        return "**"
    elif isinstance(op, ast.Lt):
        return "<"
    elif isinstance(op, ast.LtE):
        return "<="
    elif isinstance(op, ast.Gt):
        return ">"
    elif isinstance(op, ast.GtE):
        return "<="
    elif isinstance(op, ast.Eq):
        return "=="
    elif isinstance(op, ast.NotEq):
        return "!="
    elif isinstance(op, ast.BitAnd):
        return "&"
    elif isinstance(op, ast.BitOr):
        return "|"
    return None


def _process_binop(
    binop: ast.BinOp, arg: str, columns: list[str], level: int, is_expr: bool
) -> str | None:
    """
    Process a binary operation, such as `x + 1`.

    Only return if the left-hand-side, the right-hand-side, and the operator are
    simple.
    """
    if level > 5:
        # We don't want to go too deep in the AST. Note that it's incredibly unlikely
        # that we'll ever reach this point, as only single-line lambda expressions
        # are allowed.
        return None
    op = _process_operator(binop.op)
    if op is None:
        return None
    left = _process_operand(binop.left, arg, columns, level + 1, is_expr)
    if left is None:
        return None
    right = _process_operand(binop.right, arg, columns, level + 1, is_expr)
    if right is None:
        return None
    return f"{left} {op} {right}"


def _process_subscript(
    subscript: ast.Subscript, arg: str, columns: list[str]
) -> str | None:
    """
    Process a subscript expression, such as `x[0]`.

    Only handle the simple case where the slice is a constant integer,
    and replace it with the corresponding column name.
    """
    import ast

    if not isinstance(subscript.value, ast.Name):
        return None
    if subscript.value.id != arg:
        return None
    if not isinstance(subscript.slice, ast.Constant):
        return None
    if not isinstance(subscript.slice.value, int):
        return None
    return f'pl.col("{columns[subscript.slice.value]}")'


def _extract_lambda(function: Callable[[Any], Any]) -> str | None:
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


def _parse_tree(src: str) -> ast.Lambda | None:
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
    # todo: allow ast.Compare
    if not isinstance(lambda_.body, ast.BinOp):
        return None
    return lambda_


def maybe_warn_about_dataframe_apply_function(
    function: Callable[[Any], Any], columns: list[str]
) -> None:
    src = _extract_lambda(function)
    if src is None:
        return None
    lambda_ = _parse_tree(src)
    if lambda_ is None:
        return
    out = _process_operand(
        lambda_.body, lambda_.args.args[0].arg, columns, level=0, is_expr=False
    )

    if out:
        warnings.warn(
            "DataFrame.apply is much slower than the native expressions API. Only use "
            "it if you cannot implement your logic otherwise.\n"
            "In this simple case, you can replace:\n"
            f"\033[31m-    .apply({src})\033[0m\n"
            "\n"
            "with:\n"
            f"\033[32m+    .select({out})\033[0m\n",
            PolarsInefficientApplyWarning,
            stacklevel=find_stacklevel(),
        )


def maybe_warn_about_expr_apply_function(
    function: Callable[[Any], Any], columns: list[str]
) -> None:
    src = _extract_lambda(function)
    if src is None:
        return None
    lambda_ = _parse_tree(src)
    if lambda_ is None:
        return
    out = _process_operand(
        lambda_.body, lambda_.args.args[0].arg, columns, level=0, is_expr=True
    )

    if out:
        warnings.warn(
            "Expr.apply is much slower than the native expressions API. Only use "
            "it if you cannot implement your logic otherwise.\n"
            "In this simple case, you can replace:\n"
            f'\033[31m-    pl.col("{columns[0]}").apply({src})\033[0m\n'
            "\n"
            "with:\n"
            f"\033[32m+    {out}\033[0m\n",
            PolarsInefficientApplyWarning,
            stacklevel=find_stacklevel(),
        )

"""Utilities related to user defined functions (such as those passed to `apply`)."""
from __future__ import annotations

import dis
import sys
import warnings
from inspect import signature
from typing import Any, Callable

from polars.exceptions import PolarsInefficientApplyWarning
from polars.utils.various import find_stacklevel, in_terminal_that_supports_colour

# Note: in 3.11 individual binary opcodes were folded into a new BINARY_OP
_BINARY_OPCODES = {
    "BINARY_ADD": "+",
    "BINARY_AND": "&",
    "BINARY_FLOOR_DIVIDE": "//",
    "BINARY_MODULO": "%",
    "BINARY_MULTIPLY": "*",
    "BINARY_OR": "|",
    "BINARY_POWER": "**",
    "BINARY_SUBTRACT": "-",
    "BINARY_TRUE_DIVIDE": "/",
    "BINARY_XOR": "^",
}
_UNARY_OPCODES = {
    "UNARY_NEGATIVE": "-",
    "UNARY_POSITIVE": "+",
    "UNARY_NOT": "~",
}
_SIMPLE_EXPR_OPS = {
    "BINARY_OP",
    "COMPARE_OP",
    "CONTAINS_OP",
    "IS_OP",
    "LOAD_CONST",
    "LOAD_FAST",
}
_SIMPLE_EXPR_OPS |= set(_UNARY_OPCODES)
_SIMPLE_FRAME_OPS = _SIMPLE_EXPR_OPS | {"BINARY_SUBSCR"}

REPORT_MSG = (
    "Please report a bug at https://github.com/pola-rs/polars/issues, including "
    "the function which you passed to `apply`."
)


def _expr(value: str | tuple[str, str, str], col: str, param_name: str) -> str:
    if isinstance(value, tuple):
        op = value[1]
        unary_op = len(value) == 2
        e1 = _expr(value[0], col, param_name)
        e2 = _expr(value[2], col, param_name) if not unary_op else None

        if unary_op:
            return f"{op}{e1}"
        elif op in ("is", "is not") and value[2] == "None":
            not_ = "" if op == "is" else "not_"
            return f"{e1}.is_{not_}null()"
        elif op in ("in", "not in"):
            not_ = "" if op == "in" else "~"
            return f"{not_}({e1}.is_in({e2}))"
        else:
            return f"({e1} {op} {e2})"

    elif value == param_name:
        return f'pl.col("{col}")'

    return value


def _op(opname: str, argrepr: str, argval: Any) -> str:
    if argrepr:
        return argrepr
    elif opname == "IS_OP":
        return "is not" if argval else "is"
    elif opname == "CONTAINS_OP":
        return "not in" if argval else "in"
    elif opname.startswith("UNARY_"):
        return _UNARY_OPCODES[opname]
    else:
        raise AssertionError(
            "Unrecognised op - please report a bug to https://github.com/pola-rs/polars/issues "
            "with the content of function you were passing to `apply`."
        )


def _function_name(function: Callable[[Any], Any], param_name: str) -> str:
    """Return the name of the given function/method/lambda."""
    func_name = function.__name__
    if func_name == "<lambda>":
        return f"lambda {param_name}: ..."
    elif not func_name:
        return "..."
    return func_name


def _get_bytecode_ops(function: Callable[[Any], Any]) -> list[tuple[str, str, Any]]:
    """Return disassembled bytecode ops, arg-repr, and arg-specific value/flag."""
    try:
        instructions = list(dis.get_instructions(function))
        idx_last = len(instructions) - 1
        return [
            _inst(inst.opname, inst.argrepr, inst.argval)
            for idx, inst in enumerate(instructions)
            if (idx, inst.opname) not in ((0, "RESUME"), (idx_last, "RETURN_VALUE"))
        ]
    except TypeError:
        # in case we hit something that can't be disassembled,
        # eg: code object not available (such as a bare numpy
        # ufunc that isn't contained within a lambda/function)
        return []


def _inst(opname: str, argrepr: str, argval: str) -> tuple[str, str, str]:
    """Rewrite older opcodes using py 3.11 'BINARY_OP' instead."""
    if sys.version_info < (3, 11) and opname in _BINARY_OPCODES:
        argrepr = _BINARY_OPCODES[opname]
        opname = "BINARY_OP"
    return opname, argrepr, argval


def _can_rewrite_as_expression(
    ops: list[tuple[str, str, Any]], apply_target: str
) -> bool:
    """
    Determine if bytecode indicates only simple binary ops and/or comparisons.

    Note that `lambda x: x` is inefficient, but we exclude it as it's otherwise not
    guaranteed that just using the constant will return the same output. Hopefully
    nobody is writing lambdas like that anyway...
    """
    simple_ops = _SIMPLE_FRAME_OPS if apply_target == "frame" else _SIMPLE_EXPR_OPS
    return (
        bool(ops)
        and all(op in simple_ops for op, _argrepr, _argval in ops)
        and len(ops) >= 3
    )


def _to_polars_expression(
    ops: list[tuple[str, str, Any]], col: str, apply_target: str, param_name: str
) -> str | None:
    """
    Take postfix opcode stack and translate to native polars expression.

    Parameters
    ----------
    ops
        The list of opcodes, argreprs, and argvals.
    col
        The column name of the original object. In the case of an ``Expr``, this
        will be the expression's root name.
    apply_target
        The target of the ``apply`` call. One of ``"expr"``, ``"frame"``,
        or ``"series"``.
    param_name
        Argument of the function / lambda (e.g. if it's ``lambda x: x + 1``, then
        ``param_name`` will be ``"x"``).
    """
    if apply_target == "expr":
        stack = []  # type: ignore[var-annotated]
        for op in ops:
            stack.append(
                op[1]
                if op[0] in ("LOAD_FAST", "LOAD_CONST")
                else (
                    (stack.pop(), _op(*op))
                    if op[0].startswith("UNARY_")
                    else (stack.pop(-2), _op(*op), stack.pop(-1))
                )
            )
        return _expr(stack[0], col, param_name)  # type: ignore[arg-type]

    # TODO: frame apply (account for BINARY_SUBSCR)
    # TODO: series apply (rewrite col expr as series)
    return None


def _param_name_from_signature(function: Callable[[Any], Any]) -> str | None:
    try:
        sig = signature(function)
    except ValueError:
        return None
    parameters = sig.parameters
    if len(parameters) == 1:
        return next(iter(parameters.keys()))
    return None


def _generate_warning(
    function: Callable[[Any], Any], suggestion: str, col: str, param_name: str
) -> None:
    func_name = _function_name(function, param_name)
    addendum = (
        'Note: in list.eval context, pl.col("") should be written as pl.element()'
        if 'pl.col("")' in suggestion
        else ""
    )
    before_after_suggestion = (
        (
            f'  \033[31m-  pl.col("{col}").apply({func_name})\033[0m\n'
            f"  \033[32m+  {suggestion}\033[0m\n{addendum}"
        )
        if in_terminal_that_supports_colour()
        else (
            f'  - pl.col("{col}").apply({func_name})\n' f"  + {suggestion}\n{addendum}"
        )
    )
    warnings.warn(
        "\nExpr.apply is significantly slower than the native expressions API.\n"
        "Only use if you absolutely CANNOT implement your logic otherwise.\n"
        "In this case, you can replace your `apply` with an expression:\n"
        f"{before_after_suggestion}",
        PolarsInefficientApplyWarning,
        stacklevel=find_stacklevel(),
    )


def warn_on_inefficient_apply(
    function: Callable[[Any], Any], columns: list[str], apply_target: str
) -> None:
    """
    Generate ``PolarsInefficientApplyWarning`` on poor usage of ``apply`` func.

    Parameters
    ----------
    function
        The function passed to ``apply``.
    columns
        The column names of the original object. In the case of an ``Expr``, this
        will be a list of length 1 containing the expression's root name.
    apply_target
        The target of the ``apply`` call. One of ``"expr"``, ``"frame"``,
        or ``"series"``.
    """
    # only consider simple functions with a single col/param
    if not (col := columns and columns[0]):
        return None
    if (param_name := _param_name_from_signature(function)) is None:
        return None

    # fast function disassembly to get bytecode ops
    ops = _get_bytecode_ops(function)

    # if ops indicate a trivial function that should be native, warn about it
    if _can_rewrite_as_expression(ops, apply_target):
        if suggestion := _to_polars_expression(ops, col, apply_target, param_name):
            _generate_warning(function, suggestion, col, param_name)

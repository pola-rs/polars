"""Suggest native expressions for inefficient lambdas/functions used in `apply`."""
from __future__ import annotations

import dis
import sys
import warnings
from inspect import signature
from itertools import tee
from typing import Any, Callable, Iterable

from polars.exceptions import PolarsInefficientApplyWarning
from polars.utils.various import find_stacklevel

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
_SIMPLE_EXPR_OPS = {
    "BINARY_OP",
    "COMPARE_OP",
    "CONTAINS_OP",
    "IS_OP",
    "LOAD_CONST",
    "LOAD_FAST",
    "UNARY_NOT",
}
_SIMPLE_FRAME_OPS = _SIMPLE_EXPR_OPS | {"BINARY_SUBSCR"}


def _pairwise(values: Iterable[Any]) -> Iterable[tuple[Any, Any]]:
    a, b = tee(values)
    next(b, None)
    return zip(a, b)


def _expr(value: str | tuple[str, str, str], col: str | None) -> str:
    if isinstance(value, tuple):
        op = value[1]
        if len(value) == 2 and op == "not":
            return f"~{_expr(value[0], col)}"
        elif value[2] == "None" and op.startswith("is"):
            null_check = "is_not_null" if value[1].endswith("not") else "is_null"
            return f"{_expr(value[0], col)}.{null_check}()"
        elif op.endswith("in"):
            not_ = "~" if op.startswith("not") else ""
            return f"({not_}{_expr(value[0], col)}.is_in({_expr(value[2], col)}))"
        else:
            return f"({_expr(value[0], col)} {op} {_expr(value[2], col)})"

    elif col and value.isidentifier():
        if value not in ("True", "False", "None"):
            return f'pl.col("{col}")'

    return value


def _op(op: str, argrepr: str, argval: Any) -> str:
    if argrepr:
        return argrepr
    elif op == "IS_OP":
        return "is not" if argval else "is"
    elif op == "CONTAINS_OP":
        return "not in" if argval else "in"
    elif op == "UNARY_NOT":
        return "not"
    return "??"  # should be unreachable with current op list?


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
        return []  # can't disassemble non-native functions (eg: numpy/etc)


def _inst(opname: str, argrepr: str, argval: str) -> tuple[str, str, str]:
    """Rewrite older opcodes using py 3.11 'BINARY_OP' instead."""
    if sys.version_info < (3, 11) and opname in _BINARY_OPCODES:
        argrepr = _BINARY_OPCODES[opname]
        opname = "BINARY_OP"
    return opname, argrepr, argval


def _is_inefficient(ops: list[tuple[str, str, Any]], apply_target: str) -> bool:
    """Determine if bytecode indicates only simple binary ops and/or comparisons."""
    simple_ops = _SIMPLE_FRAME_OPS if apply_target == "frame" else _SIMPLE_EXPR_OPS
    return bool(ops) and all(op in simple_ops for op, _argrepr, _argval in ops)


def _rewrite_as_polars_expr(
    ops: list[tuple[str, str, Any]], col: str | None, apply_target: str
) -> str | None:
    """Take postfix opcode stack and translate to native polars expression."""
    # const / unchanged
    if len(ops) == 1:
        return _expr(ops[0][1], col)
    elif len(ops) >= 3:
        if apply_target == "expr":
            stack = []  # type: ignore[var-annotated]
            for op in ops:
                stack.append(
                    op[1]
                    if op[0].startswith("LOAD_")
                    else (
                        (stack.pop(), _op(*op))
                        if op[0] == "UNARY_NOT"
                        else (stack.pop(-2), _op(*op), stack.pop(-1))
                    )
                )
            return _expr(stack[0], col)  # type: ignore[arg-type]

    # TODO: frame apply (account for BINARY_SUBSCR)
    # TODO: series apply (rewrite col expr as series)
    return None


def _simple_signature(function: Callable[[Any], Any]) -> bool:
    try:
        return len(signature(function).parameters) == 1
    except ValueError:
        return False


def warn_on_inefficient_apply(
    function: Callable[[Any], Any], columns: list[str], apply_target: str
) -> None:
    """Generate ``PolarsInefficientApplyWarning`` on poor usage of ``apply`` func."""
    # only consider simple functions with a single col/param
    if _simple_signature(function):
        # fast function disassembly to get bytecode ops
        ops = _get_bytecode_ops(function)

        # if ops indicate a trivial function that should be native, warn about it
        if _is_inefficient(ops, apply_target):
            col = (columns and columns[0]) or None
            suggestion = _rewrite_as_polars_expr(ops, col, apply_target)
            if suggestion:
                warnings.warn(
                    "Expr.apply is significantly slower than the native expressions API.\n"
                    "Only use if you absolutely CANNOT implement your logic otherwise.\n"
                    "In this case, you can replace your function with an expression:\n"
                    f'\033[31m-  pl.col("{columns[0]}").apply({function!r})\033[0m\n'
                    f"\033[32m+  {suggestion}\033[0m\n",
                    PolarsInefficientApplyWarning,
                    stacklevel=find_stacklevel(),
                )

"""Utilities related to user defined functions (such as those passed to `apply`)."""
from __future__ import annotations

import dis
import sys
import warnings
from collections import defaultdict
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Tuple, Union

from polars.exceptions import PolarsInefficientApplyWarning
from polars.utils.various import find_stacklevel, in_terminal_that_supports_colour

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

ByteCodeInfo: TypeAlias = Tuple[str, str, Any, int]
StackValue: TypeAlias = Union[str, Tuple[str, str, str]]

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
_LOGICAL_OPCODES = {
    "POP_JUMP_FORWARD_IF_FALSE": "&",
    "POP_JUMP_FORWARD_IF_TRUE": "|",
    "POP_JUMP_IF_FALSE": "&",
    "POP_JUMP_IF_TRUE": "|",
    "JUMP_IF_FALSE_OR_POP": "&",
    "JUMP_IF_TRUE_OR_POP": "|",
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
_SIMPLE_EXPR_OPS |= set(_UNARY_OPCODES) | set(_LOGICAL_OPCODES)
_SIMPLE_FRAME_OPS = _SIMPLE_EXPR_OPS | {"BINARY_SUBSCR"}
_UPGRADE_BINARY_OPS = sys.version_info < (3, 11)

REPORT_MSG = (
    "Please report a bug at https://github.com/pola-rs/polars/issues, including "
    "the function which you passed to `apply`."
)


def _expr(value: StackValue, col: str, param_name: str, depth: int) -> str:
    """Take a stack entry value and convert to string, accounting for nested values."""
    if isinstance(value, tuple):
        op = value[1]
        unary_op = len(value) == 2
        e1 = _expr(value[0], col, param_name, depth + 1)
        e2 = _expr(value[2], col, param_name, depth + 1) if not unary_op else None

        if unary_op:
            return f"{op}{e1}"
        elif op in ("is", "is not") and value[2] == "None":
            not_ = "" if op == "is" else "not_"
            return f"{e1}.is_{not_}null()"
        elif op in ("in", "not in"):
            not_ = "" if op == "in" else "~"
            return f"{not_}({e1}.is_in({e2}))"
        else:
            expr = f"{e1} {op} {e2}"
            return f"({expr})" if depth else expr

    elif value == param_name:
        return f'pl.col("{col}")'

    return value


def _op(opname: str, argrepr: str, argval: Any, _offset: int) -> str:
    """Convert bytecode op to a distinct/suitable intermediate op string."""
    if opname in _LOGICAL_OPCODES:
        return _LOGICAL_OPCODES[opname]
    elif argrepr:
        return argrepr
    elif opname == "IS_OP":
        return "is not" if argval else "is"
    elif opname == "CONTAINS_OP":
        return "not in" if argval else "in"
    elif opname in _UNARY_OPCODES:
        return _UNARY_OPCODES[opname]
    else:
        raise AssertionError(
            "Unrecognised op - please report a bug to https://github.com/pola-rs/polars/issues "
            "with the content of function you were passing to `apply`."
        )


def _ops_to_stack_value(ops: list[ByteCodeInfo], apply_target: str) -> StackValue:
    """Take postfix bytecode and convert to an intermediate natural-order stack."""
    if apply_target == "expr":
        stack: list[StackValue] = []
        for op in ops:
            stack.append(
                op[1]  # type: ignore[arg-type]
                if op[0] in ("LOAD_FAST", "LOAD_CONST")
                else (
                    (stack.pop(), _op(*op))
                    if op[0].startswith("UNARY_")
                    else (stack.pop(-2), _op(*op), stack.pop(-1))
                )
            )

        return stack[0]

    # TODO: frame apply (account for BINARY_SUBSCR)
    # TODO: series apply (rewrite col expr as series)
    raise NotImplementedError(f"TODO: {apply_target!r} apply")


def _bytecode_to_expression(
    bytecode: list[ByteCodeInfo], col: str, apply_target: str, param_name: str
) -> str | None:
    """Take postfix bytecode and translate to polars expression string."""
    # decompose bytecode into logical 'and'/'or' expression blocks (if present)
    logical_blocks, logical_ops = defaultdict(list), []
    jump_offset = 0
    for idx, op in enumerate(bytecode):
        if op[0] in _LOGICAL_OPCODES:
            jump_offset = bytecode[idx + 1][3]
            logical_ops.append(op)
        else:
            logical_blocks[jump_offset].append(op)

    # convert each logical block to a polars expression string
    expression_strings = {
        offset: _expr(
            _ops_to_stack_value(ops, apply_target),
            col=col,
            param_name=param_name,
            depth=int(bool(logical_ops)),
        )
        for offset, ops in logical_blocks.items()
    }
    for op in logical_ops:
        expression_strings[op[3]] = _op(*op)

    # TODO: handle mixed 'and'/'or' blocks (e.g. `x > 0 AND (x != 1 OR (x % 2 == 0))`).
    #  (we need to reconstruct the correct nesting boundaries to do this properly...)

    exprs = sorted(expression_strings.items())
    return " ".join(expr for _, expr in exprs)


def _can_rewrite_as_expression(ops: list[ByteCodeInfo], apply_target: str) -> bool:
    """
    Determine if bytecode indicates only simple binary ops and/or comparisons.

    Note that `lambda x: x` is inefficient, but we ignore it because it is not
    guaranteed that using the equivalent bare constant value will return the
    same output. (Hopefully nobody is writing lambdas like that anyway...)
    """
    simple_ops = _SIMPLE_FRAME_OPS if apply_target == "frame" else _SIMPLE_EXPR_OPS
    if bool(ops) and all(op[0] in simple_ops for op in ops) and len(ops) >= 3:
        return (
            # can (currently) only handle logical 'and'/'or' if they are not mixed
            len({_LOGICAL_OPCODES[op[0]] for op in ops if op[0] in _LOGICAL_OPCODES})
            <= 1
        )
    return False


def _function_name(function: Callable[[Any], Any], param_name: str) -> str:
    """Return the name of the given function/method/lambda."""
    func_name = function.__name__
    if func_name == "<lambda>":
        return f"lambda {param_name}: ..."
    elif not func_name:
        return "..."
    return func_name


def _get_bytecode_ops(function: Callable[[Any], Any]) -> list[ByteCodeInfo]:
    """Return disassembled bytecode ops, arg-repr, and arg-specific value/flag."""
    try:
        instructions = list(dis.get_instructions(function))
        idx_last = len(instructions) - 1
        return [
            _upgrade_instruction(inst.opname, inst.argrepr, inst.argval, inst.offset)
            for idx, inst in enumerate(instructions)
            if (idx, inst.opname) not in ((0, "RESUME"), (idx_last, "RETURN_VALUE"))
        ]
    except TypeError:
        # in case we hit something that can't be disassembled (eg: code object
        # unavailable, like a bare numpy ufunc that isn't in a lambda/function)
        return []


def _upgrade_instruction(
    opname: str, argrepr: str, argval: Any, offset: int
) -> ByteCodeInfo:
    """Rewrite any older binary opcodes using py 3.11 'BINARY_OP' instead."""
    if _UPGRADE_BINARY_OPS and opname in _BINARY_OPCODES:
        argrepr = _BINARY_OPCODES[opname]
        opname = "BINARY_OP"
    return opname, argrepr, argval, offset


def _param_name_from_signature(function: Callable[[Any], Any]) -> str | None:
    """Return param name from single-argument functions."""
    try:
        sig = signature(function)
    except ValueError:
        return None
    if len(parameters := sig.parameters) == 1:
        return next(iter(parameters.keys()))
    return None


def _generate_warning(
    function: Callable[[Any], Any], suggestion: str, col: str, param_name: str
) -> None:
    """Create a warning that includes a faster native expression as an alternative."""
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
        The column names of the original object; in the case of an ``Expr`` this
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
    bytecode = _get_bytecode_ops(function)

    # if ops indicate a trivial function that should be native, warn about it
    if _can_rewrite_as_expression(bytecode, apply_target):
        if suggestion := _bytecode_to_expression(
            bytecode, col, apply_target, param_name
        ):
            _generate_warning(function, suggestion, col, param_name)

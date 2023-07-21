"""Utilities related to user defined functions (such as those passed to `apply`)."""
from __future__ import annotations

import dis
import sys
import warnings
from collections import defaultdict
from dis import get_instructions
from inspect import signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, NamedTuple, Union

from polars.exceptions import PolarsInefficientApplyWarning
from polars.utils.various import find_stacklevel, in_terminal_that_supports_colour

if TYPE_CHECKING:
    from dis import Instruction

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias


class StackValue(NamedTuple):
    operator: str
    operator_arity: int
    left_operand: str
    right_operand: str


ApplyTarget: TypeAlias = Literal["expr", "frame", "series"]
StackEntry: TypeAlias = Union[str, StackValue]


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
_SYNTHETIC_OPS = {
    "POLARS_EXPRESSION",
}
_SIMPLE_EXPR_OPS = {
    "BINARY_OP",
    "COMPARE_OP",
    "CONTAINS_OP",
    "IS_OP",
    "LOAD_CONST",
    "LOAD_DEREF",
    "LOAD_FAST",
    "LOAD_GLOBAL",
}
_SIMPLE_EXPR_OPS |= set(_UNARY_OPCODES) | set(_LOGICAL_OPCODES) | _SYNTHETIC_OPS
_SIMPLE_FRAME_OPS = _SIMPLE_EXPR_OPS | {"BINARY_SUBSCR"}
_UNARY_OPCODE_VALUES = set(_UNARY_OPCODES.values())
_UPGRADE_BINARY_OPS = sys.version_info < (3, 11)

# numpy functions that we can map to a native expression
_NUMPY_MODULE_ALIASES = {"np", "numpy"}
_NUMPY_FUNCTIONS = {"cbrt", "cos", "cosh", "sin", "sinh", "sqrt", "tan", "tanh"}

# python function that we can map to a native expression
_PYFUNCTION_MAP = {
    "lower": "str.to_lowercase",
    "title": "str.to_titlecase",
    "upper": "str.to_uppercase",
}


class BytecodeParser:
    """Introspect UDF bytecode and determine if we can rewrite as native expression."""

    _can_rewrite: dict[str, bool]

    def __init__(self, function: Callable[[Any], Any], apply_target: ApplyTarget):
        try:
            original_instructions = get_instructions(function)
        except TypeError:
            # in case we hit something that can't be disassembled (eg: code object
            # unavailable, like a bare numpy ufunc that isn't in a lambda/function)
            original_instructions = iter([])

        self._can_rewrite = {}
        self._function = function
        self._apply_target = apply_target
        self._param_name = self._get_param_name(function)
        self._rewritten_instructions = RewrittenInstructions(
            instructions=original_instructions,
        )

    @staticmethod
    def _get_param_name(function: Callable[[Any], Any]) -> str | None:
        """Return single function parameter name."""
        try:
            # note: we do not parse/handle functions with > 1 params
            sig = signature(function)
            return (
                next(iter(parameters.keys()))
                if len(parameters := sig.parameters) == 1
                else None
            )
        except ValueError:
            return None

    @property
    def apply_target(self) -> ApplyTarget:
        """The apply target, eg: one of 'expr', 'frame', or 'series'."""
        return self._apply_target

    def can_rewrite(self) -> bool:
        """
        Determine if bytecode indicates only simple binary ops and/or comparisons.

        Note that `lambda x: x` is inefficient, but we ignore it because it is not
        guaranteed that using the equivalent bare constant value will return the
        same output. (Hopefully nobody is writing lambdas like that anyway...)
        """
        if (can_rewrite := self._can_rewrite.get(self._apply_target, None)) is not None:
            return can_rewrite
        else:
            self._can_rewrite[self._apply_target] = False
            if self._rewritten_instructions and self._param_name is not None:
                simple_ops = (
                    _SIMPLE_FRAME_OPS
                    if self._apply_target == "frame"
                    else _SIMPLE_EXPR_OPS
                )
                if len(self._rewritten_instructions) >= 2 and all(
                    inst.opname in simple_ops for inst in self._rewritten_instructions
                ):
                    # can (currently) only handle logical 'and'/'or' if they not mixed
                    logical_ops = {
                        _LOGICAL_OPCODES[inst.opname]
                        for inst in self._rewritten_instructions
                        if inst.opname in _LOGICAL_OPCODES
                    }
                    self._can_rewrite[self._apply_target] = len(logical_ops) <= 1

        return self._can_rewrite[self._apply_target]

    def dis(self) -> None:
        """Print disassembled function bytecode."""
        dis.dis(self._function)

    @property
    def function(self) -> Callable[[Any], Any]:
        """The function being parsed."""
        return self._function

    @property
    def original_instructions(self) -> list[Instruction]:
        """The original bytecode instructions from the function we are parsing."""
        return list(get_instructions(self._function))

    @property
    def param_name(self) -> str | None:
        """The parameter name of the function being parsed."""
        return self._param_name

    @property
    def rewritten_instructions(self) -> list[Instruction]:
        """The rewritten bytecode instructions from the function we are parsing."""
        return list(self._rewritten_instructions)

    def to_expression(self, col: str) -> str | None:
        """Translate postfix bytecode instructions to polars expression string."""
        if not self.can_rewrite() or self._param_name is None:
            return None

        # decompose bytecode into logical 'and'/'or' expression blocks (if present)
        logical_instruction_blocks = defaultdict(list)
        logical_instructions = []
        jump_offset = 0
        for idx, inst in enumerate(self._rewritten_instructions):
            if inst.opname in _LOGICAL_OPCODES:
                jump_offset = self._rewritten_instructions[idx + 1].offset
                logical_instructions.append(inst)
            else:
                logical_instruction_blocks[jump_offset].append(inst)

        # convert each logical block to a polars expression string
        expression_strings = {
            offset: InstructionTranslator(
                instructions=ops,
                apply_target=self._apply_target,
            ).to_expression(
                col=col,
                param_name=self._param_name,
                depth=int(bool(logical_instructions)),
            )
            for offset, ops in logical_instruction_blocks.items()
        }
        for inst in logical_instructions:
            expression_strings[inst.offset] = InstructionTranslator.op(inst)

        # TODO: handle mixed 'and'/'or' blocks (e.g. `x > 0 AND (x > 0 OR (x == -1)`).
        #  (need to reconstruct the correct nesting boundaries to do this properly)

        exprs = sorted(expression_strings.items())
        polars_expr = " ".join(expr for _, expr in exprs)

        # note: if no 'pl.col' in the expression, it likely represents a compound
        # constant value (e.g. `lambda x: CONST + 123`), so we don't want to warn
        if "pl.col(" not in polars_expr:
            return None

        return polars_expr

    def warn(
        self,
        col: str,
        suggestion_override: str | None = None,
        func_name_override: str | None = None,
    ) -> None:
        """Generate warning that suggests an equivalent native polars expression."""
        suggested_expression = suggestion_override or self.to_expression(col)
        if suggested_expression is not None:
            func_name = func_name_override or self._function.__name__ or "..."
            if func_name == "<lambda>":
                func_name = f"lambda {self._param_name}: ..."

            addendum = (
                'Note: in list.eval context, pl.col("") should be written as pl.element()'
                if 'pl.col("")' in suggested_expression
                else ""
            )
            before_after_suggestion = (
                (
                    f'  \033[31m-  pl.col("{col}").apply({func_name})\033[0m\n'
                    f"  \033[32m+  {suggested_expression}\033[0m\n{addendum}"
                )
                if in_terminal_that_supports_colour()
                else (
                    f'  - pl.col("{col}").apply({func_name})\n'
                    f"  + {suggested_expression}\n{addendum}"
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


class InstructionTranslator:
    """Translates Instruction bytecode to a polars expression string."""

    def __init__(self, instructions: list[Instruction], apply_target: ApplyTarget):
        self._stack = self._to_intermediate_stack(instructions, apply_target)

    def to_expression(self, col: str, param_name: str, depth: int) -> str:
        """Convert intermediate stack to polars expression string."""
        return self._expr(self._stack, col, param_name, depth)

    @classmethod
    def op(cls, inst: Instruction) -> str:
        """Convert bytecode instruction to suitable intermediate op string."""
        if inst.opname in _LOGICAL_OPCODES:
            return _LOGICAL_OPCODES[inst.opname]
        elif inst.argrepr:
            return inst.argrepr
        elif inst.opname == "IS_OP":
            return "is not" if inst.argval else "is"
        elif inst.opname == "CONTAINS_OP":
            return "not in" if inst.argval else "in"
        elif inst.opname in _UNARY_OPCODES:
            return _UNARY_OPCODES[inst.opname]
        else:
            raise AssertionError(
                "Unrecognised opname; please report a bug to https://github.com/pola-rs/polars/issues "
                "with the content of function you were passing to `apply` and the "
                f"following instruction object:\n{inst}"
            )

    @classmethod
    def _expr(cls, value: StackEntry, col: str, param_name: str, depth: int) -> str:
        """Take stack entry value and convert to polars expression string."""
        if isinstance(value, StackValue):
            op = value.operator
            e1 = cls._expr(value.left_operand, col, param_name, depth + 1)
            if value.operator_arity == 1:
                if op not in _UNARY_OPCODE_VALUES:
                    return f"{e1}.{op}()"
                return f"{op}{e1}"
            else:
                e2 = cls._expr(value.right_operand, col, param_name, depth + 1)
                if op in ("is", "is not") and value[2] == "None":
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

    def _to_intermediate_stack(
        self, instructions: list[Instruction], apply_target: ApplyTarget
    ) -> StackEntry:
        """Take postfix bytecode and convert to an intermediate natural-order stack."""
        if apply_target == "expr":
            stack: list[StackEntry] = []
            for inst in instructions:
                stack.append(
                    inst.argrepr
                    if inst.opname.startswith("LOAD_")
                    else (
                        StackValue(
                            operator=self.op(inst),
                            operator_arity=1,
                            left_operand=stack.pop(),  # type: ignore[arg-type]
                            right_operand=None,  # type: ignore[arg-type]
                        )
                        if (
                            inst.opname in _UNARY_OPCODES
                            or inst.opname in _SYNTHETIC_OPS
                        )
                        else StackValue(
                            operator=self.op(inst),
                            operator_arity=2,
                            left_operand=stack.pop(-2),  # type: ignore[arg-type]
                            right_operand=stack.pop(-1),  # type: ignore[arg-type]
                        )
                    )
                )
            return stack[0]

        # TODO: frame apply (account for BINARY_SUBSCR)
        # TODO: series apply (rewrite col expr as series)
        raise NotImplementedError(f"TODO: {apply_target!r} apply")


class RewrittenInstructions:
    """
    Standalone class that applies Instruction rewrite/filtering rules.

    This significantly simplifies subsequent parsing by injecting
    synthetic POLARS_EXPRESSION ops into the Instruction stream for
    easy identification/translation and separates the parsing logic
    from the identification of expression translation opportunities.
    """

    def __init__(self, instructions: Iterator[Instruction]):
        self._instructions = instructions
        self._rewritten_instructions = self._apply_rules(
            self._upgrade_instruction(inst)
            for inst in instructions
            if inst.opname
            not in ("COPY_FREE_VARS", "PRECALL", "RESUME", "RETURN_VALUE")
        )

    def __len__(self) -> int:
        return len(self._rewritten_instructions)

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._rewritten_instructions)

    def __getitem__(self, item: Any) -> Instruction:
        return self._rewritten_instructions[item]

    def _apply_rules(self, instructions: Iterator[Instruction]) -> list[Instruction]:
        """
        Apply rewrite rules, potentially injecting synthetic operations.

        Rules operate on the instruction stream and can examine/modify
        it as needed, pushing updates into "updated_instructions" and
        returning True/False to indicate if any changes were made.
        """
        self._instructions = instructions
        updated_instructions: list[Instruction] = []
        for inst in instructions:
            if not any(
                apply_rewrite(inst, updated_instructions)
                for apply_rewrite in (
                    # add any other rewrite methods here
                    self._numpy_functions,
                    self._python_functions,
                )
            ):
                updated_instructions.append(inst)
        return updated_instructions

    def _numpy_functions(
        self, inst: Instruction, instructions: list[Instruction]
    ) -> bool:
        """Replace numpy function calls with a synthetic POLARS_EXPRESSION op."""
        if inst.opname == "LOAD_GLOBAL" and inst.argval in _NUMPY_MODULE_ALIASES:
            instruction_buffer = list(islice(self._instructions, 3))
            if (
                len(instruction_buffer) == 3
                and instruction_buffer[0].argval in _NUMPY_FUNCTIONS
                and instruction_buffer[1].opname.startswith("LOAD_")
                and instruction_buffer[2].opname.startswith("CALL")
            ):
                # note: synthetic POLARS_EXPRESSION is mapped as a unary
                # op, so we switch the instruction order on injection
                expr_name = instruction_buffer[0].argval
                offsets = inst.offset, instruction_buffer[1].offset
                synthetic_call = inst._replace(
                    opname="POLARS_EXPRESSION",
                    argval=expr_name,
                    argrepr=expr_name,
                    offset=offsets[1],
                )
                operand = instruction_buffer[1]._replace(offset=offsets[0])
                instructions.extend((operand, synthetic_call))
            else:
                instructions.append(inst)
                instructions.extend(instruction_buffer)
            return True
        return False

    def _python_functions(
        self, inst: Instruction, instructions: list[Instruction]
    ) -> bool:
        """Replace python method calls with synthetic POLARS_EXPRESSION op."""
        if inst.opname == "LOAD_METHOD" and inst.argval in _PYFUNCTION_MAP:
            if (
                instruction_buffer := list(islice(self._instructions, 1))
            ) and instruction_buffer[0].opname.startswith("CALL"):
                expr_name = _PYFUNCTION_MAP[inst.argval]
                synthetic_call = inst._replace(
                    opname="POLARS_EXPRESSION",
                    argval=expr_name,
                    argrepr=expr_name,
                )
                instructions.append(synthetic_call)
            else:
                instructions.append(inst)
                instructions.extend(instruction_buffer)
            return True
        return False

    @staticmethod
    def _upgrade_instruction(inst: Instruction) -> Instruction:
        """Rewrite any older binary opcodes using py 3.11 'BINARY_OP' instead."""
        if _UPGRADE_BINARY_OPS and inst.opname in _BINARY_OPCODES:
            inst = inst._replace(
                argrepr=_BINARY_OPCODES[inst.opname],
                opname="BINARY_OP",
            )
        return inst


def _is_raw_numpy_function(function: Callable[[Any], Any]) -> bool:
    """Identify numpy calls that are not wrapped in a lambda/function."""
    try:
        return (
            function.__class__.__module__ == "numpy"
            and function.__name__ in _NUMPY_FUNCTIONS
        )
    except AttributeError:
        return False


def warn_on_inefficient_apply(
    function: Callable[[Any], Any], columns: list[str], apply_target: ApplyTarget
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
    if apply_target in ("frame", "series"):
        raise NotImplementedError("TODO: 'frame' and 'series' apply-function parsing")

    # note: we only consider simple functions with a single col/param
    if not (col := columns and columns[0]):
        return None

    # the parser introspects function bytecode to determine if we can
    # rewrite as a much more optimal native polars expression instead
    parser = BytecodeParser(function, apply_target)
    if parser.can_rewrite():
        parser.warn(col)
    elif _is_raw_numpy_function(function):
        fn = function.__name__
        suggestion = f'pl.col("{col}").{fn}()'
        parser.warn(col, suggestion_override=suggestion, func_name_override=f"np.{fn}")


__all__ = [
    "BytecodeParser",
    "warn_on_inefficient_apply",
]

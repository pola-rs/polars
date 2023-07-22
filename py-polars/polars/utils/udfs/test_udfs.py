"""
Minimal test of the BytecodeParser class.

This can be run without polars installed, and so can be easily run in CI
over all supported Python versions.

All that needs to be installed is numpy and pytest.

Usage:

    $ PYTHONPATH=. pytest test_udfs.py
"""
from typing import Any, Callable

import pytest
from test_cases import TEST_CASES  # type: ignore[import]
from udfs import BytecodeParser  # type: ignore[import]


@pytest.mark.parametrize(
    ("col", "func", "expected"),
    TEST_CASES,
)
def test_bytecode_parser_expression(  # noqa: D103
    col: str, func: Callable[[Any], Any], expected: str
) -> None:
    bytecode_parser = BytecodeParser(func, apply_target="expr")
    result = bytecode_parser.to_expression(col)
    assert result == expected

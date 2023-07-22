"""
Minimal test of the BytecodeParser class.

This can be run without polars installed, and so can be easily run in CI
over all supported Python versions.

All that needs to be installed is numpy and pytest.

Usage:

    $ PYTHONPATH=. pytest tests/
"""
import pytest
from test_cases import TEST_CASES
from udfs import BytecodeParser


@pytest.mark.parametrize(
    ("col", "func", "expected"),
    TEST_CASES,
)
def test_me(col, func, expected):  # noqa: D103
    bytecode_parser = BytecodeParser(func, apply_target="expr")
    result = bytecode_parser.to_expression(col)
    assert result == expected

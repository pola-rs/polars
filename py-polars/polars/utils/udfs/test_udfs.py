"""
Minimal test of the BytecodeParser class.

This can be run without polars installed, and so can be easily run in CI
over all supported Python versions.

All that needs to be installed is numpy and pytest.

Usage:

    $ PYTHONPATH=. pytest tests/
"""
import json
from udfs import BytecodeParser
from test_cases import TEST_CASES
import numpy as np
import pytest

@pytest.mark.parametrize(
    ("col", "func", "expected"),
    TEST_CASES,
)
def test_me(col, func, expected):
    bytecode_parser = BytecodeParser(func, apply_target="expr")
    result = bytecode_parser.to_expression(col)
    assert result == expected

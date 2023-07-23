"""
Minimal test of the BytecodeParser class.

This can be run without polars installed, and so can be easily run in CI
over all supported Python versions.

All that needs to be installed is numpy and pytest.

Usage:

    $ PYTHONPATH=polars/utils:polars/testing/udfs pytest tests/test_udfs.py

Running it without `PYTHONPATH` set will result in the test being skipped.
"""
from typing import Any, Callable

import pytest

udfs = pytest.importorskip("udfs")
test_cases = pytest.importorskip("test_cases")


@pytest.mark.parametrize(
    ("col", "func", "expected"),
    test_cases.TEST_CASES,
)
def test_bytecode_parser_expression(
    col: str, func: Callable[[Any], Any], expected: str
) -> None:
    bytecode_parser = udfs.BytecodeParser(func, apply_target="expr")
    result = bytecode_parser.to_expression(col)
    assert result == expected

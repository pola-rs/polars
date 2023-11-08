"""
Minimal testing script of the BytecodeParser class.

This can be run without polars installed, and so can be easily run in CI
over all supported Python versions.

All that needs to be installed is pytest, numpy, and ipython.

Usage:

    $ PYTHONPATH=py-polars pytest .github/scripts/test_bytecode_parser.py

Running it without `PYTHONPATH` set will result in the test failing.
"""
import datetime as dt  # noqa: F401
import subprocess
from datetime import datetime  # noqa: F401
from typing import Any, Callable

import pytest
from polars.utils.udfs import BytecodeParser
from tests.unit.operations.map.test_inefficient_map_warning import (
    MY_DICT,
    NOOP_TEST_CASES,
    TEST_CASES,
)


@pytest.mark.parametrize(
    ("col", "func", "expected"),
    TEST_CASES,
)
def test_bytecode_parser_expression(col: str, func: str, expected: str) -> None:
    bytecode_parser = BytecodeParser(eval(func), map_target="expr")
    result = bytecode_parser.to_expression(col)
    assert result == expected


@pytest.mark.parametrize(
    ("col", "func", "expected"),
    TEST_CASES,
)
def test_bytecode_parser_expression_in_ipython(
    col: str, func: Callable[[Any], Any], expected: str
) -> None:
    script = (
        "from polars.utils.udfs import BytecodeParser; "
        "import datetime as dt; "
        "from datetime import datetime; "
        "import numpy as np; "
        "import json; "
        f"MY_DICT = {MY_DICT};"
        f'bytecode_parser = BytecodeParser({func}, map_target="expr");'
        f'print(bytecode_parser.to_expression("{col}"));'
    )

    output = subprocess.run(["ipython", "-c", script], text=True, capture_output=True)
    assert expected == output.stdout.rstrip("\n")


@pytest.mark.parametrize(
    "func",
    NOOP_TEST_CASES,
)
def test_bytecode_parser_expression_noop(func: str) -> None:
    parser = BytecodeParser(eval(func), map_target="expr")
    assert not parser.can_attempt_rewrite() or not parser.to_expression("x")


@pytest.mark.parametrize(
    "func",
    NOOP_TEST_CASES,
)
def test_bytecode_parser_expression_noop_in_ipython(func: str) -> None:
    script = (
        "from polars.utils.udfs import BytecodeParser; "
        f"MY_DICT = {MY_DICT};"
        f'parser = BytecodeParser({func}, map_target="expr");'
        f'print(not parser.can_attempt_rewrite() or not parser.to_expression("x"));'
    )

    output = subprocess.run(["ipython", "-c", script], text=True, capture_output=True)
    assert output.stdout == "True\n"


def test_local_imports() -> None:
    import datetime as dt  # noqa: F811
    import json

    bytecode_parser = BytecodeParser(lambda x: json.loads(x), map_target="expr")
    result = bytecode_parser.to_expression("x")
    expected = 'pl.col("x").str.json_extract()'
    assert result == expected

    bytecode_parser = BytecodeParser(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d"), map_target="expr"
    )
    result = bytecode_parser.to_expression("x")
    expected = 'pl.col("x").str.to_datetime(format="%Y-%m-%d")'
    assert result == expected

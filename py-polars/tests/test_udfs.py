"""
Minimal test of the BytecodeParser class.

This can be run without polars installed, and so can be easily run in CI
over all supported Python versions.

All that needs to be installed is numpy and pytest.

Usage:

    $ PYTHONPATH=polars/utils pytest tests/test_udfs.py

Running it without `PYTHONPATH` set will result in the test being skipped.
"""
import datetime as dt  # noqa: F401
import subprocess
from datetime import datetime  # noqa: F401
from typing import Any, Callable

import pytest

MY_CONSTANT = 3
MY_DICT = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
MY_LIST = [1, 2, 3]


# column_name, function, expected_suggestion
TEST_CASES = [
    # ---------------------------------------------
    # numeric expr: math, comparison, logic ops
    # ---------------------------------------------
    ("a", "lambda x: x + 1 - (2 / 3)", '(pl.col("a") + 1) - 0.6666666666666666'),
    ("a", "lambda x: x // 1 % 2", '(pl.col("a") // 1) % 2'),
    ("a", "lambda x: x & True", 'pl.col("a") & True'),
    ("a", "lambda x: x | False", 'pl.col("a") | False'),
    ("a", "lambda x: abs(x) != 3", 'pl.col("a").abs() != 3'),
    ("a", "lambda x: int(x) > 1", 'pl.col("a").cast(pl.Int64) > 1'),
    ("a", "lambda x: not (x > 1) or x == 2", '~(pl.col("a") > 1) | (pl.col("a") == 2)'),
    ("a", "lambda x: x is None", 'pl.col("a") is None'),
    ("a", "lambda x: x is not None", 'pl.col("a") is not None'),
    (
        "a",
        "lambda x: ((x * -x) ** x) * 1.0",
        '((pl.col("a") * -pl.col("a")) ** pl.col("a")) * 1.0',
    ),
    (
        "a",
        "lambda x: 1.0 * (x * (x**x))",
        '1.0 * (pl.col("a") * (pl.col("a") ** pl.col("a")))',
    ),
    (
        "a",
        "lambda x: (x / x) + ((x * x) - x)",
        '(pl.col("a") / pl.col("a")) + ((pl.col("a") * pl.col("a")) - pl.col("a"))',
    ),
    (
        "a",
        "lambda x: (10 - x) / (((x * 4) - x) // (2 + (x * (x - 1))))",
        '(10 - pl.col("a")) / (((pl.col("a") * 4) - pl.col("a")) // (2 + (pl.col("a") * (pl.col("a") - 1))))',
    ),
    ("a", "lambda x: x in (2, 3, 4)", 'pl.col("a").is_in((2, 3, 4))'),
    ("a", "lambda x: x not in (2, 3, 4)", '~pl.col("a").is_in((2, 3, 4))'),
    (
        "a",
        "lambda x: x in (1, 2, 3, 4, 3) and x % 2 == 0 and x > 0",
        'pl.col("a").is_in((1, 2, 3, 4, 3)) & ((pl.col("a") % 2) == 0) & (pl.col("a") > 0)',
    ),
    ("a", "lambda x: MY_CONSTANT + x", 'MY_CONSTANT + pl.col("a")'),
    ("a", "lambda x: 0 + numpy.cbrt(x)", '0 + pl.col("a").cbrt()'),
    ("a", "lambda x: np.sin(x) + 1", 'pl.col("a").sin() + 1'),
    (
        "a",  # note: functions operate on consts
        "lambda x: np.sin(3.14159265358979) + (x - 1) + abs(-3)",
        '(np.sin(3.14159265358979) + (pl.col("a") - 1)) + abs(-3)',
    ),
    (
        "a",
        "lambda x: (float(x) * int(x)) // 2",
        '(pl.col("a").cast(pl.Float64) * pl.col("a").cast(pl.Int64)) // 2',
    ),
    # ---------------------------------------------
    # logical 'and/or' (validate nesting levels)
    # ---------------------------------------------
    (
        "a",
        "lambda x: x > 1 or (x == 1 and x == 2)",
        '(pl.col("a") > 1) | (pl.col("a") == 1) & (pl.col("a") == 2)',
    ),
    (
        "a",
        "lambda x: (x > 1 or x == 1) and x == 2",
        '((pl.col("a") > 1) | (pl.col("a") == 1)) & (pl.col("a") == 2)',
    ),
    (
        "a",
        "lambda x: x > 2 or x != 3 and x not in (0, 1, 4)",
        '(pl.col("a") > 2) | (pl.col("a") != 3) & ~pl.col("a").is_in((0, 1, 4))',
    ),
    (
        "a",
        "lambda x: x > 1 and x != 2 or x % 2 == 0 and x < 3",
        '(pl.col("a") > 1) & (pl.col("a") != 2) | ((pl.col("a") % 2) == 0) & (pl.col("a") < 3)',
    ),
    (
        "a",
        "lambda x: x > 1 and (x != 2 or x % 2 == 0) and x < 3",
        '(pl.col("a") > 1) & ((pl.col("a") != 2) | ((pl.col("a") % 2) == 0)) & (pl.col("a") < 3)',
    ),
    # ---------------------------------------------
    # string expr: case/cast ops
    # ---------------------------------------------
    ("b", "lambda x: str(x).title()", 'pl.col("b").cast(pl.Utf8).str.to_titlecase()'),
    (
        "b",
        'lambda x: x.lower() + ":" + x.upper() + ":" + x.title()',
        '(((pl.col("b").str.to_lowercase() + \':\') + pl.col("b").str.to_uppercase()) + \':\') + pl.col("b").str.to_titlecase()',
    ),
    # ---------------------------------------------
    # json expr: load/extract
    # ---------------------------------------------
    ("c", "lambda x: json.loads(x)", 'pl.col("c").str.json_extract()'),
    # ---------------------------------------------
    # map_dict
    # ---------------------------------------------
    ("a", "lambda x: MY_DICT[x]", 'pl.col("a").map_dict(MY_DICT)'),
    (
        "a",
        "lambda x: MY_DICT[x - 1] + MY_DICT[1 + x]",
        '(pl.col("a") - 1).map_dict(MY_DICT) + (1 + pl.col("a")).map_dict(MY_DICT)',
    ),
    # ---------------------------------------------
    # standard library datetime parsing
    # ---------------------------------------------
    (
        "d",
        'lambda x: datetime.strptime(x, "%Y-%m-%d")',
        'pl.col("d").str.to_datetime(format="%Y-%m-%d")',
    ),
    (
        "d",
        'lambda x: dt.datetime.strptime(x, "%Y-%m-%d")',
        'pl.col("d").str.to_datetime(format="%Y-%m-%d")',
    ),
]

NOOP_TEST_CASES = [
    "lambda x: x",
    "lambda x, y: x + y",
    "lambda x: x[0] + 1",
    "lambda x: MY_LIST[x]",
    "lambda x: MY_DICT[1]",
    'lambda x: "first" if x == 1 else "not first"',
]


@pytest.mark.parametrize(
    ("col", "func", "expected"),
    TEST_CASES,
)
def test_bytecode_parser_expression(col: str, func: str, expected: str) -> None:
    try:
        import udfs  # type: ignore[import]
    except ModuleNotFoundError as exc:
        assert "No module named 'udfs'" in str(exc)  # noqa: PT017
        # Skip test if udfs can't be imported because it's not in the path.
        # Prefer this over importorskip, so that if `udfs` can't be
        # imported for some other reason, then the test
        # won't be skipped.
        return

    bytecode_parser = udfs.BytecodeParser(eval(func), map_target="expr")
    result = bytecode_parser.to_expression(col)
    assert result == expected


@pytest.mark.parametrize(
    ("col", "func", "expected"),
    TEST_CASES,
)
def test_bytecode_parser_expression_in_ipython(
    col: str, func: Callable[[Any], Any], expected: str
) -> None:
    try:
        import udfs  # noqa: F401
    except ModuleNotFoundError as exc:
        assert "No module named 'udfs'" in str(exc)  # noqa: PT017
        # Skip test if udfs can't be imported because it's not in the path.
        # Prefer this over importorskip, so that if `udfs` can't be
        # imported for some other reason, then the test
        # won't be skipped.
        return

    script = (
        "import udfs; "
        "import datetime as dt; "
        "from datetime import datetime; "
        "import numpy as np; "
        "import json; "
        f"MY_DICT = {MY_DICT};"
        f'bytecode_parser = udfs.BytecodeParser({func}, map_target="expr");'
        f'print(bytecode_parser.to_expression("{col}"));'
    )

    output = subprocess.run(["ipython", "-c", script], text=True, capture_output=True)
    assert expected == output.stdout.rstrip("\n")


@pytest.mark.parametrize(
    "func",
    NOOP_TEST_CASES,
)
def test_bytecode_parser_expression_noop(func: str) -> None:
    try:
        import udfs
    except ModuleNotFoundError as exc:
        assert "No module named 'udfs'" in str(exc)  # noqa: PT017
        # Skip test if udfs can't be imported because it's not in the path.
        # Prefer this over importorskip, so that if `udfs` can't be
        # imported for some other reason, then the test
        # won't be skipped.
        return

    parser = udfs.BytecodeParser(eval(func), map_target="expr")
    assert not parser.can_attempt_rewrite() or not parser.to_expression("x")


@pytest.mark.parametrize(
    "func",
    NOOP_TEST_CASES,
)
def test_bytecode_parser_expression_noop_in_ipython(func: str) -> None:
    try:
        import udfs  # noqa: F401
    except ModuleNotFoundError as exc:
        assert "No module named 'udfs'" in str(exc)  # noqa: PT017
        # Skip test if udfs can't be imported because it's not in the path.
        # Prefer this over importorskip, so that if `udfs` can't be
        # imported for some other reason, then the test
        # won't be skipped.
        return

    script = (
        "import udfs; "
        f"MY_DICT = {MY_DICT};"
        f'parser = udfs.BytecodeParser({func}, map_target="expr");'
        f'print(not parser.can_attempt_rewrite() or not parser.to_expression("x"));'
    )

    output = subprocess.run(["ipython", "-c", script], text=True, capture_output=True)
    assert output.stdout == "True\n"


def test_local_imports() -> None:
    try:
        import udfs
    except ModuleNotFoundError as exc:
        assert "No module named 'udfs'" in str(exc)  # noqa: PT017
        return
    import datetime as dt  # noqa: F811
    import json

    bytecode_parser = udfs.BytecodeParser(lambda x: json.loads(x), map_target="expr")
    result = bytecode_parser.to_expression("x")
    expected = 'pl.col("x").str.json_extract()'
    assert result == expected

    bytecode_parser = udfs.BytecodeParser(
        lambda x: dt.datetime.strptime(x, "%Y-%m-%d"), map_target="expr"
    )
    result = bytecode_parser.to_expression("x")
    expected = 'pl.col("x").str.to_datetime(format="%Y-%m-%d")'
    assert result == expected

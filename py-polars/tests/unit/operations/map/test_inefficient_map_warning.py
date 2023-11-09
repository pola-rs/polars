from __future__ import annotations

import datetime as dt
import json
import re
from datetime import datetime
from typing import Any, Callable

import numpy
import numpy as np  # noqa: F401
import pytest

import polars as pl
from polars.exceptions import PolarsInefficientMapWarning
from polars.testing import assert_frame_equal, assert_series_equal
from polars.utils.udfs import _NUMPY_FUNCTIONS, BytecodeParser
from polars.utils.various import in_terminal_that_supports_colour

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

EVAL_ENVIRONMENT = {
    "np": numpy,
    "pl": pl,
    "MY_CONSTANT": MY_CONSTANT,
    "MY_DICT": MY_DICT,
    "MY_LIST": MY_LIST,
    "dt": dt,
    "datetime": datetime,
}


@pytest.mark.parametrize(
    "func",
    NOOP_TEST_CASES,
)
def test_parse_invalid_function(func: str) -> None:
    # functions we don't (yet?) offer suggestions for
    parser = BytecodeParser(eval(func), map_target="expr")
    assert not parser.can_attempt_rewrite() or not parser.to_expression("x")


@pytest.mark.parametrize(
    ("col", "func", "expr_repr"),
    TEST_CASES,
)
def test_parse_apply_functions(col: str, func: str, expr_repr: str) -> None:
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Expr\.map_elements.*In this case, you can replace",
    ):
        parser = BytecodeParser(eval(func), map_target="expr")
        suggested_expression = parser.to_expression(col)
        assert suggested_expression == expr_repr

        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["AB", "cd", "eF"],
                "c": ['{"a": 1}', '{"b": 2}', '{"c": 3}'],
                "d": ["2020-01-01", "2020-01-02", "2020-01-03"],
            }
        )
        result_frame = df.select(
            x=col,
            y=eval(suggested_expression, EVAL_ENVIRONMENT),
        )
        expected_frame = df.select(
            x=pl.col(col),
            y=pl.col(col).apply(eval(func)),
        )
        assert_frame_equal(result_frame, expected_frame)


def test_parse_apply_raw_functions() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    # test bare 'numpy' functions
    for func_name in _NUMPY_FUNCTIONS:
        func = getattr(numpy, func_name)

        # note: we can't parse/rewrite raw numpy functions...
        parser = BytecodeParser(func, map_target="expr")
        assert not parser.can_attempt_rewrite()

        # ...but we ARE still able to warn
        with pytest.warns(
            PolarsInefficientMapWarning,
            match=rf"(?s)Expr\.map_elements.*In this case, you can replace.*np\.{func_name}",
        ):
            df1 = lf.select(pl.col("a").map_elements(func)).collect()
            df2 = lf.select(getattr(pl.col("a"), func_name)()).collect()
            assert_frame_equal(df1, df2)

    # test bare 'json.loads'
    result_frames = []
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Expr\.map_elements.*In this case, you can replace.*\.str\.json_extract",
    ):
        for expr in (
            pl.col("value").str.json_extract(),
            pl.col("value").map_elements(json.loads),
        ):
            result_frames.append(
                pl.LazyFrame({"value": ['{"a":1, "b": true, "c": "xx"}', None]})
                .select(extracted=expr)
                .unnest("extracted")
                .collect()
            )

    assert_frame_equal(*result_frames)

    # test primitive python casts
    for py_cast, pl_dtype in ((str, pl.Utf8), (int, pl.Int64), (float, pl.Float64)):
        with pytest.warns(
            PolarsInefficientMapWarning,
            match=rf'(?s)replace.*pl\.col\("a"\)\.cast\(pl\.{pl_dtype.__name__}\)',
        ):
            assert_frame_equal(
                lf.select(pl.col("a").map_elements(py_cast)).collect(),
                lf.select(pl.col("a").cast(pl_dtype)).collect(),
            )


def test_parse_apply_miscellaneous() -> None:
    # note: can also identify inefficient functions and methods as well as lambdas
    class Test:
        def x10(self, x: pl.Expr) -> pl.Expr:
            return x * 10

    parser = BytecodeParser(Test().x10, map_target="expr")
    suggested_expression = parser.to_expression(col="colx")
    assert suggested_expression == 'pl.col("colx") * 10'

    # note: all constants - should not create a warning/suggestion
    suggested_expression = BytecodeParser(
        lambda x: MY_CONSTANT + 42, map_target="expr"
    ).to_expression(col="colx")
    assert suggested_expression is None

    # literals as method parameters
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Series\.map_elements.*replace.*\(np\.cos\(3\) \+ s\) - abs\(-1\)",
    ):
        pl_series = pl.Series("srs", [0, 1, 2, 3, 4])
        assert_series_equal(
            pl_series.apply(lambda x: numpy.cos(3) + x - abs(-1)),
            numpy.cos(3) + pl_series - 1,
        )

    # if 's' is already the name of a global variable then the series alias
    # used in the user warning will fall back (in priority order) through
    # various aliases until it finds one that is available.
    s, srs, series = -1, 0, 1
    expr1 = BytecodeParser(lambda x: x + s, map_target="series")
    expr2 = BytecodeParser(lambda x: srs + x + s, map_target="series")
    expr3 = BytecodeParser(lambda x: srs + x + s - x + series, map_target="series")

    assert expr1.to_expression(col="srs") == "srs + s"
    assert expr2.to_expression(col="srs") == "(srs + series) + s"
    assert expr3.to_expression(col="srs") == "(((srs + srs0) + s) - srs0) + series"


@pytest.mark.parametrize(
    ("data", "func", "expr_repr"),
    [
        (
            [1, 2, 3],
            lambda x: str(x),
            "s.cast(pl.Utf8)",
        ),
        (
            [-20, -12, -5, 0, 5, 12, 20],
            lambda x: (abs(x) != 12) and (x > 10 or x < -10 or x == 0),
            "(s.abs() != 12) & ((s > 10) | ((s < -10) | (s == 0)))",
        ),
    ],
)
def test_parse_apply_series(
    data: list[Any], func: Callable[[Any], Any], expr_repr: str
) -> None:
    # expression/series generate same warning, with 's' as the series placeholder
    with pytest.warns(
        PolarsInefficientMapWarning, match=r"(?s)Series\.map_elements.*s\.\w+\("
    ):
        s = pl.Series("srs", data)

        parser = BytecodeParser(func, map_target="series")
        suggested_expression = parser.to_expression(s.name)
        assert suggested_expression == expr_repr

        expected_series = s.apply(func)
        result_series = eval(suggested_expression)
        assert_series_equal(expected_series, result_series)


def test_expr_exact_warning_message() -> None:
    red, green, end_escape = (
        ("\x1b[31m", "\x1b[32m", "\x1b[0m")
        if in_terminal_that_supports_colour()
        else ("", "", "")
    )
    msg = re.escape(
        "\n"
        "Expr.map_elements is significantly slower than the native expressions API.\n"
        "Only use if you absolutely CANNOT implement your logic otherwise.\n"
        "In this case, you can replace your `map_elements` with the following:\n"
        f'  {red}- pl.col("a").map_elements(lambda x: ...){end_escape}\n'
        f'  {green}+ pl.col("a") + 1{end_escape}\n'
    )
    # Check the EXACT warning message. If modifying the message in the future,
    # please make sure to keep the `^` and `$`,
    # and to keep the assertion on `len(warnings)`.
    with pytest.warns(PolarsInefficientMapWarning, match=rf"^{msg}$") as warnings:
        df = pl.DataFrame({"a": [1, 2, 3]})
        df.select(pl.col("a").map_elements(lambda x: x + 1))

    assert len(warnings) == 1

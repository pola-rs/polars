from __future__ import annotations

import json
import re
from typing import Any, Callable

import numpy
import pytest

import polars as pl
from polars.exceptions import PolarsInefficientApplyWarning
from polars.testing import assert_frame_equal, assert_series_equal
from polars.utils.udfs import _NUMPY_FUNCTIONS, BytecodeParser
from polars.utils.various import in_terminal_that_supports_colour
from tests.test_udfs import MY_CONSTANT, MY_DICT, MY_LIST, NOOP_TEST_CASES, TEST_CASES

EVAL_ENVIRONMENT = {
    "np": numpy,
    "pl": pl,
    "MY_CONSTANT": MY_CONSTANT,
    "MY_DICT": MY_DICT,
    "MY_LIST": MY_LIST,
}


@pytest.mark.parametrize(
    "func",
    NOOP_TEST_CASES,
)
def test_parse_invalid_function(func: Callable[[Any], Any]) -> None:
    # functions we don't (yet?) offer suggestions for
    assert not BytecodeParser(func, apply_target="expr").can_rewrite()


@pytest.mark.parametrize(
    ("col", "func", "expr_repr"),
    TEST_CASES,
)
def test_parse_apply_functions(
    col: str, func: Callable[[Any], Any], expr_repr: str
) -> None:
    with pytest.warns(
        PolarsInefficientApplyWarning,
        match=r"(?s)Expr\.apply.*In this case, you can replace",
    ):
        parser = BytecodeParser(func, apply_target="expr")
        suggested_expression = parser.to_expression(col)
        assert suggested_expression == expr_repr

        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["AB", "cd", "eF"],
                "c": ['{"a": 1}', '{"b": 2}', '{"c": 3}'],
            }
        )
        result_frame = df.select(
            x=col,
            y=eval(suggested_expression, EVAL_ENVIRONMENT),
        )
        expected_frame = df.select(
            x=pl.col(col),
            y=pl.col(col).apply(func),
        )
        assert_frame_equal(result_frame, expected_frame)


def test_parse_apply_raw_functions() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    # test bare 'numpy' functions
    for func_name in _NUMPY_FUNCTIONS:
        func = getattr(numpy, func_name)

        # note: we can't parse/rewrite raw numpy functions...
        parser = BytecodeParser(func, apply_target="expr")
        assert not parser.can_rewrite()

        # ...but we ARE still able to warn
        with pytest.warns(
            PolarsInefficientApplyWarning,
            match=rf"(?s)Expr\.apply.*In this case, you can replace.*np\.{func_name}",
        ):
            df1 = lf.select(pl.col("a").apply(func)).collect()
            df2 = lf.select(getattr(pl.col("a"), func_name)()).collect()
            assert_frame_equal(df1, df2)

    # test bare 'json.loads'
    result_frames = []
    with pytest.warns(
        PolarsInefficientApplyWarning,
        match=r"(?s)Expr\.apply.*In this case, you can replace.*\.str\.json_extract",
    ):
        for expr in (
            pl.col("value").str.json_extract(),
            pl.col("value").apply(json.loads),
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
            PolarsInefficientApplyWarning,
            match=rf'(?s)replace.*pl\.col\("a"\)\.cast\(pl\.{pl_dtype.__name__}\)',
        ):
            assert_frame_equal(
                lf.select(pl.col("a").apply(py_cast)).collect(),
                lf.select(pl.col("a").cast(pl_dtype)).collect(),
            )


def test_parse_apply_miscellaneous() -> None:
    # note: can also identify inefficient functions and methods as well as lambdas
    class Test:
        def x10(self, x: pl.Expr) -> pl.Expr:
            return x * 10

    parser = BytecodeParser(Test().x10, apply_target="expr")
    suggested_expression = parser.to_expression(col="colx")
    assert suggested_expression == 'pl.col("colx") * 10'

    # note: all constants - should not create a warning/suggestion
    suggested_expression = BytecodeParser(
        lambda x: MY_CONSTANT + 42, apply_target="expr"
    ).to_expression(col="colx")
    assert suggested_expression is None

    # literals as method parameters
    with pytest.warns(
        PolarsInefficientApplyWarning,
        match=r"(?s)Series\.apply.*replace.*\(np\.cos\(3\) \+ s\) - abs\(-1\)",
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
    expr1 = BytecodeParser(lambda x: x + s, apply_target="series")
    expr2 = BytecodeParser(lambda x: srs + x + s, apply_target="series")
    expr3 = BytecodeParser(lambda x: srs + x + s - x + series, apply_target="series")

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
        PolarsInefficientApplyWarning, match=r"(?s)Series\.apply.*s\.\w+\("
    ):
        s = pl.Series("srs", data)

        parser = BytecodeParser(func, apply_target="series")
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
        "Expr.apply is significantly slower than the native expressions API.\n"
        "Only use if you absolutely CANNOT implement your logic otherwise.\n"
        "In this case, you can replace your `apply` with the following:\n"
        f'  {red}- pl.col("a").apply(lambda x: ...){end_escape}\n'
        f'  {green}+ pl.col("a") + 1{end_escape}\n'
    )
    # Check the EXACT warning message. If modifying the message in the future,
    # please make sure to keep the `^` and `$`,
    # and to keep the assertion on `len(warnings)`.
    with pytest.warns(PolarsInefficientApplyWarning, match=rf"^{msg}$") as warnings:
        df = pl.DataFrame({"a": [1, 2, 3]})
        df.select(pl.col("a").apply(lambda x: x + 1))

    assert len(warnings) == 1

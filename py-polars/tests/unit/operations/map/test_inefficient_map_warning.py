from __future__ import annotations

import datetime as dt
import json
import math
import re
from datetime import date, datetime
from functools import partial
from math import cosh
from typing import Any, Callable, Literal

import numpy as np
import pytest

import polars as pl
from polars._utils.udfs import _BYTECODE_PARSER_CACHE_, _NUMPY_FUNCTIONS, BytecodeParser
from polars._utils.various import in_terminal_that_supports_colour
from polars.exceptions import PolarsInefficientMapWarning
from polars.testing import assert_frame_equal, assert_series_equal

MY_CONSTANT = 3
MY_DICT = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
MY_LIST = [1, 2, 3]

# column_name, function, expected_suggestion
TEST_CASES = [
    # ---------------------------------------------
    # numeric expr: math, comparison, logic ops
    # ---------------------------------------------
    ("a", "lambda x: x + 1 - (2 / 3)", '(pl.col("a") + 1) - 0.6666666666666666', None),
    ("a", "lambda x: x // 1 % 2", '(pl.col("a") // 1) % 2', None),
    ("a", "lambda x: x & True", 'pl.col("a") & True', None),
    ("a", "lambda x: x | False", 'pl.col("a") | False', None),
    ("a", "lambda x: abs(x) != 3", 'pl.col("a").abs() != 3', None),
    ("a", "lambda x: int(x) > 1", 'pl.col("a").cast(pl.Int64) > 1', None),
    (
        "a",
        "lambda x: not (x > 1) or x == 2",
        '~(pl.col("a") > 1) | (pl.col("a") == 2)',
        None,
    ),
    ("a", "lambda x: x is None", 'pl.col("a") is None', None),
    ("a", "lambda x: x is not None", 'pl.col("a") is not None', None),
    (
        "a",
        "lambda x: ((x * -x) ** x) * 1.0",
        '((pl.col("a") * -pl.col("a")) ** pl.col("a")) * 1.0',
        None,
    ),
    (
        "a",
        "lambda x: 1.0 * (x * (x**x))",
        '1.0 * (pl.col("a") * (pl.col("a") ** pl.col("a")))',
        None,
    ),
    (
        "a",
        "lambda x: (x / x) + ((x * x) - x)",
        '(pl.col("a") / pl.col("a")) + ((pl.col("a") * pl.col("a")) - pl.col("a"))',
        None,
    ),
    (
        "a",
        "lambda x: (10 - x) / (((x * 4) - x) // (2 + (x * (x - 1))))",
        '(10 - pl.col("a")) / (((pl.col("a") * 4) - pl.col("a")) // (2 + (pl.col("a") * (pl.col("a") - 1))))',
        None,
    ),
    ("a", "lambda x: x in (2, 3, 4)", 'pl.col("a").is_in((2, 3, 4))', None),
    ("a", "lambda x: x not in (2, 3, 4)", '~pl.col("a").is_in((2, 3, 4))', None),
    (
        "a",
        "lambda x: x in (1, 2, 3, 4, 3) and x % 2 == 0 and x > 0",
        'pl.col("a").is_in((1, 2, 3, 4, 3)) & ((pl.col("a") % 2) == 0) & (pl.col("a") > 0)',
        None,
    ),
    ("a", "lambda x: MY_CONSTANT + x", 'MY_CONSTANT + pl.col("a")', None),
    (
        "a",
        "lambda x: (float(x) * int(x)) // 2",
        '(pl.col("a").cast(pl.Float64) * pl.col("a").cast(pl.Int64)) // 2',
        None,
    ),
    (
        "a",
        "lambda x: 1 / (1 + np.exp(-x))",
        '1 / (1 + (-pl.col("a")).exp())',
        None,
    ),
    # ---------------------------------------------
    # math module
    # ---------------------------------------------
    ("e", "lambda x: math.asin(x)", 'pl.col("e").arcsin()', None),
    ("e", "lambda x: math.asinh(x)", 'pl.col("e").arcsinh()', None),
    ("e", "lambda x: math.atan(x)", 'pl.col("e").arctan()', None),
    ("e", "lambda x: math.atanh(x)", 'pl.col("e").arctanh()', "self"),
    ("e", "lambda x: math.cos(x)", 'pl.col("e").cos()', None),
    ("e", "lambda x: math.degrees(x)", 'pl.col("e").degrees()', None),
    ("e", "lambda x: math.exp(x)", 'pl.col("e").exp()', None),
    ("e", "lambda x: math.log(x)", 'pl.col("e").log()', None),
    ("e", "lambda x: math.log10(x)", 'pl.col("e").log10()', None),
    ("e", "lambda x: math.log1p(x)", 'pl.col("e").log1p()', None),
    ("e", "lambda x: math.radians(x)", 'pl.col("e").radians()', None),
    ("e", "lambda x: math.sin(x)", 'pl.col("e").sin()', None),
    ("e", "lambda x: math.sinh(x)", 'pl.col("e").sinh()', None),
    ("e", "lambda x: math.sqrt(x)", 'pl.col("e").sqrt()', None),
    ("e", "lambda x: math.tan(x)", 'pl.col("e").tan()', None),
    ("e", "lambda x: math.tanh(x)", 'pl.col("e").tanh()', None),
    # ---------------------------------------------
    # numpy module
    # ---------------------------------------------
    ("e", "lambda x: np.arccos(x)", 'pl.col("e").arccos()', None),
    ("e", "lambda x: np.arccosh(x)", 'pl.col("e").arccosh()', None),
    ("e", "lambda x: np.arcsin(x)", 'pl.col("e").arcsin()', None),
    ("e", "lambda x: np.arcsinh(x)", 'pl.col("e").arcsinh()', None),
    ("e", "lambda x: np.arctan(x)", 'pl.col("e").arctan()', None),
    ("e", "lambda x: np.arctanh(x)", 'pl.col("e").arctanh()', "self"),
    ("a", "lambda x: 0 + np.cbrt(x)", '0 + pl.col("a").cbrt()', None),
    ("e", "lambda x: np.ceil(x)", 'pl.col("e").ceil()', None),
    ("e", "lambda x: np.cos(x)", 'pl.col("e").cos()', None),
    ("e", "lambda x: np.cosh(x)", 'pl.col("e").cosh()', None),
    ("e", "lambda x: np.degrees(x)", 'pl.col("e").degrees()', None),
    ("e", "lambda x: np.exp(x)", 'pl.col("e").exp()', None),
    ("e", "lambda x: np.floor(x)", 'pl.col("e").floor()', None),
    ("e", "lambda x: np.log(x)", 'pl.col("e").log()', None),
    ("e", "lambda x: np.log10(x)", 'pl.col("e").log10()', None),
    ("e", "lambda x: np.log1p(x)", 'pl.col("e").log1p()', None),
    ("e", "lambda x: np.radians(x)", 'pl.col("e").radians()', None),
    ("a", "lambda x: np.sign(x)", 'pl.col("a").sign()', None),
    ("a", "lambda x: np.sin(x) + 1", 'pl.col("a").sin() + 1', None),
    (
        "a",  # note: functions operate on consts
        "lambda x: np.sin(3.14159265358979) + (x - 1) + abs(-3)",
        '(np.sin(3.14159265358979) + (pl.col("a") - 1)) + abs(-3)',
        None,
    ),
    ("a", "lambda x: np.sinh(x) + 1", 'pl.col("a").sinh() + 1', None),
    ("a", "lambda x: np.sqrt(x) + 1", 'pl.col("a").sqrt() + 1', None),
    ("a", "lambda x: np.tan(x) + 1", 'pl.col("a").tan() + 1', None),
    ("e", "lambda x: np.tanh(x)", 'pl.col("e").tanh()', None),
    # ---------------------------------------------
    # logical 'and/or' (validate nesting levels)
    # ---------------------------------------------
    (
        "a",
        "lambda x: x > 1 or (x == 1 and x == 2)",
        '(pl.col("a") > 1) | ((pl.col("a") == 1) & (pl.col("a") == 2))',
        None,
    ),
    (
        "a",
        "lambda x: (x > 1 or x == 1) and x == 2",
        '((pl.col("a") > 1) | (pl.col("a") == 1)) & (pl.col("a") == 2)',
        None,
    ),
    (
        "a",
        "lambda x: x > 2 or x != 3 and x not in (0, 1, 4)",
        '(pl.col("a") > 2) | ((pl.col("a") != 3) & ~pl.col("a").is_in((0, 1, 4)))',
        None,
    ),
    (
        "a",
        "lambda x: x > 1 and x != 2 or x % 2 == 0 and x < 3",
        '((pl.col("a") > 1) & (pl.col("a") != 2)) | (((pl.col("a") % 2) == 0) & (pl.col("a") < 3))',
        None,
    ),
    (
        "a",
        "lambda x: x > 1 and (x != 2 or x % 2 == 0) and x < 3",
        '(pl.col("a") > 1) & ((pl.col("a") != 2) | ((pl.col("a") % 2) == 0)) & (pl.col("a") < 3)',
        None,
    ),
    # ---------------------------------------------
    # string exprs
    # ---------------------------------------------
    (
        "b",
        "lambda x: str(x).title()",
        'pl.col("b").cast(pl.String).str.to_titlecase()',
        None,
    ),
    (
        "b",
        'lambda x: x.lower() + ":" + x.upper() + ":" + x.title()',
        '(((pl.col("b").str.to_lowercase() + \':\') + pl.col("b").str.to_uppercase()) + \':\') + pl.col("b").str.to_titlecase()',
        None,
    ),
    (
        "b",
        "lambda x: x.strip().startswith('#')",
        """pl.col("b").str.strip_chars().str.starts_with('#')""",
        None,
    ),
    (
        "b",
        """lambda x: x.rstrip().endswith(('!','#','?','"'))""",
        """pl.col("b").str.strip_chars_end().str.contains(r'(!|\\#|\\?|")$')""",
        None,
    ),
    (
        "b",
        """lambda x: x.lstrip().startswith(('!','#','?',"'"))""",
        """pl.col("b").str.strip_chars_start().str.contains(r"^(!|\\#|\\?|')")""",
        None,
    ),
    (
        "b",
        "lambda x: x.replace(':','')",
        """pl.col("b").str.replace_all(':','',literal=True)""",
        None,
    ),
    (
        "b",
        "lambda x: x.replace(':','',2)",
        """pl.col("b").str.replace(':','',n=2,literal=True)""",
        None,
    ),
    (
        "b",
        "lambda x: x.removeprefix('A').removesuffix('F')",
        """pl.col("b").str.strip_prefix('A').str.strip_suffix('F')""",
        None,
    ),
    (
        "b",
        "lambda x: x.zfill(8)",
        """pl.col("b").str.zfill(8)""",
        None,
    ),
    # ---------------------------------------------
    # replace
    # ---------------------------------------------
    ("a", "lambda x: MY_DICT[x]", 'pl.col("a").replace_strict(MY_DICT)', None),
    (
        "a",
        "lambda x: MY_DICT[x - 1] + MY_DICT[1 + x]",
        '(pl.col("a") - 1).replace_strict(MY_DICT) + (1 + pl.col("a")).replace_strict(MY_DICT)',
        None,
    ),
    # ---------------------------------------------
    # standard library datetime parsing
    # ---------------------------------------------
    (
        "d",
        'lambda x: datetime.strptime(x, "%Y-%m-%d")',
        'pl.col("d").str.to_datetime(format="%Y-%m-%d")',
        pl.Datetime("us"),
    ),
    (
        "d",
        'lambda x: dt.datetime.strptime(x, "%Y-%m-%d")',
        'pl.col("d").str.to_datetime(format="%Y-%m-%d")',
        pl.Datetime("us"),
    ),
    # ---------------------------------------------
    # temporal attributes/methods
    # ---------------------------------------------
    (
        "f",
        "lambda x: x.isoweekday()",
        'pl.col("f").dt.weekday()',
        None,
    ),
    (
        "f",
        "lambda x: x.hour + x.minute + x.second",
        '(pl.col("f").dt.hour() + pl.col("f").dt.minute()) + pl.col("f").dt.second()',
        None,
    ),
    # ---------------------------------------------
    # Bitwise shifts
    # ---------------------------------------------
    (
        "a",
        "lambda x: (3 << (30-x)) & 3",
        '(3 * 2**(30 - pl.col("a"))).cast(pl.Int64) & 3',
        None,
    ),
    (
        "a",
        "lambda x: (x << 32) & 3",
        '(pl.col("a") * 2**32).cast(pl.Int64) & 3',
        None,
    ),
    (
        "a",
        "lambda x: ((32-x) >> (3)) & 3",
        '((32 - pl.col("a")) / 2**3).cast(pl.Int64) & 3',
        None,
    ),
    (
        "a",
        "lambda x: (32 >> (3-x)) & 3",
        '(32 / 2**(3 - pl.col("a"))).cast(pl.Int64) & 3',
        None,
    ),
]

NOOP_TEST_CASES = [
    "lambda x: x",
    "lambda x, y: x + y",
    "lambda x: x[0] + 1",
    "lambda x: MY_LIST[x]",
    "lambda x: MY_DICT[1]",
    'lambda x: "first" if x == 1 else "not first"',
    'lambda x: np.sign(x, casting="unsafe")',
]

EVAL_ENVIRONMENT = {
    "MY_CONSTANT": MY_CONSTANT,
    "MY_DICT": MY_DICT,
    "MY_LIST": MY_LIST,
    "cosh": cosh,
    "datetime": datetime,
    "dt": dt,
    "math": math,
    "np": np,
    "pl": pl,
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
    ("col", "func", "expr_repr", "dtype"),
    TEST_CASES,
)
@pytest.mark.filterwarnings(
    "ignore:.*:polars.exceptions.MapWithoutReturnDtypeWarning",
    "ignore:invalid value encountered:RuntimeWarning",
    "ignore:.*without specifying `return_dtype`:polars.exceptions.MapWithoutReturnDtypeWarning",
)
@pytest.mark.may_fail_auto_streaming  # dtype not set
@pytest.mark.may_fail_cloud  # reason: eager - return_dtype must be set
def test_parse_apply_functions(
    col: str, func: str, expr_repr: str, dtype: Literal["self"] | pl.DataType | None
) -> None:
    return_dtype: pl.DataTypeExpr | None = None
    if dtype == "self":
        return_dtype = pl.self_dtype()
    elif dtype is None:
        return_dtype = None
    else:
        return_dtype = dtype.to_dtype_expr()  # type: ignore[union-attr]
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Expr\.map_elements.*with this one instead",
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
                "e": [0.5, 0.4, 0.1],
                "f": [
                    datetime(1969, 12, 31),
                    datetime(2024, 5, 6),
                    datetime(2077, 10, 20),
                ],
            }
        )

        result_frame = df.select(
            x=col,
            y=eval(suggested_expression, EVAL_ENVIRONMENT),
        )
        expected_frame = df.select(
            x=pl.col(col),
            y=pl.col(col).map_elements(eval(func), return_dtype=return_dtype),
        )
        assert_frame_equal(
            result_frame,
            expected_frame,
            check_dtypes=(".dt." not in suggested_expression),
        )


@pytest.mark.filterwarnings(
    "ignore:.*:polars.exceptions.MapWithoutReturnDtypeWarning",
    "ignore:invalid value encountered:RuntimeWarning",
    "ignore:.*without specifying `return_dtype`:polars.exceptions.MapWithoutReturnDtypeWarning",
)
@pytest.mark.may_fail_auto_streaming  # dtype is not set
def test_parse_apply_raw_functions() -> None:
    lf = pl.LazyFrame({"a": [1.1, 2.0, 3.4]})

    # test bare 'numpy' functions
    for func_name in _NUMPY_FUNCTIONS:
        func = getattr(np, func_name)

        # note: we can't parse/rewrite raw numpy functions...
        parser = BytecodeParser(func, map_target="expr")
        assert not parser.can_attempt_rewrite()

        # ...but we ARE still able to warn
        with pytest.warns(
            PolarsInefficientMapWarning,
            match=rf"(?s)Expr\.map_elements.*Replace this expression.*np\.{func_name}",
        ):
            df1 = lf.select(
                pl.col("a").map_elements(func, return_dtype=pl.self_dtype())
            ).collect()
            df2 = lf.select(getattr(pl.col("a"), func_name)()).collect()
            assert_frame_equal(df1, df2)

    # test bare 'json.loads'
    result_frames = []
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Expr\.map_elements.*with this one instead:.*\.str\.json_decode",
    ):
        for expr in (
            pl.col("value").str.json_decode(
                pl.Struct(
                    {
                        "a": pl.Int64,
                        "b": pl.Boolean,
                        "c": pl.String,
                    }
                )
            ),
            pl.col("value").map_elements(
                json.loads,
                return_dtype=pl.Struct(
                    {
                        "a": pl.Int64,
                        "b": pl.Boolean,
                        "c": pl.String,
                    }
                ),
            ),
        ):
            result_frames.append(  # noqa: PERF401
                pl.LazyFrame({"value": ['{"a":1, "b": true, "c": "xx"}', None]})
                .select(extracted=expr)
                .unnest("extracted")
                .collect()
            )

    assert_frame_equal(*result_frames)

    # test primitive python casts
    for py_cast, pl_dtype in ((str, pl.String), (int, pl.Int64), (float, pl.Float64)):
        with pytest.warns(
            PolarsInefficientMapWarning,
            match=rf'(?s)with this one instead.*pl\.col\("a"\)\.cast\(pl\.{pl_dtype.__name__}\)',
        ):
            assert_frame_equal(
                lf.select(
                    pl.col("a").map_elements(py_cast, return_dtype=pl_dtype)
                ).collect(),
                lf.select(pl.col("a").cast(pl_dtype)).collect(),
            )


def test_parse_apply_miscellaneous() -> None:
    # note: can also identify inefficient functions and methods as well as lambdas
    class Test:
        def x10(self, x: float) -> float:
            return x * 10

        def mcosh(self, x: float) -> float:
            return cosh(x)

    parser = BytecodeParser(Test().x10, map_target="expr")
    suggested_expression = parser.to_expression(col="colx")
    assert suggested_expression == 'pl.col("colx") * 10'

    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Series\.map_elements.*with this one instead.*s\.cosh\(\)",
    ):
        pl.Series("colx", [0.5, 0.25]).map_elements(
            function=Test().mcosh,
            return_dtype=pl.Float64,
        )

    # note: all constants - should not create a warning/suggestion
    suggested_expression = BytecodeParser(
        lambda x: MY_CONSTANT + 42, map_target="expr"
    ).to_expression(col="colx")
    assert suggested_expression is None

    # literals as method parameters
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Series\.map_elements.*with this one instead.*\(np\.cos\(3\) \+ s\) - abs\(-1\)",
    ):
        s = pl.Series("srs", [0, 1, 2, 3, 4])
        assert_series_equal(
            s.map_elements(lambda x: np.cos(3) + x - abs(-1), return_dtype=pl.Float64),
            np.cos(3) + s - 1,
        )

    # if 's' is already the name of a global variable then the series alias
    # used in the user warning will fall back (in priority order) through
    # various aliases until it finds one that is available.
    s, srs, series = -1, 0, 1  # type: ignore[assignment]
    expr1 = BytecodeParser(lambda x: x + s, map_target="series")
    expr2 = BytecodeParser(lambda x: srs + x + s, map_target="series")
    expr3 = BytecodeParser(lambda x: srs + x + s - x + series, map_target="series")

    assert expr1.to_expression(col="srs") == "srs + s"
    assert expr2.to_expression(col="srs") == "(srs + series) + s"
    assert expr3.to_expression(col="srs") == "(((srs + srs0) + s) - srs0) + series"


@pytest.mark.parametrize(
    ("name", "data", "func", "expr_repr"),
    [
        (
            "srs",
            [1, 2, 3],
            lambda x: str(x),
            "s.cast(pl.String)",
        ),
        (
            "s",
            [date(2077, 10, 10), date(1999, 12, 31)],
            lambda d: d.month,
            "s.dt.month()",
        ),
        (
            "",
            [-20, -12, -5, 0, 5, 12, 20],
            lambda x: (abs(x) != 12) and (x > 10 or x < -10 or x == 0),
            "(s.abs() != 12) & ((s > 10) | (s < -10) | (s == 0))",
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:.*without specifying `return_dtype`:polars.exceptions.MapWithoutReturnDtypeWarning"
)
def test_parse_apply_series(
    name: str, data: list[Any], func: Callable[[Any], Any], expr_repr: str
) -> None:
    # expression/series generate same warning, with 's' as the series placeholder
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)Series\.map_elements.*s\.\w+\(",
    ):
        s = pl.Series(name, data)

        parser = BytecodeParser(func, map_target="series")
        suggested_expression = parser.to_expression(s.name)
        assert suggested_expression == expr_repr

        expected_series = s.map_elements(func)
        result_series = eval(suggested_expression)
        assert_series_equal(expected_series, result_series, check_dtypes=False)


@pytest.mark.may_fail_auto_streaming
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
        "Replace this expression...\n"
        f'  {red}- pl.col("a").map_elements(lambda x: ...){end_escape}\n'
        "with this one instead:\n"
        f'  {green}+ pl.col("a") + 1{end_escape}\n'
    )

    fn = lambda x: x + 1  # noqa: E731

    # check the EXACT warning messages - if modifying the message in the future,
    # make sure to keep the `^` and `$`, and the assertion on `len(warnings)`
    with pytest.warns(PolarsInefficientMapWarning, match=rf"^{msg}$") as warnings:
        df = pl.DataFrame({"a": [1, 2, 3]})
        for _ in range(3):  # << loop a few times to exercise the caching path
            df.select(pl.col("a").map_elements(fn, return_dtype=pl.Int64))

    assert len(warnings) == 3

    # confirm that the associated parser/etc was cached
    bp = _BYTECODE_PARSER_CACHE_[(fn, "expr")]
    assert isinstance(bp, BytecodeParser)
    assert bp.to_expression("a") == 'pl.col("a") + 1'


def test_omit_implicit_bool() -> None:
    parser = BytecodeParser(
        function=lambda x: x and x and x.date(),
        map_target="expr",
    )
    suggested_expression = parser.to_expression("d")
    assert suggested_expression == 'pl.col("d").dt.date()'


def test_partial_functions_13523() -> None:
    def plus(value: int, amount: int) -> int:
        return value + amount

    data = {"a": [1, 2], "b": [3, 4]}
    df = pl.DataFrame(data)
    # should not warn
    _ = df["a"].map_elements(partial(plus, amount=1))

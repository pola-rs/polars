from __future__ import annotations

import json
from typing import Any, Callable

import numpy
import numpy as np
import pytest

import polars as pl
from polars.exceptions import PolarsInefficientApplyWarning
from polars.testing import assert_frame_equal
from polars.utils.udfs import _NUMPY_FUNCTIONS, BytecodeParser

MY_CONSTANT = 3


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x,
        lambda x, y: x + y,
        lambda x: x[0] + 1,
        lambda x: x > 0 and (x < 100 or (x % 2 == 0)),
    ],
)
def test_parse_invalid_function(func: Callable[[Any], Any]) -> None:
    # functions we don't offer suggestions for (at all, or just not yet)
    assert not BytecodeParser(func, apply_target="expr").can_rewrite()


@pytest.mark.parametrize(
    ("col", "func"),
    [
        # numeric expr: math, comparison, logic ops
        ("a", lambda x: x + 1 - (2 / 3)),
        ("a", lambda x: x // 1 % 2),
        ("a", lambda x: x & True),
        ("a", lambda x: x | False),
        ("a", lambda x: x != 3),
        ("a", lambda x: int(x) > 1),
        ("a", lambda x: not (x > 1) or x == 2),
        ("a", lambda x: x is None),
        ("a", lambda x: x is not None),
        ("a", lambda x: ((x * -x) ** x) * 1.0),
        ("a", lambda x: 1.0 * (x * (x**x))),
        ("a", lambda x: (x / x) + ((x * x) - x)),
        ("a", lambda x: (10 - x) / (((x * 4) - x) // (2 + (x * (x - 1))))),
        ("a", lambda x: x in (2, 3, 4)),
        ("a", lambda x: x not in (2, 3, 4)),
        ("a", lambda x: x in (1, 2, 3, 4, 3) and x % 2 == 0 and x > 0),
        ("a", lambda x: MY_CONSTANT + x),
        ("a", lambda x: 0 + numpy.cbrt(x)),
        ("a", lambda x: np.sin(x) + 1),
        ("a", lambda x: (float(x) * int(x)) // 2),
        # string expr: case/cast ops
        ("a", lambda x: str(x).title()),
        ("b", lambda x: x.lower() + ":" + x.upper() + ":" + x.title()),
        # json expr: load/extract
        ("c", lambda x: json.loads(x)),
    ],
)
def test_parse_apply_functions(col: str, func: Callable[[Any], Any]) -> None:
    with pytest.warns(
        PolarsInefficientApplyWarning, match="In this case, you can replace"
    ):
        parser = BytecodeParser(func, apply_target="expr")
        suggested_expression = parser.to_expression(col)
        assert isinstance(suggested_expression, str)

        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["AB", "cd", "eF"],
                "c": ['{"a": 1}', '{"b": 2}', '{"c": 3}'],
            }
        )
        result = df.select(
            x=col,
            y=eval(suggested_expression),
        )
        expected = df.select(
            x=pl.col(col),
            y=pl.col(col).apply(func),
        )
        assert_frame_equal(result, expected)


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
            match=rf"(?s)In this case, you can replace.*np\.{func_name}",
        ):
            df1 = lf.select(pl.col("a").apply(func)).collect()
            df2 = lf.select(getattr(pl.col("a"), func_name)()).collect()
            assert_frame_equal(df1, df2)

    # test bare 'json.loads'
    result_frames = []
    with pytest.warns(
        PolarsInefficientApplyWarning,
        match=r"(?s)In this case, you can replace.*\.str\.json_extract",
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
            match=rf'(?s) replace.*pl\.col\("a"\)\.cast\(pl\.{pl_dtype.__name__}\)',
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

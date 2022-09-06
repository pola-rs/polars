from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import reduce
from typing import Sequence, no_type_check

import numpy as np

import polars as pl


def test_apply_none() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2, 5],
            "a": [2, 4, 5, 190, 1, 4, 1],
            "b": [1, 3, 2, 1, 43, 3, 1],
        }
    )

    out = (
        df.groupby("g", maintain_order=True).agg(
            pl.apply(
                exprs=["a", pl.col("b") ** 4, pl.col("a") / 4],
                f=lambda x: x[0] * x[1] + x[2].sum(),
            ).alias("multiple")
        )
    )["multiple"]
    assert out[0].to_list() == [4.75, 326.75, 82.75]
    assert out[1].to_list() == [238.75, 3418849.75, 372.75]

    out_df = df.select(pl.map(exprs=["a", "b"], f=lambda s: s[0] * s[1]))
    assert out_df["a"].to_list() == (df["a"] * df["b"]).to_list()

    # check if we can return None
    def func(s: Sequence[pl.Series]) -> pl.Series | None:
        if s[0][0] == 190:
            return None
        else:
            return s[0]

    out = (
        df.groupby("g", maintain_order=True).agg(
            pl.apply(exprs=["a", pl.col("b") ** 4, pl.col("a") / 4], f=func).alias(
                "multiple"
            )
        )
    )["multiple"]
    assert out[1] is None


def test_apply_return_py_object() -> None:
    df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    out = df.select([pl.all().map(lambda s: reduce(lambda a, b: a + b, s))])

    assert out.shape == (1, 2)


@no_type_check
def test_agg_objects() -> None:
    df = pl.DataFrame(
        {
            "names": ["foo", "ham", "spam", "cheese", "egg", "foo"],
            "dates": ["1", "1", "2", "3", "3", "4"],
            "groups": ["A", "A", "B", "B", "B", "C"],
        }
    )

    class Foo:
        def __init__(self, payload):
            self.payload = payload

    out = df.groupby("groups").agg(
        [
            pl.apply(
                [pl.col("dates"), pl.col("names")], lambda s: Foo(dict(zip(s[0], s[1])))
            )
        ]
    )
    assert out.dtypes == [pl.Utf8, pl.Object]


def test_apply_infer_list() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2],
            "str": ["a", "b"],
            "bool": [True, None],
        }
    )
    assert df.select([pl.all().apply(lambda x: [x])]).dtypes == [pl.List] * 3


def test_apply_arithmetic_consistency() -> None:
    df = pl.DataFrame({"A": ["a", "a"], "B": [2, 3]})
    assert df.groupby("A").agg(pl.col("B").apply(lambda x: x + 1.0))["B"].to_list() == [
        [3.0, 4.0]
    ]


def test_apply_struct() -> None:
    df = pl.DataFrame(
        {"A": ["a", "a"], "B": [2, 3], "C": [True, False], "D": [12.0, None]}
    )
    out = df.with_column(pl.struct(df.columns).alias("struct")).select(
        [
            pl.col("struct").apply(lambda x: x["A"]).alias("A_field"),
            pl.col("struct").apply(lambda x: x["B"]).alias("B_field"),
            pl.col("struct").apply(lambda x: x["C"]).alias("C_field"),
            pl.col("struct").apply(lambda x: x["D"]).alias("D_field"),
        ]
    )
    expected = pl.DataFrame(
        {
            "A_field": ["a", "a"],
            "B_field": [2, 3],
            "C_field": [True, False],
            "D_field": [12.0, None],
        }
    )

    assert out.frame_equal(expected)


def test_apply_numpy_out_3057() -> None:
    df = pl.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 1],
            "t": [2.0, 4.3, 5, 10, 11, 14],
            "y": [0.0, 1, 1.3, 2, 3, 4],
        }
    )

    assert (
        df.groupby("id", maintain_order=True)
        .agg(
            pl.apply(["y", "t"], lambda lst: np.trapz(y=lst[0], x=lst[1])).alias(
                "result"
            )
        )
        .frame_equal(pl.DataFrame({"id": [0, 1], "result": [1.955, 13.0]}))
    )


def test_apply_numpy_int_out() -> None:
    df = pl.DataFrame({"col1": [2, 4, 8, 16]})
    assert df.with_column(
        pl.col("col1").apply(lambda x: np.left_shift(x, 8)).alias("result")
    ).frame_equal(
        pl.DataFrame({"col1": [2, 4, 8, 16], "result": [512, 1024, 2048, 4096]})
    )
    df = pl.DataFrame({"col1": [2, 4, 8, 16], "shift": [1, 1, 2, 2]})

    assert df.select(
        pl.struct(["col1", "shift"])
        .apply(lambda cols: np.left_shift(cols["col1"], cols["shift"]))
        .alias("result")
    ).frame_equal(pl.DataFrame({"result": [4, 8, 32, 64]}))


def test_datelike_identity() -> None:
    for s in [
        pl.Series([datetime(year=2000, month=1, day=1)]),
        pl.Series([timedelta(hours=2)]),
        pl.Series([date(year=2000, month=1, day=1)]),
    ]:
        assert s.apply(lambda x: x).to_list() == s.to_list()


def test_apply_list_anyvalue_fallback() -> None:
    import json

    df = pl.DataFrame({"text": ['[{"x": 1, "y": 2}, {"x": 3, "y": 4}]']})
    assert df.select(pl.col("text").apply(json.loads)).to_dict(False) == {
        "text": [[{"x": 1, "y": 2}, {"x": 3, "y": 4}]]
    }

    # starts with empty list '[]'
    df = pl.DataFrame(
        {"text": ["[]", '[{"x": 1, "y": 2}, {"x": 3, "y": 4}]', '[{"x": 1, "y": 2}]']}
    )
    assert df.select(pl.col("text").apply(json.loads)).to_dict(False) == {
        "text": [[], [{"x": 1, "y": 2}, {"x": 3, "y": 4}], [{"x": 1, "y": 2}]]
    }


def test_apply_all_types() -> None:
    dtypes = [
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
    ]
    # test we don't panic
    for dtype in dtypes:
        pl.Series([1, 2, 3, 4, 5], dtype=dtype).apply(lambda x: x)


def test_apply_type_propagation() -> None:
    assert (
        pl.from_dict(
            {
                "a": [1, 2, 3],
                "b": [{"c": 1, "d": 2}, {"c": 2, "d": 3}, {"c": None, "d": None}],
            }
        )
        .groupby("a", maintain_order=True)
        .agg(
            [
                pl.when(pl.col("b").null_count() == 0)
                .then(
                    pl.col("b").apply(
                        lambda s: s[0]["c"],
                        return_dtype=pl.Float64,
                    )
                )
                .otherwise(None)
            ]
        )
    ).to_dict(False) == {"a": [1, 2, 3], "b": [1.0, 2.0, None]}


def test_empty_list_in_apply() -> None:
    df = pl.DataFrame(
        {"a": [[1], [1, 2], [3, 4], [5, 6]], "b": [[3], [1, 2], [1, 2], [4, 5]]}
    )

    assert df.select(
        pl.struct(["a", "b"]).apply(lambda row: list(set(row["a"]) & set(row["b"])))
    ).to_dict(False) == {"a": [[], [1, 2], [], [5]]}

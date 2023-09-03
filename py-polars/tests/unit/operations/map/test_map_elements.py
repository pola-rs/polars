from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pytest

import polars as pl
from polars.exceptions import PolarsInefficientMapWarning
from polars.testing import assert_frame_equal


def test_map_elements_infer_list() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2],
            "str": ["a", "b"],
            "bool": [True, None],
        }
    )
    assert df.select([pl.all().map_elements(lambda x: [x])]).dtypes == [pl.List] * 3


def test_map_elements_arithmetic_consistency() -> None:
    df = pl.DataFrame({"A": ["a", "a"], "B": [2, 3]})
    with pytest.warns(
        PolarsInefficientMapWarning, match="In this case, you can replace"
    ):
        assert df.group_by("A").agg(pl.col("B").map_elements(lambda x: x + 1.0))[
            "B"
        ].to_list() == [[3.0, 4.0]]


def test_map_elements_struct() -> None:
    df = pl.DataFrame(
        {"A": ["a", "a"], "B": [2, 3], "C": [True, False], "D": [12.0, None]}
    )
    out = df.with_columns(pl.struct(df.columns).alias("struct")).select(
        pl.col("struct").map_elements(lambda x: x["A"]).alias("A_field"),
        pl.col("struct").map_elements(lambda x: x["B"]).alias("B_field"),
        pl.col("struct").map_elements(lambda x: x["C"]).alias("C_field"),
        pl.col("struct").map_elements(lambda x: x["D"]).alias("D_field"),
    )
    expected = pl.DataFrame(
        {
            "A_field": ["a", "a"],
            "B_field": [2, 3],
            "C_field": [True, False],
            "D_field": [12.0, None],
        }
    )

    assert_frame_equal(out, expected)


def test_map_elements_numpy_int_out() -> None:
    df = pl.DataFrame({"col1": [2, 4, 8, 16]})
    result = df.with_columns(
        pl.col("col1").map_elements(lambda x: np.left_shift(x, 8)).alias("result")
    )
    expected = pl.DataFrame({"col1": [2, 4, 8, 16], "result": [512, 1024, 2048, 4096]})
    assert_frame_equal(result, expected)

    df = pl.DataFrame({"col1": [2, 4, 8, 16], "shift": [1, 1, 2, 2]})
    result = df.select(
        pl.struct(["col1", "shift"])
        .map_elements(lambda cols: np.left_shift(cols["col1"], cols["shift"]))
        .alias("result")
    )
    expected = pl.DataFrame({"result": [4, 8, 32, 64]})
    assert_frame_equal(result, expected)


def test_datelike_identity() -> None:
    for s in [
        pl.Series([datetime(year=2000, month=1, day=1)]),
        pl.Series([timedelta(hours=2)]),
        pl.Series([date(year=2000, month=1, day=1)]),
    ]:
        assert s.map_elements(lambda x: x).to_list() == s.to_list()


def test_map_elements_list_anyvalue_fallback() -> None:
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r'(?s)replace your `map_elements` with.*pl.col\("text"\).str.json_extract()',
    ):
        df = pl.DataFrame({"text": ['[{"x": 1, "y": 2}, {"x": 3, "y": 4}]']})
        assert df.select(pl.col("text").map_elements(json.loads)).to_dict(False) == {
            "text": [[{"x": 1, "y": 2}, {"x": 3, "y": 4}]]
        }

        # starts with empty list '[]'
        df = pl.DataFrame(
            {
                "text": [
                    "[]",
                    '[{"x": 1, "y": 2}, {"x": 3, "y": 4}]',
                    '[{"x": 1, "y": 2}]',
                ]
            }
        )
        assert df.select(pl.col("text").map_elements(json.loads)).to_dict(False) == {
            "text": [[], [{"x": 1, "y": 2}, {"x": 3, "y": 4}], [{"x": 1, "y": 2}]]
        }


def test_map_elements_all_types() -> None:
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
        pl.Series([1, 2, 3, 4, 5], dtype=dtype).map_elements(lambda x: x)


def test_map_elements_type_propagation() -> None:
    assert (
        pl.from_dict(
            {
                "a": [1, 2, 3],
                "b": [{"c": 1, "d": 2}, {"c": 2, "d": 3}, {"c": None, "d": None}],
            }
        )
        .group_by("a", maintain_order=True)
        .agg(
            [
                pl.when(pl.col("b").null_count() == 0)
                .then(
                    pl.col("b").map_elements(
                        lambda s: s[0]["c"],
                        return_dtype=pl.Float64,
                    )
                )
                .otherwise(None)
            ]
        )
    ).to_dict(False) == {"a": [1, 2, 3], "b": [1.0, 2.0, None]}


def test_empty_list_in_map_elements() -> None:
    df = pl.DataFrame(
        {"a": [[1], [1, 2], [3, 4], [5, 6]], "b": [[3], [1, 2], [1, 2], [4, 5]]}
    )

    assert df.select(
        pl.struct(["a", "b"]).map_elements(
            lambda row: list(set(row["a"]) & set(row["b"]))
        )
    ).to_dict(False) == {"a": [[], [1, 2], [], [5]]}


def test_map_elements_skip_nulls() -> None:
    some_map = {None: "a", 1: "b"}
    s = pl.Series([None, 1])

    assert s.map_elements(lambda x: some_map[x]).to_list() == [None, "b"]
    assert s.map_elements(lambda x: some_map[x], skip_nulls=False).to_list() == [
        "a",
        "b",
    ]


def test_map_elements_object_dtypes() -> None:
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r"(?s)replace your `map_elements` with.*lambda x:",
    ):
        assert pl.DataFrame(
            {"a": pl.Series([1, 2, "a", 4, 5], dtype=pl.Object)}
        ).with_columns(
            [
                pl.col("a").map_elements(lambda x: x * 2, return_dtype=pl.Object),
                pl.col("a")
                .map_elements(
                    lambda x: isinstance(x, (int, float)), return_dtype=pl.Boolean
                )
                .alias("is_numeric1"),
                pl.col("a")
                .map_elements(lambda x: isinstance(x, (int, float)))
                .alias("is_numeric_infer"),
            ]
        ).to_dict(
            False
        ) == {
            "a": [2, 4, "aa", 8, 10],
            "is_numeric1": [True, True, False, True, True],
            "is_numeric_infer": [True, True, False, True, True],
        }


def test_map_elements_explicit_list_output_type() -> None:
    out = pl.DataFrame({"str": ["a", "b"]}).with_columns(
        [
            pl.col("str").map_elements(
                lambda _: pl.Series([1, 2, 3]), return_dtype=pl.List(pl.Int64)
            )
        ]
    )

    assert out.dtypes == [pl.List(pl.Int64)]
    assert out.to_dict(False) == {"str": [[1, 2, 3], [1, 2, 3]]}


def test_map_elements_dict() -> None:
    with pytest.warns(
        PolarsInefficientMapWarning,
        match=r'(?s)replace your `map_elements` with.*pl.col\("abc"\).str.json_extract()',
    ):
        df = pl.DataFrame({"abc": ['{"A":"Value1"}', '{"B":"Value2"}']})
        assert df.select(pl.col("abc").map_elements(json.loads)).to_dict(False) == {
            "abc": [{"A": "Value1", "B": None}, {"A": None, "B": "Value2"}]
        }
        assert pl.DataFrame(
            {"abc": ['{"A":"Value1", "B":"Value2"}', '{"B":"Value3"}']}
        ).select(pl.col("abc").map_elements(json.loads)).to_dict(False) == {
            "abc": [{"A": "Value1", "B": "Value2"}, {"A": None, "B": "Value3"}]
        }


def test_map_elements_pass_name() -> None:
    df = pl.DataFrame(
        {
            "bar": [1, 1, 2],
            "foo": [1, 2, 3],
        }
    )

    mapper = {"foo": "foo1"}

    def element_mapper(s: pl.Series) -> pl.Series:
        return pl.Series([mapper[s.name]])

    assert df.group_by("bar", maintain_order=True).agg(
        pl.col("foo").map_elements(element_mapper, pass_name=True),
    ).to_dict(False) == {"bar": [1, 2], "foo": [["foo1"], ["foo1"]]}


def test_map_elements_binary() -> None:
    assert pl.DataFrame({"bin": [b"\x11" * 12, b"\x22" * 12, b"\xaa" * 12]}).select(
        pl.col("bin").map_elements(bytes.hex)
    ).to_dict(False) == {
        "bin": [
            "111111111111111111111111",
            "222222222222222222222222",
            "aaaaaaaaaaaaaaaaaaaaaaaa",
        ]
    }


def test_map_elements_set_datetime_output_8984() -> None:
    df = pl.DataFrame({"a": [""]})
    payload = datetime(2001, 1, 1)
    assert df.select(
        pl.col("a").map_elements(lambda _: payload, return_dtype=pl.Datetime),
    )["a"].to_list() == [payload]


def test_map_elements_dict_order_10128() -> None:
    df = pl.select(pl.lit("").map_elements(lambda x: {"c": 1, "b": 2, "a": 3}))
    assert df.to_dict(False) == {"literal": [{"c": 1, "b": 2, "a": 3}]}


def test_map_elements_10237() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert (
        df.select(pl.all().map_elements(lambda x: x > 50))["a"].to_list() == [False] * 3
    )


def test_map_elements_on_empty_col_10639() -> None:
    df = pl.DataFrame({"A": [], "B": []})
    res = df.group_by("B").agg(
        pl.col("A")
        .map_elements(lambda x: x, return_dtype=pl.Int32, strategy="threading")
        .alias("Foo")
    )
    assert res.to_dict(False) == {
        "B": [],
        "Foo": [],
    }
    res = df.group_by("B").agg(
        pl.col("A")
        .map_elements(lambda x: x, return_dtype=pl.Int32, strategy="thread_local")
        .alias("Foo")
    )
    assert res.to_dict(False) == {
        "B": [],
        "Foo": [],
    }


def test_apply_deprecated() -> None:
    with pytest.deprecated_call():
        pl.col("a").apply(lambda x: x + 1)
    with pytest.deprecated_call():
        pl.Series([1, 2, 3]).apply(lambda x: x + 1)

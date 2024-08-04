from __future__ import annotations

import itertools
import random
from datetime import datetime
from typing import Any

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal


def test_when_then() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    expr = pl.when(pl.col("a") < 3).then(pl.lit("x"))

    result = df.select(
        expr.otherwise(pl.lit("y")).alias("a"),
        expr.alias("b"),
    )

    expected = pl.DataFrame(
        {
            "a": ["x", "x", "y", "y", "y"],
            "b": ["x", "x", None, None, None],
        }
    )
    assert_frame_equal(result, expected)


def test_when_then_chained() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    expr = (
        pl.when(pl.col("a") < 3)
        .then(pl.lit("x"))
        .when(pl.col("a") > 4)
        .then(pl.lit("z"))
    )

    result = df.select(
        expr.otherwise(pl.lit("y")).alias("a"),
        expr.alias("b"),
    )

    expected = pl.DataFrame(
        {
            "a": ["x", "x", "y", "y", "z"],
            "b": ["x", "x", None, None, "z"],
        }
    )
    assert_frame_equal(result, expected)


def test_when_then_invalid_chains() -> None:
    with pytest.raises(AttributeError):
        pl.when("a").when("b")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").otherwise(2)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).then(2)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).otherwise(2).otherwise(3)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).when("b").when("c")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).when("b").otherwise("2")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        pl.when("a").then(1).when("b").then(2).when("c").when("d")  # type: ignore[attr-defined]


def test_when_then_implicit_none() -> None:
    df = pl.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B", "C"],
            "points": [11, 8, 10, 6, 6, 5],
        }
    )

    result = df.select(
        pl.when(pl.col("points") > 7).then(pl.lit("Foo")),
        pl.when(pl.col("points") > 7).then(pl.lit("Foo")).alias("bar"),
    )

    expected = pl.DataFrame(
        {
            "literal": ["Foo", "Foo", "Foo", None, None, None],
            "bar": ["Foo", "Foo", "Foo", None, None, None],
        }
    )
    assert_frame_equal(result, expected)


def test_when_then_empty_list_5547() -> None:
    out = pl.DataFrame({"a": []}).select([pl.when(pl.col("a") > 1).then([1])])
    assert out.shape == (0, 1)
    assert out.dtypes == [pl.List(pl.Int64)]


def test_nested_when_then_and_wildcard_expansion_6284() -> None:
    df = pl.DataFrame(
        {
            "1": ["a", "b"],
            "2": ["c", "d"],
        }
    )

    out0 = df.with_columns(
        pl.when(pl.any_horizontal(pl.all() == "a"))
        .then(pl.lit("a"))
        .otherwise(
            pl.when(pl.any_horizontal(pl.all() == "d"))
            .then(pl.lit("d"))
            .otherwise(None)
        )
        .alias("result")
    )

    out1 = df.with_columns(
        pl.when(pl.any_horizontal(pl.all() == "a"))
        .then(pl.lit("a"))
        .when(pl.any_horizontal(pl.all() == "d"))
        .then(pl.lit("d"))
        .otherwise(None)
        .alias("result")
    )

    assert_frame_equal(out0, out1)
    assert out0.to_dict(as_series=False) == {
        "1": ["a", "b"],
        "2": ["c", "d"],
        "result": ["a", "d"],
    }


def test_list_zip_with_logical_type() -> None:
    df = pl.DataFrame(
        {
            "start": [datetime(2023, 1, 1, 1, 1, 1), datetime(2023, 1, 1, 1, 1, 1)],
            "stop": [datetime(2023, 1, 1, 1, 3, 1), datetime(2023, 1, 1, 1, 4, 1)],
            "use": [1, 0],
        }
    )

    df = df.with_columns(
        pl.datetime_ranges(
            pl.col("start"), pl.col("stop"), interval="1h", eager=False, closed="left"
        ).alias("interval_1"),
        pl.datetime_ranges(
            pl.col("start"), pl.col("stop"), interval="1h", eager=False, closed="left"
        ).alias("interval_2"),
    )

    out = df.select(
        pl.when(pl.col("use") == 1)
        .then(pl.col("interval_2"))
        .otherwise(pl.col("interval_1"))
        .alias("interval_new")
    )
    assert out.dtypes == [pl.List(pl.Datetime(time_unit="us", time_zone=None))]


def test_type_coercion_when_then_otherwise_2806() -> None:
    out = (
        pl.DataFrame({"names": ["foo", "spam", "spam"], "nrs": [1, 2, 3]})
        .select(
            pl.when(pl.col("names") == "spam")
            .then(pl.col("nrs") * 2)
            .otherwise(pl.lit("other"))
            .alias("new_col"),
        )
        .to_series()
    )
    expected = pl.Series("new_col", ["other", "4", "6"])
    assert out.to_list() == expected.to_list()

    # test it remains float32
    assert (
        pl.Series("a", [1.0, 2.0, 3.0], dtype=pl.Float32)
        .to_frame()
        .select(pl.when(pl.col("a") > 2.0).then(pl.col("a")).otherwise(0.0))
    ).to_series().dtype == pl.Float32


def test_when_then_edge_cases_3994() -> None:
    df = pl.DataFrame(data={"id": [1, 1], "type": [2, 2]})

    # this tests if lazy correctly assigns the list schema to the column aggregation
    assert (
        df.lazy()
        .group_by(["id"])
        .agg(pl.col("type"))
        .with_columns(
            pl.when(pl.col("type").list.len() == 0)
            .then(pl.lit(None))
            .otherwise(pl.col("type"))
            .name.keep()
        )
        .collect()
    ).to_dict(as_series=False) == {"id": [1], "type": [[2, 2]]}

    # this tests ternary with an empty argument
    assert (
        df.filter(pl.col("id") == 42)
        .group_by(["id"])
        .agg(pl.col("type"))
        .with_columns(
            pl.when(pl.col("type").list.len() == 0)
            .then(pl.lit(None))
            .otherwise(pl.col("type"))
            .name.keep()
        )
    ).to_dict(as_series=False) == {"id": [], "type": []}


def test_object_when_then_4702() -> None:
    # please don't ever do this
    x = pl.DataFrame({"Row": [1, 2], "Type": [pl.Date, pl.UInt8]})

    assert x.with_columns(
        pl.when(pl.col("Row") == 1)
        .then(pl.lit(pl.UInt16, allow_object=True))
        .otherwise(pl.lit(pl.UInt8, allow_object=True))
        .alias("New_Type")
    ).to_dict(as_series=False) == {
        "Row": [1, 2],
        "Type": [pl.Date, pl.UInt8],
        "New_Type": [pl.UInt16, pl.UInt8],
    }


def test_comp_categorical_lit_dtype() -> None:
    df = pl.DataFrame(
        data={"column": ["a", "b", "e"], "values": [1, 5, 9]},
        schema=[("column", pl.Categorical), ("more", pl.Int32)],
    )

    assert df.with_columns(
        pl.when(pl.col("column") == "e")
        .then(pl.lit("d"))
        .otherwise(pl.col("column"))
        .alias("column")
    ).dtypes == [pl.Categorical, pl.Int32]


def test_comp_incompatible_enum_dtype() -> None:
    df = pl.DataFrame({"a": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))})

    with pytest.raises(
        InvalidOperationError,
        match="conversion from `str` to `enum` failed in column 'literal'",
    ):
        df.with_columns(
            pl.when(pl.col("a") == "a").then(pl.col("a")).otherwise(pl.lit("c"))
        )


def test_predicate_broadcast() -> None:
    df = pl.DataFrame(
        {
            "key": ["a", "a", "b", "b", "c", "c"],
            "val": [1, 2, 3, 4, 5, 6],
        }
    )
    out = df.group_by("key", maintain_order=True).agg(
        agg=pl.when(pl.col("val").min() >= 3).then(pl.col("val")),
    )
    assert out.to_dict(as_series=False) == {
        "key": ["a", "b", "c"],
        "agg": [[None, None], [3, 4], [5, 6]],
    }


@pytest.mark.parametrize(
    "mask_expr",
    [
        pl.lit(True),
        pl.first("true"),
        pl.lit(False),
        pl.first("false"),
        pl.lit(None, dtype=pl.Boolean),
        pl.col("null_bool"),
        pl.col("true"),
        pl.col("false"),
    ],
)
@pytest.mark.parametrize(
    "truthy_expr",
    [
        pl.lit(1),
        pl.first("x"),
        pl.col("x"),
    ],
)
@pytest.mark.parametrize(
    "falsy_expr",
    [
        pl.lit(1),
        pl.first("x"),
        pl.col("x"),
    ],
)
def test_single_element_broadcast(
    mask_expr: pl.Expr,
    truthy_expr: pl.Expr,
    falsy_expr: pl.Expr,
) -> None:
    df = (
        pl.Series("x", 5 * [1], dtype=pl.Int32)
        .to_frame()
        .with_columns(true=True, false=False, null_bool=pl.lit(None, dtype=pl.Boolean))
    )

    # Given that the lengths of the mask, truthy and falsy are all either:
    # - Length 1
    # - Equal length to the maximum length of the 3.
    # This test checks that all length-1 exprs are broadcast to the max length.
    result = df.select(
        pl.when(mask_expr).then(truthy_expr.alias("x")).otherwise(falsy_expr)
    )
    expected = df.select("x").head(
        df.select(
            pl.max_horizontal(mask_expr.len(), truthy_expr.len(), falsy_expr.len())
        ).item()
    )
    assert_frame_equal(result, expected)

    result = (
        df.group_by(pl.lit(True).alias("key"))
        .agg(pl.when(mask_expr).then(truthy_expr.alias("x")).otherwise(falsy_expr))
        .drop("key")
    )
    if expected.height > 1:
        result = result.explode(pl.all())
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "df",
    [pl.DataFrame({"x": range(5)}), pl.DataFrame({"x": 5 * [[*range(5)]]})],
)
@pytest.mark.parametrize(
    "ternary_expr",
    [
        pl.when(True).then(pl.col("x").head(2)).otherwise(pl.col("x")),
        pl.when(False).then(pl.col("x").head(2)).otherwise(pl.col("x")),
    ],
)
def test_mismatched_height_should_raise(
    df: pl.DataFrame, ternary_expr: pl.Expr
) -> None:
    with pytest.raises(ShapeError):
        df.select(ternary_expr)

    with pytest.raises(ShapeError):
        df.group_by(pl.lit(True).alias("key")).agg(ternary_expr)


def test_when_then_output_name_12380() -> None:
    df = pl.DataFrame(
        {"x": range(5), "y": range(5, 10)}, schema={"x": pl.Int8, "y": pl.Int64}
    ).with_columns(true=True, false=False, null_bool=pl.lit(None, dtype=pl.Boolean))

    expect = df.select(pl.col("x").cast(pl.Int64))
    for true_expr in (pl.first("true"), pl.col("true"), pl.lit(True)):
        ternary_expr = pl.when(true_expr).then(pl.col("x")).otherwise(pl.col("y"))

        actual = df.select(ternary_expr)
        assert_frame_equal(
            expect,
            actual,
        )
        actual = (
            df.group_by(pl.lit(True).alias("key"))
            .agg(ternary_expr)
            .drop("key")
            .explode(pl.all())
        )
        assert_frame_equal(
            expect,
            actual,
        )

    expect = df.select(pl.col("y").alias("x"))
    for false_expr in (
        pl.first("false"),
        pl.col("false"),
        pl.lit(False),
        pl.first("null_bool"),
        pl.col("null_bool"),
        pl.lit(None, dtype=pl.Boolean),
    ):
        ternary_expr = pl.when(false_expr).then(pl.col("x")).otherwise(pl.col("y"))

        actual = df.select(ternary_expr)
        assert_frame_equal(
            expect,
            actual,
        )
        actual = (
            df.group_by(pl.lit(True).alias("key"))
            .agg(ternary_expr)
            .drop("key")
            .explode(pl.all())
        )
        assert_frame_equal(
            expect,
            actual,
        )


def test_when_then_nested_non_unit_literal_predicate_agg_broadcast_12242() -> None:
    df = pl.DataFrame(
        {
            "array_name": ["A", "A", "A", "B", "B"],
            "array_idx": [5, 0, 3, 7, 2],
            "array_val": [1, 2, 3, 4, 5],
        }
    )

    int_range = pl.int_range(pl.min("array_idx"), pl.max("array_idx") + 1)

    is_valid_idx = int_range.is_in("array_idx")

    idxs = is_valid_idx.cum_sum() - 1

    ternary_expr = pl.when(is_valid_idx).then(pl.col("array_val").gather(idxs))

    expect = pl.DataFrame(
        [
            pl.Series("array_name", ["A", "B"], dtype=pl.String),
            pl.Series(
                "array_val",
                [[1, None, None, 2, None, 3], [4, None, None, None, None, 5]],
                dtype=pl.List(pl.Int64),
            ),
        ]
    )

    assert_frame_equal(
        expect, df.group_by("array_name").agg(ternary_expr).sort("array_name")
    )


def test_when_then_non_unit_literal_predicate_agg_broadcast_12382() -> None:
    df = pl.DataFrame({"id": [1, 1], "value": [0, 3]})

    expect = pl.DataFrame({"id": [1], "literal": [["yes", None, None, "yes", None]]})
    actual = df.group_by("id").agg(
        pl.when(pl.int_range(0, 5).is_in("value")).then(pl.lit("yes"))
    )

    assert_frame_equal(expect, actual)


def test_when_then_binary_op_predicate_agg_12526() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1],
            "b": [1, 2, 5],
        }
    )

    expect = pl.DataFrame(
        {"a": [1], "col": [None]}, schema={"a": pl.Int64, "col": pl.String}
    )

    actual = df.group_by("a").agg(
        col=(
            pl.when(
                pl.col("a").shift(1) > 2,
                pl.col("b").is_not_null(),
            )
            .then(pl.lit("abc"))
            .when(
                pl.col("a").shift(1) > 1,
                pl.col("b").is_not_null(),
            )
            .then(pl.lit("def"))
            .otherwise(pl.lit(None))
            .first()
        )
    )

    assert_frame_equal(expect, actual)


def test_when_predicates_kwargs() -> None:
    df = pl.DataFrame(
        {
            "x": [10, 20, 30, 40],
            "y": [15, -20, None, 1],
            "z": ["a", "b", "c", "d"],
        }
    )
    assert_frame_equal(  # kwargs only
        df.select(matched=pl.when(x=30, z="c").then(True).otherwise(False)),
        pl.DataFrame({"matched": [False, False, True, False]}),
    )
    assert_frame_equal(  # mixed predicates & kwargs
        df.select(matched=pl.when(pl.col("x") < 30, z="b").then(True).otherwise(False)),
        pl.DataFrame({"matched": [False, True, False, False]}),
    )
    assert_frame_equal(  # chained when/then with mixed predicates/kwargs
        df.select(
            misc=pl.when(pl.col("x") > 50)
            .then(pl.lit("x>50"))
            .when(y=1)
            .then(pl.lit("y=1"))
            .when(pl.col("z").is_in(["a", "b"]), pl.col("y") < 0)
            .then(pl.lit("z in (a|b), y<0"))
            .otherwise(pl.lit("?"))
        ),
        pl.DataFrame({"misc": ["?", "z in (a|b), y<0", "?", "y=1"]}),
    )


def test_when_then_null_broadcast() -> None:
    assert (
        pl.select(
            pl.when(pl.repeat(True, 2, dtype=pl.Boolean)).then(
                pl.repeat(None, 1, dtype=pl.Null)
            )
        ).height
        == 2
    )


@pytest.mark.slow()
@pytest.mark.parametrize("len", [1, 10, 100, 500])
@pytest.mark.parametrize(
    ("dtype", "vals"),
    [
        pytest.param(pl.Boolean, [False, True], id="Boolean"),
        pytest.param(pl.UInt8, [0, 1], id="UInt8"),
        pytest.param(pl.UInt16, [0, 1], id="UInt16"),
        pytest.param(pl.UInt32, [0, 1], id="UInt32"),
        pytest.param(pl.UInt64, [0, 1], id="UInt64"),
        pytest.param(pl.Float32, [0.0, 1.0], id="Float32"),
        pytest.param(pl.Float64, [0.0, 1.0], id="Float64"),
        pytest.param(pl.String, ["0", "12"], id="String"),
        pytest.param(pl.Array(pl.String, 2), [["0", "1"], ["3", "4"]], id="StrArray"),
        pytest.param(pl.Array(pl.Int64, 2), [[0, 1], [3, 4]], id="IntArray"),
        pytest.param(pl.List(pl.String), [["0"], ["1", "2"]], id="List"),
        pytest.param(
            pl.Struct({"foo": pl.Int32, "bar": pl.String}),
            [{"foo": 0, "bar": "1"}, {"foo": 1, "bar": "2"}],
            id="Struct",
        ),
        pytest.param(pl.Object, ["x", "y"], id="Object"),
    ],
)
@pytest.mark.parametrize("broadcast", list(itertools.product([False, True], repeat=3)))
def test_when_then_parametric(
    len: int, dtype: pl.DataType, vals: list[Any], broadcast: list[bool]
) -> None:
    # Makes no sense to broadcast all columns.
    if all(broadcast):
        return

    rng = random.Random(42)

    for _ in range(10):
        mask = rng.choices([False, True, None], k=len)
        if_true = rng.choices(vals + [None], k=len)
        if_false = rng.choices(vals + [None], k=len)

        py_mask, py_true, py_false = (
            [c[0]] * len if b else c
            for b, c in zip(broadcast, [mask, if_true, if_false])
        )
        pl_mask, pl_true, pl_false = (
            c.first() if b else c
            for b, c in zip(broadcast, [pl.col.mask, pl.col.if_true, pl.col.if_false])
        )

        ref = pl.DataFrame(
            {"if_true": [t if m else f for m, t, f in zip(py_mask, py_true, py_false)]},
            schema={"if_true": dtype},
        )
        df = pl.DataFrame(
            {
                "mask": mask,
                "if_true": if_true,
                "if_false": if_false,
            },
            schema={"mask": pl.Boolean, "if_true": dtype, "if_false": dtype},
        )

        ans = df.select(pl.when(pl_mask).then(pl_true).otherwise(pl_false))
        if dtype != pl.Object:
            assert_frame_equal(ref, ans)
        else:
            assert ref["if_true"].to_list() == ans["if_true"].to_list()


def test_when_then_supertype_15975() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert df.with_columns(
        pl.when(True).then(1 ** pl.col("a") + 1.0 * pl.col("a"))
    ).to_dict(as_series=False) == {"a": [1, 2, 3], "literal": [2.0, 3.0, 4.0]}


def test_when_then_supertype_15975_comment() -> None:
    df = pl.LazyFrame({"foo": [1, 3, 4], "bar": [3, 4, 0]})

    q = df.with_columns(
        pl.when(pl.col("foo") == 1)
        .then(1)
        .when(pl.col("foo") == 2)
        .then(4)
        .when(pl.col("foo") == 3)
        .then(1.5)
        .when(pl.col("foo") == 4)
        .then(16)
        .otherwise(0)
        .alias("val")
    )

    assert q.collect()["val"].to_list() == [1.0, 1.5, 16.0]


def test_chained_when_no_subclass_17142() -> None:
    # https://github.com/pola-rs/polars/pull/17142
    when = pl.when(True).then(1).when(True)

    assert not isinstance(when, pl.Expr)
    assert "<polars.expr.whenthen.ChainedWhen object at" in str(when)

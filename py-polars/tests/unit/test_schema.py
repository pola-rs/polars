import pickle
from datetime import datetime
from typing import Any

import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal


def test_schema() -> None:
    s = pl.Schema({"foo": pl.Int8(), "bar": pl.String()})

    assert s["foo"] == pl.Int8()
    assert s["bar"] == pl.String()
    assert s.len() == 2
    assert s.names() == ["foo", "bar"]
    assert s.dtypes() == [pl.Int8(), pl.String()]

    with pytest.raises(
        TypeError,
        match="dtypes must be fully-specified, got: List",
    ):
        pl.Schema({"foo": pl.String, "bar": pl.List})


@pytest.mark.parametrize(
    "schema",
    [
        pl.Schema(),
        pl.Schema({"foo": pl.Int8()}),
        pl.Schema({"foo": pl.Datetime("us"), "bar": pl.String()}),
        pl.Schema(
            {
                "foo": pl.UInt32(),
                "bar": pl.Categorical("physical"),
                "baz": pl.Struct({"x": pl.Int64(), "y": pl.Float64()}),
            }
        ),
    ],
)
def test_schema_empty_frame(schema: pl.Schema) -> None:
    assert_frame_equal(
        schema.to_frame(),
        pl.DataFrame(schema=schema),
    )


def test_schema_equality() -> None:
    s1 = pl.Schema({"foo": pl.Int8(), "bar": pl.Float64()})
    s2 = pl.Schema({"foo": pl.Int8(), "bar": pl.String()})
    s3 = pl.Schema({"bar": pl.Float64(), "foo": pl.Int8()})

    assert s1 == s1
    assert s2 == s2
    assert s3 == s3
    assert s1 != s2
    assert s1 != s3
    assert s2 != s3

    s4 = pl.Schema({"foo": pl.Datetime("us"), "bar": pl.Duration("ns")})
    s5 = pl.Schema({"foo": pl.Datetime("ns"), "bar": pl.Duration("us")})
    s6 = {"foo": pl.Datetime, "bar": pl.Duration}

    assert s4 != s5
    assert s4 != s6


def test_schema_parse_python_dtypes() -> None:
    cardinal_directions = pl.Enum(["north", "south", "east", "west"])

    s = pl.Schema({"foo": pl.List(pl.Int32), "bar": int, "baz": cardinal_directions})  # type: ignore[arg-type]
    s["ham"] = datetime

    assert s["foo"] == pl.List(pl.Int32)
    assert s["bar"] == pl.Int64
    assert s["baz"] == cardinal_directions
    assert s["ham"] == pl.Datetime("us")

    assert s.len() == 4
    assert s.names() == ["foo", "bar", "baz", "ham"]
    assert s.dtypes() == [pl.List, pl.Int64, cardinal_directions, pl.Datetime("us")]

    assert list(s.to_python().values()) == [list, int, str, datetime]
    assert [tp.to_python() for tp in s.dtypes()] == [list, int, str, datetime]


def test_schema_picklable() -> None:
    s = pl.Schema(
        {
            "foo": pl.Int8(),
            "bar": pl.String(),
            "ham": pl.Struct({"x": pl.List(pl.Date)}),
        }
    )
    pickled = pickle.dumps(s)
    s2 = pickle.loads(pickled)
    assert s == s2


def test_schema_python() -> None:
    input = {
        "foo": pl.Int8(),
        "bar": pl.String(),
        "baz": pl.Categorical("lexical"),
        "ham": pl.Object(),
        "spam": pl.Struct({"time": pl.List(pl.Duration), "dist": pl.Float64}),
    }
    expected = {
        "foo": int,
        "bar": str,
        "baz": str,
        "ham": object,
        "spam": dict,
    }
    for schema in (input, input.items(), list(input.items())):
        s = pl.Schema(schema)
        assert expected == s.to_python()


def test_schema_in_map_elements_returns_scalar() -> None:
    schema = pl.Schema([("portfolio", pl.String()), ("irr", pl.Float64())])

    ldf = pl.LazyFrame(
        {
            "portfolio": ["A", "A", "B", "B"],
            "amounts": [100.0, -110.0] * 2,
        }
    )
    q = ldf.group_by("portfolio").agg(
        pl.col("amounts")
        .map_elements(
            lambda x: float(x.sum()), return_dtype=pl.Float64, returns_scalar=True
        )
        .alias("irr")
    )
    assert q.collect_schema() == schema
    assert q.collect().schema == schema


def test_ir_cache_unique_18198() -> None:
    lf = pl.LazyFrame({"a": [1]})
    lf.collect_schema()
    assert pl.concat([lf, lf]).collect().to_dict(as_series=False) == {"a": [1, 1]}


def test_schema_functions_in_agg_with_literal_arg_19011() -> None:
    q = (
        pl.LazyFrame({"a": [1, 2, 3, None, 5]})
        .rolling(index_column=pl.int_range(pl.len()).alias("idx"), period="3i")
        .agg(pl.col("a").fill_null(0).alias("a_1"), pl.col("a").pow(2.0).alias("a_2"))
    )
    assert q.collect_schema() == pl.Schema(
        [("idx", pl.Int64), ("a_1", pl.List(pl.Int64)), ("a_2", pl.List(pl.Float64))]
    )


def test_lazy_explode_in_agg_schema_19562() -> None:
    def new_df_check_schema(
        value: dict[str, Any], schema: dict[str, Any]
    ) -> pl.DataFrame:
        df = pl.DataFrame(value)
        assert df.schema == schema
        return df

    lf = pl.LazyFrame({"a": [1], "b": [[1]]})

    q = lf.group_by("a").agg(pl.col("b"))
    schema = {"a": pl.Int64, "b": pl.List(pl.List(pl.Int64))}

    assert q.collect_schema() == schema
    assert_frame_equal(
        q.collect(), new_df_check_schema({"a": [1], "b": [[[1]]]}, schema)
    )

    q = lf.group_by("a").agg(pl.col("b").explode())
    schema = {"a": pl.Int64, "b": pl.List(pl.Int64)}

    assert q.collect_schema() == schema
    assert_frame_equal(q.collect(), new_df_check_schema({"a": [1], "b": [[1]]}, schema))

    q = lf.group_by("a").agg(pl.col("b").explode().explode())
    schema = {"a": pl.Int64, "b": pl.List(pl.Int64)}

    assert q.collect_schema() == schema
    assert_frame_equal(q.collect(), new_df_check_schema({"a": [1], "b": [[1]]}, schema))

    # 2x nested
    lf = pl.LazyFrame({"a": [1], "b": [[[1]]]})

    q = lf.group_by("a").agg(pl.col("b"))
    schema = {
        "a": pl.Int64,
        "b": pl.List(pl.List(pl.List(pl.Int64))),
    }

    assert q.collect_schema() == schema
    assert_frame_equal(
        q.collect(), new_df_check_schema({"a": [1], "b": [[[[1]]]]}, schema)
    )

    q = lf.group_by("a").agg(pl.col("b").explode())
    schema = {"a": pl.Int64, "b": pl.List(pl.List(pl.Int64))}

    assert q.collect_schema() == schema
    assert_frame_equal(
        q.collect(), new_df_check_schema({"a": [1], "b": [[[1]]]}, schema)
    )

    q = lf.group_by("a").agg(pl.col("b").explode().explode())
    schema = {"a": pl.Int64, "b": pl.List(pl.Int64)}

    assert q.collect_schema() == schema
    assert_frame_equal(q.collect(), new_df_check_schema({"a": [1], "b": [[1]]}, schema))


def test_lazy_nested_function_expr_agg_schema() -> None:
    q = (
        pl.LazyFrame({"k": [1, 1, 2]})
        .group_by(pl.first(), maintain_order=True)
        .agg(o=pl.int_range(pl.len()).reverse() < 1)
    )

    assert q.collect_schema() == {"k": pl.Int64, "o": pl.List(pl.Boolean)}
    assert_frame_equal(
        q.collect(), pl.DataFrame({"k": [1, 2], "o": [[False, True], [True]]})
    )


def test_lazy_agg_scalar_return_schema() -> None:
    q = pl.LazyFrame({"k": [1]}).group_by("k").agg(pl.col("k").null_count().alias("o"))

    schema = {"k": pl.Int64, "o": pl.UInt32}
    assert q.collect_schema() == schema
    assert_frame_equal(q.collect(), pl.DataFrame({"k": 1, "o": 0}, schema=schema))


def test_lazy_agg_nested_expr_schema() -> None:
    q = (
        pl.LazyFrame({"k": [1]})
        .group_by("k")
        .agg(
            (
                (
                    (pl.col("k").reverse().shuffle() + 1)
                    + pl.col("k").shuffle().reverse()
                )
                .shuffle()
                .reverse()
                .sum()
                * 0
            ).alias("o")
        )
    )

    schema = {"k": pl.Int64, "o": pl.Int64}
    assert q.collect_schema() == schema
    assert_frame_equal(q.collect(), pl.DataFrame({"k": 1, "o": 0}, schema=schema))


def test_lazy_agg_lit_explode() -> None:
    q = (
        pl.LazyFrame({"k": [1]})
        .group_by("k")
        .agg(pl.lit(1, dtype=pl.Int64).explode().alias("o"))
    )

    schema = {"k": pl.Int64, "o": pl.List(pl.Int64)}
    assert q.collect_schema() == schema
    assert_frame_equal(q.collect(), pl.DataFrame({"k": 1, "o": [[1]]}, schema=schema))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "expr_op", [
        "approx_n_unique", "arg_max", "arg_min", "bitwise_and", "bitwise_or",
        "bitwise_xor", "count", "entropy", "first", "has_nulls", "implode", "kurtosis",
        "last", "len", "lower_bound", "max", "mean", "median", "min", "n_unique", "nan_max",
        "nan_min", "null_count", "product", "sample", "skew", "std", "sum", "upper_bound",
        "var"
    ]
)  # fmt: skip
@pytest.mark.parametrize("lhs", [pl.col("b"), pl.lit(1, dtype=pl.Int64).alias("b")])
def test_lazy_agg_to_scalar_schema_19752(lhs: pl.Expr, expr_op: str) -> None:
    op = getattr(pl.Expr, expr_op)

    lf = pl.LazyFrame({"a": 1, "b": 1})

    q = lf.group_by("a").agg(lhs.reverse().pipe(op))
    assert q.collect_schema() == q.collect().collect_schema()

    q = lf.group_by("a").agg(lhs.shuffle().reverse().pipe(op))

    assert q.collect_schema() == q.collect().collect_schema()


def test_lazy_agg_schema_after_elementwise_19984() -> None:
    lf = pl.LazyFrame({"a": 1, "b": 1})

    q = lf.group_by("a").agg(pl.col("b").first().fill_null(0))
    assert q.collect_schema() == q.collect().collect_schema()

    q = lf.group_by("a").agg(pl.col("b").first().fill_null(0).fill_null(0))
    assert q.collect_schema() == q.collect().collect_schema()

    q = lf.group_by("a").agg(pl.col("b").first() + 1)
    assert q.collect_schema() == q.collect().collect_schema()

    q = lf.group_by("a").agg(1 + pl.col("b").first())
    assert q.collect_schema() == q.collect().collect_schema()


@pytest.mark.parametrize(
    "expr", [pl.col("b"), pl.col("b").sum(), pl.col("b").reverse()]
)
@pytest.mark.parametrize("mapping_strategy", ["explode", "join", "group_to_rows"])
def test_lazy_window_schema(expr: pl.Expr, mapping_strategy: str) -> None:
    q = pl.LazyFrame({"a": 1, "b": 1}).select(
        expr.over("a", mapping_strategy=mapping_strategy)  # type: ignore[arg-type]
    )

    assert q.collect_schema() == q.collect().collect_schema()


def test_lazy_explode_schema() -> None:
    lf = pl.LazyFrame({"k": [1], "x": pl.Series([[1]], dtype=pl.Array(pl.Int64, 1))})

    q = lf.select(pl.col("x").explode())
    assert q.collect_schema() == {"x": pl.Int64}

    q = lf.select(pl.col("x").arr.explode())
    assert q.collect_schema() == {"x": pl.Int64}

    lf = pl.LazyFrame({"k": [1], "x": pl.Series([[1]], dtype=pl.List(pl.Int64))})

    q = lf.select(pl.col("x").explode())
    assert q.collect_schema() == {"x": pl.Int64}

    q = lf.select(pl.col("x").list.explode())
    assert q.collect_schema() == {"x": pl.Int64}

    # `LazyFrame.explode()` goes through a different codepath than `Expr.expode`
    lf = pl.LazyFrame().with_columns(
        pl.Series([[1]], dtype=pl.List(pl.Int64)).alias("list"),
        pl.Series([[1]], dtype=pl.Array(pl.Int64, 1)).alias("array"),
    )

    q = lf.explode("*")
    assert q.collect_schema() == {"list": pl.Int64, "array": pl.Int64}

    q = lf.explode("list")
    assert q.collect_schema() == {"list": pl.Int64, "array": pl.Array(pl.Int64, 1)}


def test_raise_subnodes_18787() -> None:
    df = pl.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        (
            df.select(pl.struct(pl.all())).select(
                pl.first().struct.field("a", "b").filter(pl.col("foo") == 1)
            )
        )


def test_scalar_agg_schema_20044() -> None:
    assert (
        pl.DataFrame(None, schema={"a": pl.Int64, "b": pl.String, "c": pl.String})
        .with_columns(d=pl.col("a").max())
        .group_by("c")
        .agg(pl.col("d").mean())
    ).schema == pl.Schema([("c", pl.String), ("d", pl.Float64)])

from typing import Any, Literal

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.may_fail_auto_streaming
@pytest.mark.may_fail_cloud
def test_invalid_broadcast() -> None:
    df = pl.DataFrame(
        {
            "a": [100, 103],
            "group": [0, 1],
        }
    )
    with pytest.raises(pl.exceptions.ShapeError):
        df.select(pl.col("group").filter(pl.col("group") == 0), "a")


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Null,
        pl.Int32,
        pl.String,
        pl.Enum(["foo"]),
        pl.Binary,
        pl.List(pl.Int32),
        pl.Struct({"a": pl.Int32}),
        pl.Array(pl.Int32, 1),
        pl.List(pl.List(pl.Int32)),
    ],
)
def test_null_literals(dtype: pl.DataType) -> None:
    assert (
        pl.DataFrame([pl.Series("a", [1, 2], pl.Int64)])
        .with_columns(pl.lit(None).cast(dtype).alias("b"))
        .collect_schema()
        .dtypes()
    ) == [pl.Int64, dtype]


def test_scalar_19957() -> None:
    value = 1
    values = [value] * 5
    foo = pl.DataFrame({"foo": values})
    foo_with_bar_from_literal = foo.with_columns(pl.lit(value).alias("bar"))
    assert foo_with_bar_from_literal.gather_every(2).to_dict(as_series=False) == {
        "foo": [1, 1, 1],
        "bar": [1, 1, 1],
    }


def test_scalar_len_20046() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert (
        df.lazy()
        .select(
            pl.col("a"),
            pl.lit(1),
        )
        .select(pl.len())
        .collect()
        .item()
        == 3
    )

    q = pl.LazyFrame({"a": range(3)}).select(
        pl.first("a"),
        pl.col("a").alias("b"),
    )

    assert q.select(pl.len()).collect().item() == 3


def test_scalar_identification_function_expr_in_binary() -> None:
    x = pl.Series("x", [1, 2, 3])
    assert_frame_equal(
        pl.select(x).with_columns(o=pl.col("x").null_count() > 0),
        pl.select(x, o=False),
    )


def test_scalar_rechunk_20627() -> None:
    df = pl.concat(2 * [pl.Series([1])]).filter(pl.Series([False, True])).to_frame()
    assert df.rechunk().to_series().n_chunks() == 1


def test_split_scalar_21581() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    df = df.with_columns(
        [
            pl.col("a").shift(-1).alias("next_a"),
            pl.lit(True).alias("lit"),
        ]
    )

    assert df.filter(df["next_a"] != 99.0).with_columns(
        [pl.lit(False).alias("lit")]
    ).to_dict(as_series=False) == {
        "a": [1.0, 2.0],
        "next_a": [2.0, 3.0],
        "lit": [False, False],
    }


def _broadcast_and_materialized(values: list[Any]) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    series = pl.Series("l", [values], dtype=pl.List(pl.Int64))
    broadcast = pl.LazyFrame({"a": [1, 2, 3]}).with_columns(pl.lit(series).first())
    materialized = pl.LazyFrame({"a": [1, 2, 3], "l": [values] * 3})
    return broadcast, materialized


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("l").list.len(),
        pl.col("l").list.sum(),
        pl.col("l").list.contains(30),
        pl.col("l").list.contains(pl.col("a")),
        pl.col("l").list.get(0, null_on_oob=True),
        pl.col("a").is_in(pl.col("l")),
        pl.col("a").is_in(pl.col("l"), nulls_equal=True),
    ],
)
def test_elementwise_over_broadcast_scalar(expr: pl.Expr) -> None:
    broadcast, materialized = _broadcast_and_materialized([10, None, 30])
    assert_frame_equal(
        broadcast.select(expr).collect(), materialized.select(expr).collect()
    )


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("l").arr.contains(30),
        pl.col("l").arr.contains(pl.col("a")),
        pl.col("a").is_in(pl.col("l")),
    ],
)
def test_elementwise_over_broadcast_scalar_array(expr: pl.Expr) -> None:
    values = [10, 20, 30]
    dtype = pl.Array(pl.Int64, len(values))
    broadcast = pl.LazyFrame({"a": [1, 2, 30]}).with_columns(
        pl.lit(pl.Series("l", [values], dtype=dtype)).first()
    )
    materialized = pl.LazyFrame(
        {"a": [1, 2, 30], "l": pl.Series([values] * 3, dtype=dtype)}
    )
    assert_frame_equal(
        broadcast.select(expr).collect(), materialized.select(expr).collect()
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(pl.col("lhs").arr.dot("rhs"), id="scalar-rhs"),
        pytest.param(pl.col("rhs").arr.dot("lhs"), id="scalar-lhs"),
    ],
)
def test_arr_dot_over_broadcast_scalar_array(
    engine: Literal["in-memory", "streaming"], expr: pl.Expr
) -> None:
    dtype = pl.Array(pl.Int64, 3)
    lhs = pl.Series("lhs", [[1, 2, 3], [4, None, 6], None], dtype=dtype)
    rhs = pl.Series("rhs", [[10, None, 30]], dtype=dtype)
    broadcast = pl.LazyFrame({"lhs": lhs}).with_columns(pl.lit(rhs).first())
    materialized = pl.LazyFrame(
        {
            "lhs": lhs,
            "rhs": pl.Series("rhs", [[10, None, 30]] * len(lhs), dtype=dtype),
        }
    )
    assert_frame_equal(
        broadcast.select(expr).collect(engine=engine),
        materialized.select(expr).collect(engine=engine),
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_arr_dot_over_empty_broadcast_scalar_array(
    engine: Literal["in-memory", "streaming"],
) -> None:
    dtype = pl.Array(pl.Float64, 2)
    broadcast = pl.LazyFrame(schema={"lhs": dtype}).with_columns(
        pl.lit(pl.Series("rhs", [[10.0, 20.0]], dtype=dtype)).first()
    )
    materialized = pl.LazyFrame(schema={"lhs": dtype, "rhs": dtype})
    expr = pl.col("lhs").arr.dot("rhs")

    assert_frame_equal(
        broadcast.select(expr).collect(engine=engine),
        materialized.select(expr).collect(engine=engine),
    )


def test_elementwise_over_empty_scalar() -> None:
    df = pl.DataFrame({"a": [1]}).with_columns(b=pl.lit(5)).head(0)
    assert df.select(pl.col("b").is_null()).to_dict(as_series=False) == {"b": []}

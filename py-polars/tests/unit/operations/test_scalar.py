import io
from typing import Any, Literal

import pytest

import polars as pl
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal


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


def _reprs(df: pl.DataFrame) -> list[str]:
    return df._to_metadata()["repr"].to_list()


def test_vstack_scalar_columns_stay_scalar() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]}).with_columns(i=pl.lit(5), s=pl.lit("x"))
    assert _reprs(df) == ["series", "scalar", "scalar"]

    out = pl.concat([df, df, df], rechunk=False)
    assert _reprs(out) == ["series", "scalar", "scalar"]
    assert out.to_dict(as_series=False) == {
        "a": [1, 2, 3] * 3,
        "i": [5] * 9,
        "s": ["x"] * 9,
    }


def test_extend_scalar_columns_stay_scalar() -> None:
    df = pl.DataFrame({"a": [1, 2]}).with_columns(i=pl.lit(5))
    other = pl.DataFrame({"a": [3, 4]}).with_columns(i=pl.lit(5))

    df.extend(other)

    assert _reprs(df) == ["series", "scalar"]
    assert df.to_dict(as_series=False) == {"a": [1, 2, 3, 4], "i": [5] * 4}


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected_repr", "expected"),
    [
        (pl.lit(5), pl.lit(5), "scalar", [5] * 4),
        (pl.lit(5), pl.lit(6), "series", [5, 5, 6, 6]),
        (
            pl.lit("x", dtype=pl.Categorical),
            pl.lit("x", dtype=pl.Categorical),
            "scalar",
            ["x"] * 4,
        ),
        (
            pl.lit(None, dtype=pl.Int64),
            pl.lit(None, dtype=pl.Int64),
            "scalar",
            [None] * 4,
        ),
        (
            pl.lit(None, dtype=pl.Int64),
            pl.lit(7, dtype=pl.Int64),
            "series",
            [None, None, 7, 7],
        ),
    ],
)
def test_vstack_scalar(
    lhs: pl.Expr, rhs: pl.Expr, expected_repr: str, expected: list[Any]
) -> None:
    a = pl.DataFrame({"a": [1, 2]}).with_columns(v=lhs)
    b = pl.DataFrame({"a": [3, 4]}).with_columns(v=rhs)

    out = a.vstack(b)

    assert _reprs(out) == ["series", expected_repr]
    assert out["v"].to_list() == expected


def test_vstack_scalar_empty_frame() -> None:
    df = pl.DataFrame({"a": [1, 2]}).with_columns(i=pl.lit(5))
    empty = df.clear()
    expected = {"a": [1, 2], "i": [5, 5]}

    assert empty.vstack(df).to_dict(as_series=False) == expected
    assert df.vstack(empty).to_dict(as_series=False) == expected
    assert empty.vstack(empty).to_dict(as_series=False) == {"a": [], "i": []}


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (pl.lit(0.0), pl.lit(-0.0)),
        (pl.lit([0.0]), pl.lit([-0.0])),
        (pl.struct(x=pl.lit(0.0)), pl.struct(x=pl.lit(-0.0))),
    ],
)
def test_vstack_scalar_signed_zero(lhs: pl.Expr, rhs: pl.Expr) -> None:
    # 0.0 and -0.0 are equal, but not the same value. Both have to be kept.
    pos = pl.DataFrame({"a": [1, 2]}).with_columns(z=lhs)
    neg = pl.DataFrame({"a": [3, 4]}).with_columns(z=rhs)

    out = pos.vstack(neg)

    assert _reprs(out) == ["series", "series"]
    assert str(out["z"].to_list()) == str(pos["z"].to_list() + neg["z"].to_list())


def test_vstack_scalar_nan() -> None:
    df = pl.DataFrame({"a": [1, 2]}).with_columns(f=pl.lit(float("nan")))

    out = df.vstack(df)

    assert _reprs(out) == ["series", "scalar"]
    assert out["f"].is_nan().to_list() == [True] * 4


def test_vstack_scalar_dtype_mismatch_still_raises() -> None:
    value = pl.DataFrame({"a": [1]}).with_columns(v=pl.lit(1.0))
    null = pl.DataFrame({"a": [2]}).with_columns(v=pl.lit(None, dtype=pl.Null))

    # A null column is cast into the existing dtype; the reverse raises.
    assert value.vstack(null)["v"].to_list() == [1.0, None]
    with pytest.raises(SchemaError):
        null.vstack(value)


class _AlwaysEqual:
    """An object whose `__eq__` reports equality with everything."""

    def __init__(self, i: int) -> None:
        self.i = i

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


def test_append_object_scalars_not_merged() -> None:
    # Object equality runs Python code and does not have to mean the values are the
    # same. `vstack` reaches `Column::append` and `pl.concat` reaches `append_owned`.
    a, b = (
        pl.DataFrame({"o": pl.Series([_AlwaysEqual(i)], dtype=pl.Object)})
        for i in range(2)
    )

    for out in [a.vstack(b), pl.concat([a, b])]:
        assert _reprs(out) == ["series"]
        assert [o.i for o in out["o"]] == [0, 1]


def test_write_materialized_scalar_beside_chunked_column() -> None:
    # The scalar column comes first and is materialized, but it must not decide the
    # chunk layout of the frame.
    df = pl.concat(
        [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3]})], rechunk=False
    ).select(s=pl.lit("x"), a=pl.col("a"))
    assert _reprs(df) == ["scalar", "series"]
    assert df.n_chunks("all") == [1, 2]
    df.get_column("s")

    expected = {"s": ["x"] * 3, "a": [1, 2, 3]}
    buf = io.BytesIO()
    df.write_parquet(buf, row_group_size=2)
    buf.seek(0)
    assert pl.read_parquet(buf).to_dict(as_series=False) == expected
    buf = io.BytesIO()
    df.write_ipc(buf)
    buf.seek(0)
    assert pl.read_ipc(buf).to_dict(as_series=False) == expected
    assert df.to_arrow().to_pydict() == expected


def test_concat_scalar_run_keeps_sorted_flag() -> None:
    # The first two frames become one scalar column of two rows. A run of one value
    # is sorted in both directions, so the descending flag must survive.
    tail = pl.DataFrame({"a": [3, 2]}).sort("a", descending=True)
    out = pl.concat([pl.DataFrame({"a": [5]}), pl.DataFrame({"a": [5]}), tail])

    assert out["a"].to_list() == [5, 5, 3, 2]
    assert out["a"].flags["SORTED_DESC"]

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import SchemaError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType


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


def test_vstack_scalar_signed_zero() -> None:
    # Polars treats -0.0 and 0.0 as the same value (`Series.unique` collapses them), so
    # the fast path may merge them, keeping the left-hand value.
    pos = pl.DataFrame({"a": [1, 2]}).with_columns(z=pl.lit(0.0))
    neg = pl.DataFrame({"a": [3, 4]}).with_columns(z=pl.lit(-0.0))

    out = pos.vstack(neg)

    assert _reprs(out) == ["series", "scalar"]
    assert np.signbit(out["z"].to_numpy()).tolist() == [False] * 4


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
    """An object whose `__eq__` reports equality with everything.

    Mirrors `Foo` in `test_hashing_on_python_objects`. Polars requires object equality
    to mean the values are the same, so it is entitled to treat these as one value -- as
    `group_by` and `unique` already do.
    """

    def __init__(self, i: int) -> None:
        self.i = i

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


class _Distinct:
    """An object with well-behaved equality: distinct values compare unequal."""

    def __init__(self, i: int) -> None:
        self.i = i

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Distinct) and self.i == other.i

    def __hash__(self) -> int:
        return hash(self.i)


def _object_frame(value: object) -> pl.DataFrame:
    return pl.DataFrame({"o": pl.Series("o", [value], dtype=pl.Object)})


def test_append_distinct_object_scalars_not_merged() -> None:
    # Distinct values must survive the fast path. `vstack` reaches `Column::append` and
    # `pl.concat` reaches `append_owned`, so both are worth checking.
    a, b = _object_frame(_Distinct(1)), _object_frame(_Distinct(2))

    assert [o.i for o in a.vstack(b)["o"]] == [1, 2]
    assert [o.i for o in pl.concat([a, b])["o"]] == [1, 2]


def test_scalar_object_column_is_sorted() -> None:
    # A constant column is sorted whatever it holds, `Object` included. Consumers have
    # to cope with that: `unique` used to pick the row-encoding `SortedUnique` strategy
    # off the back of it and panic.
    df = _object_frame(_AlwaysEqual(1)).vstack(_object_frame(_AlwaysEqual(2)))
    md = df._to_metadata(stats=["column_name", "repr", "sorted_asc"])

    assert md["repr"].to_list() == ["scalar"]
    assert md["sorted_asc"].to_list() == [True]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_object_column_unique_after_append(engine: EngineType) -> None:
    # Regression test: the streaming engine appends per-morsel results, and the
    # resulting scalar object column used to panic in `unique` via row encoding.
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]}).with_columns(
        pl.col("a").map_elements(_AlwaysEqual, return_dtype=pl.Object).alias("o")
    )
    df = lf.collect(engine=engine)

    assert df.select("o").unique().height == 1
    assert df.unique().height == 4

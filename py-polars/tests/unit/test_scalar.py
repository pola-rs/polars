import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.may_fail_auto_streaming
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

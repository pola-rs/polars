from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def set_nulls(s: pl.Series, mask: list[bool]) -> pl.Series:
    return pl.select(pl.when(pl.Series(mask)).then(s).alias(s.name)).to_series()


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize(
    "nulls",
    [
        [True] * 3,
        [False, True, True],
        [True, False, True],
        [True, True, False],
        [False, False, True],
        [True, False, False],
        [False] * 3,
    ],
)
def test_eval_basic(as_list: bool, nulls: list[bool]) -> None:
    if as_list:

        def rtdt(dt: pl.DataType) -> pl.DataType:
            return pl.List(dt)
    else:

        def rtdt(dt: pl.DataType) -> pl.DataType:
            return pl.Array(dt, 2)

    s = set_nulls(
        pl.Series("a", [[1, 4], [8, 5], [3, 2]], pl.Array(pl.Int64(), 2)), nulls
    )

    assert_series_equal(
        s.arr.eval(pl.element().rank(), as_list=as_list),
        set_nulls(
            pl.Series("a", [[1.0, 2.0], [2.0, 1.0], [2.0, 1.0]], rtdt(pl.Float64())),
            nulls,
        ),
    )
    assert_series_equal(
        s.arr.eval(pl.element() + 1, as_list=as_list),
        set_nulls(pl.Series("a", [[2, 5], [9, 6], [4, 3]], rtdt(pl.Int64())), nulls),
    )
    assert_series_equal(
        s.arr.eval(pl.element().cast(pl.String()), as_list=as_list),
        s.cast(rtdt(pl.Int64())).cast(rtdt(pl.String())),
    )

    if as_list:
        assert_series_equal(
            s.arr.eval(pl.element().unique(maintain_order=True), as_list=True),
            s.cast(rtdt(pl.Int64())),
        )


def test_eval_raises_for_non_length_preserving() -> None:
    s = pl.Series(
        "a", [["A", "B", "C"], ["C", "C", "D"], ["D", "E", "E"]], pl.Array(pl.String, 3)
    )

    with pytest.raises(pl.exceptions.InvalidOperationError, match="as_list"):
        s.arr.eval(pl.element().unique(maintain_order=True))


@pytest.mark.parametrize(
    "nulls",
    [
        [True] * 3,
        [False, True, True],
        [True, False, True],
        [True, True, False],
        [False, False, True],
        [True, False, False],
        [False] * 3,
    ],
)
def test_eval_changing_length(nulls: list[bool]) -> None:
    s = set_nulls(
        pl.Series(
            "a",
            [["A", "B", "C"], ["C", "C", "D"], ["D", "E", "E"]],
            pl.Array(pl.String, 3),
        ),
        nulls,
    )

    assert_series_equal(
        s.arr.eval(pl.element().unique(maintain_order=True), as_list=True),
        set_nulls(
            pl.Series(
                "a", [["A", "B", "C"], ["C", "D"], ["D", "E"]], pl.List(pl.String)
            ),
            nulls,
        ),
    )


def set_validity(s: pl.Series, validity: list[bool]) -> pl.Series:
    return s.zip_with(pl.Series(validity), pl.Series([None], dtype=s.dtype))


@pytest.mark.parametrize(
    "sum_expr",
    [pl.element().sum(), pl.element().unique().sum(), pl.element().fill_null(1).sum()],
)
def test_arr_agg_sum(sum_expr: pl.Expr) -> None:
    assert_series_equal(
        pl.Series("a", [], pl.Array(pl.Int64, 2)).arr.agg(sum_expr),
        pl.Series("a", [], pl.Int64),
    )

    assert_series_equal(
        pl.Series("a", [[0, 1, 2], [1, 3, 5]], pl.Array(pl.Int64, 3)).arr.agg(sum_expr),
        pl.Series("a", [3, 9]),
    )

    assert_series_equal(
        pl.Series("a", [[], []], pl.Array(pl.Int64, 0)).arr.agg(sum_expr),
        pl.Series("a", [0, 0]),
    )

    assert_series_equal(
        pl.Series("a", [None, [1, 3, 5]], pl.Array(pl.Int64, 3)).arr.agg(sum_expr),
        pl.Series("a", [None, 9]),
    )

    assert_series_equal(
        set_validity(
            pl.Series("a", [[1, 2, 3], [3, 4, 5], [1, 3, 5]], pl.Array(pl.Int64, 3)),
            [True, False, True],
        ).arr.agg(sum_expr),
        pl.Series("a", [6, None, 9]),
    )


@pytest.mark.parametrize(
    ("expr", "is_scalar"),
    [
        (pl.Expr.null_count, True),
        (lambda e: e.rank().null_count(), True),
        (pl.Expr.rank, False),
        (lambda e: e + pl.lit(1), False),
        (lambda e: e.filter(e != 0), False),
        (pl.Expr.drop_nulls, False),
        (pl.Expr.n_unique, True),
    ],
)
def test_arr_agg_parametric(
    expr: Callable[[pl.Expr], pl.Expr], is_scalar: bool
) -> None:
    def test_case(s: pl.Series) -> None:
        out = s.arr.agg(expr(pl.element()))

        for i, v in enumerate(s):
            if v is None:
                assert out[i] is None
                continue

            assert isinstance(v, pl.Series)

            v = v.rename("")
            v = v.to_frame().select(expr(pl.col(""))).to_series()

            if not is_scalar:
                v = v.implode()

            assert_series_equal(out.rename("").slice(i, 1), v)

    test_case(pl.Series("a", [], pl.Array(pl.Int64, 2)))
    test_case(pl.Series("a", [[]], pl.Array(pl.Int64, 0)))
    test_case(pl.Series("a", [[7], [0]], pl.Array(pl.Int64, 1)))
    test_case(pl.Series("a", [[8], [0], None], pl.Array(pl.Int64, 1)))
    test_case(pl.Series("a", [None, [0], None], pl.Array(pl.Int64, 1)))
    test_case(pl.Series("a", [[1, 2, 3], [4, 5, 6]], pl.Array(pl.Int64, 3)))


@pytest.mark.parametrize("insert_none", [False, True])
@pytest.mark.parametrize("keys", [pl.lit(42), pl.col.g])
@pytest.mark.parametrize("filter", [None, pl.lit(True), pl.col.b])
@pytest.mark.parametrize(
    ("expr", "as_list", "result"),
    [
        (
            pl.element(),
            False,
            pl.Series("a", [[0, 1, 2], [5, 3, 4], [7, 7, 8]], pl.Array(pl.Int64, 3)),
        ),
        (
            pl.element() + pl.element(),
            False,
            pl.Series(
                "a", [[0, 2, 4], [10, 6, 8], [14, 14, 16]], pl.Array(pl.Int64, 3)
            ),
        ),
        (
            pl.element().rank(),
            False,
            pl.Series(
                "a",
                [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [1.5, 1.5, 3.0]],
                pl.Array(pl.Float64, 3),
            ),
        ),
        (pl.element().unique(), True, pl.Series("a", [[0, 1, 2], [5, 3, 4], [7, 8]])),
    ],
)
def test_arr_eval_with_filter_in_agg_25384(
    insert_none: bool,
    keys: pl.Expr,
    filter: pl.Expr | None,
    expr: pl.Expr,
    as_list: bool,
    result: pl.Series,
) -> None:
    s = pl.Series("a", [[0, 1, 2], [5, 3, 4], [7, 7, 8]], pl.Array(pl.Int64, 3))
    df = s.to_frame().with_columns(
        pl.Series("g", [10, 10, 20]), pl.Series("b", [True, True, True])
    )
    q_inner = (
        pl.col("a").arr.eval(expr, as_list=as_list)
        if filter is None
        else pl.col("a").filter(filter).arr.eval(expr, as_list=as_list)
    )

    if insert_none:
        df = df.with_columns(
            pl.when(pl.int_range(0, pl.len()) != 1).then(pl.col.a).otherwise(None)
        )
        result = (
            result.to_frame()
            .with_columns(
                pl.when(pl.int_range(0, pl.len()) != 1).then(pl.col.a).otherwise(None)
            )
            .to_series()
        )

    # no agg
    q = df.lazy().select(q_inner)
    assert_series_equal(q.collect().to_series(), result)

    # over
    q = df.lazy().select(q_inner.over(keys))
    assert_series_equal(q.collect().to_series(), result)

    # group_by
    q = df.lazy().group_by(keys, maintain_order=True).agg(q_inner)
    out = q.collect().select(pl.col.a).explode("a")
    assert_series_equal(out.to_series(), result)


@pytest.mark.parametrize("insert_none", [False, True])
@pytest.mark.parametrize("keys", [pl.lit(42), pl.col.g])
@pytest.mark.parametrize("filter", [None, pl.lit(True), pl.col.b])
@pytest.mark.parametrize(
    ("expr", "result"),
    [
        (pl.element().sum(), pl.Series("a", [1, 8, 22])),
        (pl.element().null_count(), pl.Series("a", [1, 1, 0], pl.get_index_type())),
    ],
)
def test_arr_agg_with_filter_in_agg_25384(
    insert_none: bool,
    keys: pl.Expr,
    filter: pl.Expr | None,
    expr: pl.Expr,
    result: pl.Series,
) -> None:
    s = pl.Series("a", [[0, 1, None], [5, 3, None], [7, 7, 8]], pl.Array(pl.Int64, 3))
    df = s.to_frame().with_columns(
        pl.Series("g", [10, 10, 20]), pl.Series("b", [True, True, True])
    )
    q_inner = (
        pl.col("a").arr.agg(expr)
        if filter is None
        else pl.col("a").filter(filter).arr.agg(expr)
    )

    if insert_none:
        df = df.with_columns(
            pl.when(pl.int_range(0, pl.len()) != 1).then(pl.col.a).otherwise(None)
        )
        result = (
            result.to_frame()
            .with_columns(
                pl.when(pl.int_range(0, pl.len()) != 1).then(pl.col.a).otherwise(None)
            )
            .to_series()
        )

    # no agg
    q = df.lazy().select(q_inner)
    assert_series_equal(q.collect().to_series(), result)

    # over
    q = df.lazy().select(q_inner.over(keys))
    assert_series_equal(q.collect().to_series(), result)

    # group_by
    q = df.lazy().group_by(keys, maintain_order=True).agg(q_inner)
    out = q.collect().select(pl.col.a).explode("a")
    assert_series_equal(out.to_series(), result)

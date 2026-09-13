from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import (
    StructFieldNotFoundError,
)
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable


def test_list_eval_dtype_inference() -> None:
    grades = pl.DataFrame(
        {
            "student": ["bas", "laura", "tim", "jenny"],
            "arithmetic": [10, 5, 6, 8],
            "biology": [4, 6, 2, 7],
            "geography": [8, 4, 9, 7],
        }
    )

    rank_pct = pl.col("").rank(descending=True) / pl.col("").count().cast(pl.UInt16)

    # the .list.first() would fail if .list.eval did not correctly infer the output type
    assert grades.with_columns(
        pl.concat_list(pl.all().exclude("student")).alias("all_grades")
    ).select(
        pl.col("all_grades")
        .list.eval(rank_pct, parallel=True)
        .alias("grades_rank")
        .list.first()
    ).to_series().to_list() == [
        0.3333333333333333,
        0.6666666666666666,
        0.6666666666666666,
        0.3333333333333333,
    ]


def test_list_eval_categorical() -> None:
    df = pl.DataFrame({"test": [["a", None]]}, schema={"test": pl.List(pl.Categorical)})
    df = df.select(
        pl.col("test").list.eval(pl.element().filter(pl.element().is_not_null()))
    )
    assert_series_equal(
        df.get_column("test"), pl.Series("test", [["a"]], dtype=pl.List(pl.Categorical))
    )


def test_list_eval_cast_categorical() -> None:
    df = pl.DataFrame({"test": [["a", None], ["c"], [], ["a", "b", "c"]]})
    expected = pl.DataFrame(
        {"test": [["a", None], ["c"], [], ["a", "b", "c"]]},
        schema={"test": pl.List(pl.Categorical)},
    )
    result = df.select(pl.col("test").list.eval(pl.element().cast(pl.Categorical)))
    assert_frame_equal(result, expected)


def test_list_eval_type_coercion() -> None:
    last_non_null_value = pl.element().fill_null(3).last()
    df = pl.DataFrame({"array_cols": [[1, None]]})

    assert df.select(
        pl.col("array_cols")
        .list.eval(last_non_null_value, parallel=False)
        .alias("col_last")
    ).to_dict(as_series=False) == {"col_last": [[3]]}


def test_list_eval_all_null() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, None, None]}).with_columns(
        pl.col("bar").cast(pl.List(pl.String))
    )

    assert df.select(pl.col("bar").list.eval(pl.element())).to_dict(
        as_series=False
    ) == {"bar": [None, None, None]}


def test_empty_eval_dtype_5546() -> None:
    # https://github.com/pola-rs/polars/issues/5546
    df = pl.DataFrame([{"a": [{"name": 1}, {"name": 2}]}])

    dtype = df.dtypes[0]

    assert (
        df.limit(0).with_columns(
            pl.col("a")
            .list.eval(pl.element().filter(pl.element().struct.field("name") == 1))
            .alias("a_filtered")
        )
    ).dtypes == [dtype, dtype]


def test_list_eval_gather_every_13410() -> None:
    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]]})
    out = df.with_columns(result=pl.col("a").list.eval(pl.element().gather_every(2)))
    expected = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6]], "result": [[1, 3], [4, 6]]})
    assert_frame_equal(out, expected)


def test_list_eval_err_raise_15653() -> None:
    df = pl.DataFrame({"foo": [[]]})
    with pytest.raises(StructFieldNotFoundError):
        df.with_columns(bar=pl.col("foo").list.eval(pl.element().struct.field("baz")))


def test_list_eval_type_cast_11188() -> None:
    df = pl.DataFrame(
        [
            {"a": None},
        ],
        schema={"a": pl.List(pl.Int64)},
    )
    assert df.select(
        pl.col("a").list.eval(pl.element().cast(pl.String)).alias("a_str")
    ).schema == {"a_str": pl.List(pl.String)}


@pytest.mark.parametrize(
    "data",
    [
        {"a": [["0"], ["1"]]},
        {"a": [["0", "1"], ["2", "3"]]},
        {"a": [["0", "1"]]},
        {"a": [["0"]]},
    ],
)
@pytest.mark.parametrize(
    "expr",
    [
        pl.lit(""),
        pl.format("test: {}", pl.element()),
    ],
)
def test_list_eval_list_output_18510(data: dict[str, Any], expr: pl.Expr) -> None:
    df = pl.DataFrame(data)
    result = df.select(pl.col("a").list.eval(expr))
    assert result.to_series().dtype == pl.List(pl.String)


def test_list_eval_when_then_23089() -> None:
    assert_series_equal(
        pl.Series([[1, 2]]).list.eval(pl.when(pl.int_range(pl.len()) > 0).then(42)),
        pl.Series([[None, 42]]),
        check_dtypes=False,
    )


def test_list_eval_selectors_23187() -> None:
    df = pl.DataFrame({"x": [[{"id": "foo"}]]})
    assert_frame_equal(
        df.with_columns(pl.col("x").list.eval(pl.element().struct[0])),
        pl.DataFrame({"x": [["foo"]]}),
    )


def test_list_eval_in_filter_23300() -> None:
    df = pl.DataFrame({"a": [[{"r": "n"}], [{"r": "ab"}]]})
    assert (
        df.filter(
            pl.col("a").list.eval(pl.element().struct.field("r") == "n").list.any()
        ).height
        == 1
    )


@pytest.mark.parametrize(
    "ldf",
    [
        pl.LazyFrame(
            {"a": [[1, 2, 3], [6, 4, 5], [7, 9, 8]], "id": [1, 1, 2]},
        ),
        pl.LazyFrame(
            {"a": [[{"b": 5}, {"b": 6}], [{"c": 7}]], "id": [1, 2]},
        ),
        pl.LazyFrame(
            {"a": [[]], "id": [1]},
        ),
        pl.LazyFrame(
            {"a": [[{}]], "id": [1]},
        ),
    ],
)
@pytest.mark.parametrize(
    "expr",
    [
        pl.lit(""),
        pl.element(),
        pl.element().is_not_null(),
        pl.element().first(),
        pl.element().sum(),
        pl.element().min(),
        pl.element().rank(),
        pl.element().get(0),
        pl.element().gather([0]),
    ],
)
def test_list_eval_in_group_by_schema(ldf: pl.LazyFrame, expr: pl.Expr) -> None:
    q_select = ldf.select(x=pl.col("a").list.eval(expr))
    q_group_by = ldf.group_by("id").agg(x=pl.col("a").list.eval(expr))
    q_over = ldf.select(x=pl.col("a").list.eval(expr).over("id"))

    # skip index 0 on the empty list
    skip = ("get" in str(expr) or "gather" in str(expr)) and ldf.select(
        pl.col("a").first().list.len()
    ).collect().to_series()[0] == 0

    # skip sum on struct types
    dtype = ldf.collect_schema()["a"]
    assert isinstance(dtype, pl.List)
    skip = skip or ("sum" in str(expr) and isinstance(dtype.inner, pl.Struct))

    for q in [q_select, q_group_by, q_over]:
        if not skip:
            assert q.collect_schema() == q.collect().schema


def test_list_eval_in_group_by_value() -> None:
    ldf = pl.LazyFrame(
        {"a": [[1, 2, 3], [6, 4, 5], [7, 9, 8]], "id": [1, 1, 2]},
    )

    expr = pl.element().first()

    # select
    q = ldf.select(x=pl.col("a").list.eval(expr))
    expected = pl.Series("x", [[1], [6], [7]])
    assert_series_equal(q.collect().to_series(), expected)

    # group_by
    q = ldf.group_by("id").agg(x=pl.col("a").list.eval(expr)).select("x")
    expected = pl.Series("x", [[[1], [6]], [[7]]])
    assert_series_equal(q.collect().to_series().sort(), expected.sort())

    # over
    q = ldf.select(x=pl.col("a").list.eval(expr).over("id"))
    expected = pl.Series("x", [[1], [6], [7]])
    assert_series_equal(q.collect().to_series().sort(), expected.sort())


def test_list_eval_struct_in_group_by_23846() -> None:
    dict = {"b": 5}
    ldf = pl.LazyFrame(
        {"a": [[dict, dict], [dict]], "id": [1, 2]},
    )

    expr = pl.element().struct.field("b")

    # select
    q = ldf.select(x=pl.col("a").list.eval(expr))
    expected = pl.Series("x", [[5, 5], [5]])
    assert_series_equal(q.collect().to_series(), expected)
    assert q.collect_schema() == q.collect().schema

    # group_by
    q = ldf.group_by("id").agg(x=pl.col("a").list.eval(expr)).select("x")
    expected = pl.Series("x", [[[5, 5]], [[5]]])
    assert_series_equal(q.collect().to_series().sort(), expected.sort())
    assert q.collect_schema() == q.collect().schema

    # over
    q = ldf.select(x=pl.col("a").list.eval(expr).over("id"))
    expected = pl.Series("x", [[5, 5], [5]])
    assert_series_equal(q.collect().to_series().sort(), expected.sort())
    assert q.collect_schema() == q.collect().schema


@pytest.mark.parametrize("filter_flag", [True, False])
@pytest.mark.parametrize(
    "col",
    [
        [],
        [1, 2, 3],
        [[1, 2], [3]],
    ],
)
def test_cumulative_eval_on_empty_list_schema_24635(
    col: list[Any], filter_flag: bool
) -> None:
    df = pl.DataFrame({"n": col})

    # over
    q = (
        df.lazy()
        # Force empty with a filter that removes everything
        .filter(pl.lit(filter_flag))
        .select(pl.col("n").cumulative_eval(pl.element().last()).over(1))
    )
    expected = df.head(df.height if filter_flag else 0)
    assert_frame_equal(q.collect(), expected)
    assert q.collect_schema() == q.collect().schema

    # group_by
    q = (
        df.lazy()
        # Force empty with a filter that removes everything
        .filter(pl.lit(filter_flag))
        .group_by([1])
        .agg(pl.col("n").cumulative_eval(pl.element().last()))
    )
    assert q.collect_schema() == q.collect().schema


def set_validity(s: pl.Series, validity: list[bool]) -> pl.Series:
    return s.zip_with(pl.Series(validity), pl.Series([None], dtype=s.dtype))


@pytest.mark.parametrize(
    "sum_expr",
    [pl.element().sum(), pl.element().unique().sum(), pl.element().fill_null(1).sum()],
)
def test_list_agg_sum(sum_expr: pl.Expr) -> None:
    assert_series_equal(
        pl.Series("a", [], pl.List(pl.Int64)).list.agg(sum_expr),
        pl.Series("a", [], pl.Int64),
    )

    assert_series_equal(
        pl.Series("a", [[0, 1, 2], [1, 3, 5]]).list.agg(sum_expr),
        pl.Series("a", [3, 9]),
    )

    assert_series_equal(
        pl.Series("a", [[], []], pl.List(pl.Int64)).list.agg(sum_expr),
        pl.Series("a", [0, 0]),
    )

    assert_series_equal(
        pl.Series("a", [None, [1, 3, 5]]).list.agg(sum_expr),
        pl.Series("a", [None, 9]),
    )

    assert_series_equal(
        set_validity(
            pl.Series("a", [[1, 2, 3], [3], [1, 3, 5]]), [True, False, True]
        ).list.agg(sum_expr),
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
def test_list_agg_parametric(
    expr: Callable[[pl.Expr], pl.Expr], is_scalar: bool
) -> None:
    def test_case(s: pl.Series) -> None:
        out = s.list.agg(expr(pl.element()))

        for i, v in enumerate(s):
            if v is None:
                assert out[i] is None
                continue

            assert isinstance(v, pl.Series)

            v = v.to_frame().select(expr(pl.col(""))).to_series()

            if not is_scalar:
                v = v.implode()

            assert_series_equal(out.rename("").slice(i, 1), v)

    test_case(pl.Series("a", [], pl.List(pl.Int64)))
    test_case(pl.Series("a", [[]], pl.List(pl.Int64)))
    test_case(pl.Series("a", [[], [0]]))
    test_case(pl.Series("a", [[], [0], None]))
    test_case(pl.Series("a", [None, [0], None]))
    test_case(pl.Series("a", [[1, 2, 3], [4, 5]]))


def test_list_eval_matching_slice_lengths() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4]]})
    out = df.select(
        pl.col.a.list.eval(
            (pl.element().slice(0, 1) * (pl.element().slice(1, 1))).sum()
        )
    )
    expected = pl.DataFrame({"a": [[2], [12]]})
    assert_frame_equal(out, expected)


def test_unique_in_list_agg() -> None:
    df = pl.DataFrame({"a": [[1, 2, 3]]}).select(
        uniq=pl.col.a.list.agg(pl.element().first().unique()),
        drop_nulls=pl.col.a.list.agg(pl.element().first().drop_nulls()),
    )

    assert_frame_equal(
        df,
        pl.DataFrame({"uniq": [[1]], "drop_nulls": [[1]]}),
    )


@pytest.mark.parametrize(
    ("df"),
    [
        pl.DataFrame({"a": [[1, 2, 3], [4], [5]]}),
        pl.DataFrame({"a": [[1, 2, 3], None, [5]]}),
        pl.DataFrame({"a": [[1, 2, 3], [None], [5]]}),
        pl.DataFrame({"a": [[1, 2, 3], [], [5]]}),
        pl.DataFrame({"a": [[None, None, 3], [4], [5]]}),
    ],
)
@pytest.mark.parametrize("expr", [pl.element(), 2 * pl.element() - pl.element()])
def test_list_eval_parametric_element(df: pl.DataFrame, expr: pl.Expr) -> None:
    out = df.select(pl.col.a.list.eval(expr))
    assert_frame_equal(out, df)

    out = df.select(pl.col.a.list.eval(expr).over(42))
    assert_frame_equal(out, df)


@pytest.mark.parametrize(
    ("df"),
    [
        pl.DataFrame({"a": [[1, 2, 3], [1], [1]]}),
        pl.DataFrame({"a": [[1, 2, 3], None, [1]]}),
        pl.DataFrame({"a": [[1, 2, 3], [None], [1]]}),
        pl.DataFrame({"a": [[1, 2, 3], [], [1]]}),
        pl.DataFrame({"a": [[None, None, 1], [1], [1]]}),
    ],
)
@pytest.mark.parametrize("expr", [pl.element().rank()])
def test_list_eval_parametric_rank(df: pl.DataFrame, expr: pl.Expr) -> None:
    out = df.select(pl.col.a.list.eval(expr))
    expected = df.cast(pl.List(pl.Float64))
    assert_frame_equal(out, expected)

    out = df.select(pl.col.a.list.eval(expr).over(42))
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("df", "expected"),
    [
        (
            pl.DataFrame({"a": [[1, 2, 3], [1], [1]]}),
            pl.DataFrame({"a": [[1], [1], [1]]}),
        ),
        (
            pl.DataFrame({"a": [[1, 2, 3], None, [1]]}),
            pl.DataFrame({"a": [[1], None, [1]]}),
        ),
        (
            pl.DataFrame({"a": [[1, 2, 3], [None], [1]]}),
            pl.DataFrame({"a": [[1], [None], [1]]}),
        ),
        (
            pl.DataFrame({"a": [[1, 2, 3], [], [1]]}),
            pl.DataFrame({"a": [[1], [None], [1]]}),
        ),
        (
            pl.DataFrame({"a": [[None, None, 1], [1], [1]]}),
            pl.DataFrame({"a": [[None], [1], [1]]}),
        ),
    ],
)
@pytest.mark.parametrize(
    "expr",
    [
        pl.element().first(),
        pl.element().get(0, null_on_oob=True),
        pl.element().reverse().last(),
    ],
)
def test_list_eval_parametric_first_scalar(
    df: pl.DataFrame, expected: pl.DataFrame, expr: pl.Expr
) -> None:
    out = df.select(pl.col.a.list.eval(expr))
    assert_frame_equal(out, expected)

    out = df.select(pl.col.a.list.eval(expr).over(42))
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("df", "expected"),
    [
        (
            pl.DataFrame({"a": [[1, 2, 3], [1], [1]]}),
            pl.DataFrame({"a": [[1], [1], [1]]}),
        ),
        (
            pl.DataFrame({"a": [[1, 2, 3], None, [1]]}),
            pl.DataFrame({"a": [[1], None, [1]]}),
        ),
        (
            pl.DataFrame({"a": [[1, 2, 3], [None], [1]]}),
            pl.DataFrame({"a": [[1], [None], [1]]}),
        ),
        (
            pl.DataFrame({"a": [[1, 2, 3], [], [1]]}),
            pl.DataFrame({"a": [[1], [], [1]]}),
        ),
        (
            pl.DataFrame({"a": [[None, None, 1], [1], [1]]}),
            pl.DataFrame({"a": [[None], [1], [1]]}),
        ),
    ],
)
@pytest.mark.parametrize(
    "expr", [pl.element().head(1), pl.element().reverse().slice(-1)]
)
def test_list_eval_parametric_first_slice(
    df: pl.DataFrame, expected: pl.DataFrame, expr: pl.Expr
) -> None:
    out = df.select(pl.col.a.list.eval(expr))
    assert_frame_equal(out, expected)

    out = df.select(pl.col.a.list.eval(expr).over(42))
    assert_frame_equal(out, expected)


def test_list_eval_ternary() -> None:
    df = pl.DataFrame({"a": [[1, 2, 3], [4], [5]]})
    expr = pl.col.a.list.eval(
        pl.when(pl.element() % 2 == 0)
        .then(pl.element() // 2)
        .otherwise(pl.element() + pl.element() + pl.element() + 1)
    )
    expected = pl.DataFrame({"a": [[4, 1, 10], [2], [16]]})
    q = df.lazy().select(expr)
    assert_frame_equal(q.collect(), expected)

    q = df.lazy().select(expr.over([1]))
    assert_frame_equal(q.collect(), expected)

    q = df.lazy().select(expr.filter(pl.lit(True)).over([1]))
    assert_frame_equal(q.collect(), expected)

    expected = pl.DataFrame({"a": [[2, 4, 5], [1], [8]]})
    q = df.lazy().select(expr).lazy().select(expr)
    assert_frame_equal(q.collect(), expected)


@pytest.mark.parametrize(
    "series_a",
    [
        pl.Series("a", [[1, 2, 3], [4, 5], [6]]),
        pl.Series("a", [[1, 2, 3], None, [6]]),
    ],
)
@pytest.mark.parametrize("keys", [pl.col.g, pl.lit(42)])
@pytest.mark.parametrize("predicate", [pl.col.b, pl.lit(True)])
@pytest.mark.parametrize("maintain_order", [True, False])
def test_list_eval_after_filter_in_agg_25361(
    series_a: pl.Series, keys: pl.Expr, predicate: pl.Expr, maintain_order: bool
) -> None:
    df = pl.DataFrame({"a": series_a, "b": [True, True, True], "g": [10, 10, 10]})

    # group_by
    expected = df.select(pl.col.a.implode())
    q = (
        df.lazy()
        .group_by(keys, maintain_order=maintain_order)
        .agg(pl.col.a.filter(predicate).list.eval(pl.element()))
        .select(pl.col.a)
    )
    out = q.collect()
    assert_frame_equal(out, expected, check_row_order=maintain_order)
    assert out.item().len() == df.height

    # over
    q = df.lazy().select(pl.col.a.filter(predicate).list.eval(pl.element()).over(keys))
    assert_frame_equal(q.collect(), df.select(pl.col.a))
    assert q.collect().height == df.height


@pytest.mark.parametrize(
    "series_a",
    [
        [[1, 2, 3], [4, 5], [6]],
        [[1, 2, 3], None, [6]],
        [None, None, None],
    ],
)
def test_list_eval_after_arange_in_agg_25361(series_a: list[list[int]]) -> None:
    df = pl.DataFrame({"a": series_a})
    inner = pl.list(pl.arange(pl.len()))
    expected = df.select(inner)
    q = df.lazy().select(inner.list.eval(pl.element()).over([True]))
    assert_frame_equal(q.collect(), expected)


@pytest.mark.parametrize(
    "series_a",
    [
        pl.Series("a", [[1, 2, 3], [4, 5], [6]]),
        pl.Series("a", [[1, 2, 3], None, [6]]),
    ],
)
@pytest.mark.parametrize("keys", [pl.col.g, pl.lit(42)])
@pytest.mark.parametrize("predicate", [pl.col.b, pl.lit(True)])
def test_list_agg_after_filter_in_agg_25361(
    series_a: pl.Series, keys: pl.Expr, predicate: pl.Expr
) -> None:
    df = pl.DataFrame({"a": series_a, "b": [True, True, True], "g": [10, 10, 10]})
    q = df.lazy().select(
        pl.col.a.filter(pl.lit(True)).list.agg(pl.element().sum()).over([True])
    )
    out = q.collect()
    expected = (
        df.lazy().select(pl.col.a.list.agg(pl.element().sum()).over([True])).collect()
    )
    assert_frame_equal(out, expected)
    assert out.select(pl.col.a.count()).item() == df.select(pl.col.a.count()).item()


def test_array_eval_after_slice() -> None:
    df = pl.DataFrame({"array": [[0], [1]]}, schema={"array": pl.Array(pl.Int64, 1)})
    df = pl.concat([df, df.slice(1)], rechunk=False)

    assert_frame_equal(
        df.select(pl.col("array").arr.eval(pl.element().first(), as_list=True)),
        pl.DataFrame({"array": [[0], [1], [1]]}),
    )


def test_list_eval_after_slice() -> None:
    df = pl.DataFrame({"list": [[1], range(1, 10), range(1, 10)]})
    df_concat = pl.concat([df, df.slice(1)])
    grouped_result = df_concat.select_seq(
        has_no_duplicates=(
            ~pl.col("list").list.eval(pl.element().is_duplicated()).list.any()
        )
    )
    assert all(grouped_result.to_series().to_list())

    df = pl.DataFrame({"list": [[0], [1]]})
    df = pl.concat([df, df.slice(1)], rechunk=False)

    assert_frame_equal(
        df.select(pl.col("list").list.eval(pl.element().first())),
        pl.DataFrame({"list": [[0], [1], [1]]}),
    )


def test_list_agg_after_slice() -> None:
    df = pl.DataFrame({"list": [[1], range(1, 10), range(1, 10)]})
    df_concat = pl.concat([df, df.slice(1)])
    grouped_result = df_concat.select_seq(
        has_no_duplicates=~pl.col("list").list.agg(pl.element().is_duplicated().any())
    )
    assert all(grouped_result.to_series().to_list())


def test_list_eval_categorical_min_max_25906() -> None:
    s = pl.Series(
        "a",
        [["c", "a", "b"], ["z", None, "m"], [None, None]],
        dtype=pl.List(pl.Categorical),
    )
    assert_series_equal(
        s.list.agg(pl.element().min()),
        pl.Series("a", ["a", "m", None], dtype=pl.Categorical),
    )
    assert_series_equal(
        s.list.agg(pl.element().max()),
        pl.Series("a", ["c", "z", None], dtype=pl.Categorical),
    )
    assert_series_equal(
        s.list.agg(pl.element().arg_min()),
        pl.Series("a", [1, 2, None], dtype=pl.get_index_type()),
    )
    assert_series_equal(
        s.list.agg(pl.element().arg_max()),
        pl.Series("a", [0, 0, None], dtype=pl.get_index_type()),
    )


def test_list_eval_groupby_sample_25796() -> None:
    df = pl.DataFrame({"g": [10, 10], "x": [[1, 1], [1, 1]]})
    out = df.group_by("g").agg(pl.col("x").sample(n=2).list.eval(pl.element()))
    expected = pl.DataFrame({"g": [10], "x": [[[1, 1], [1, 1]]]})
    assert_frame_equal(out, expected)


def test_list_eval_returns_scalar_groups_update_26850() -> None:
    s = pl.Series([[1, 2], [], [3]])

    assert_series_equal(
        s.list.eval(pl.element().first()),
        pl.Series([[1], [None], [3]]),
    )


def test_list_agg_nulls_panic_26237() -> None:
    df = pl.DataFrame({"a": [[1], None, [], None, []]})

    assert_frame_equal(
        df.with_columns(
            first=pl.col("a").list.agg(pl.element().first()),
            sum=pl.col("a").list.agg(pl.element().sum()),
        ),
        pl.DataFrame(
            {
                "a": [[1], None, [], None, []],
                "first": [1, None, None, None, None],
                "sum": [1, None, 0, None, 0],
            }
        ),
    )


def test_list_eval_column_reference() -> None:
    df = pl.DataFrame({"x": [[1, 2, 3], [4, 5]], "y": [10, 20]})
    result = df.select(pl.col("x").list.eval(pl.element() + pl.col("y")))
    expected = pl.DataFrame({"x": [[11, 12, 13], [24, 25]]})
    assert_frame_equal(result, expected)


def test_list_eval_column_reference_with_nulls() -> None:
    df = pl.DataFrame({"x": [[1, 2], None, [3, 4]], "y": [10, 20, 30]})
    result = df.select(pl.col("x").list.eval(pl.element() + pl.col("y")))
    expected = pl.DataFrame({"x": [[11, 12], None, [33, 34]]})
    assert_frame_equal(result, expected)


def test_list_eval_column_reference_in_group_by() -> None:
    df = pl.DataFrame(
        {"g": [1, 1, 2], "x": [[1, 2], [3, 4], [5, 6]], "y": [10, 20, 30]}
    )
    result = (
        df.group_by("g", maintain_order=True)
        .agg(pl.col("x").list.eval(pl.element() + pl.col("y")))
        .sort("g")
    )
    expected = pl.DataFrame({"g": [1, 2], "x": [[[11, 12], [23, 24]], [[35, 36]]]})
    assert_frame_equal(result, expected)


def test_list_agg_column_reference() -> None:
    df = pl.DataFrame({"x": [[1, 2, 3], [4, 5]], "y": [10, 20]})
    result = df.select(pl.col("x").list.agg((pl.element() + pl.col("y")).sum()))
    expected = pl.DataFrame({"x": [36, 49]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("f", "is_scalar"),
    [(lambda a, b: a.list.eval(b), False), (lambda a, b: a.list.agg(b), True)],
)
def test_named_ref_broadcast(
    f: Callable[[pl.Expr, pl.Expr], pl.Expr], is_scalar: bool
) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 1, 1, 2],
            "values": [2.0, 3.0, 5.0, 1.0, 4.0],
        }
    )
    assert_frame_equal(
        df.group_by("a").agg(b=f(pl.lit([]), pl.col.values * 2.0)),
        pl.DataFrame({"a": [1, 2], "b": [[4.0, 10.0, 2.0], [6.0, 8.0]]})
        if is_scalar
        else pl.DataFrame(
            {"a": [1, 2], "b": [[[4.0], [10.0], [2.0]], [[6.0], [8.0]]]},
            schema={"a": pl.Int64, "b": pl.List(pl.List(pl.Float64))},
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df.select(b=f(pl.lit([]), pl.col.values * 2.0)),
        pl.DataFrame({"b": [4.0, 6.0, 10.0, 2.0, 8.0]})
        if is_scalar
        else pl.DataFrame(
            {"b": [[4.0], [6.0], [10.0], [2.0], [8.0]]},
            schema={"b": pl.List(pl.Float64)},
        ),
    )


def test_nested_list_eval_list_agg_col_ref() -> None:
    df = pl.DataFrame(
        {
            "lst": [[[1.0, 2.0], [3.0]], [[4.0, 5.0, 6.0]]],
            "values": [2.0, 3.0],
        },
        schema={"lst": pl.List(pl.List(pl.Float64)), "values": pl.Float64},
    )
    result = df.select(
        pl.col("lst").list.eval(
            pl.element().list.agg(pl.element().sum() * pl.col("values"))
        )
    )
    assert_frame_equal(
        result,
        pl.DataFrame({"lst": [[6.0, 6.0], [45.0]]}),
    )


def test_nested_list_eval_list_agg_col_ref_with_nulls() -> None:
    df = pl.DataFrame(
        {
            "lst": [
                [[1.0, 2.0, 3.0], None, [2.0, 3.0]],
                [None, [2.0, 5.0]],
                None,
                [[4.0]],
            ],
            "values": [2.0, 5.0, 3.0, 4.0],
        },
        schema={"lst": pl.List(pl.List(pl.Float64)), "values": pl.Float64},
    )
    result = df.select(
        pl.col("lst").list.eval(
            pl.element().list.agg(pl.element().sum() * pl.col("values"))
        )
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"lst": [[12.0, None, 10.0], [None, 35.0], None, [16.0]]},
            schema={"lst": pl.List(pl.Float64)},
        ),
    )


def test_nested_list_eval_list_eval_col_ref() -> None:
    df = pl.DataFrame(
        {
            "lst": [[[1, 2], [3, 4]], [[5, 6]]],
            "scale": [10, 100],
        },
        schema={"lst": pl.List(pl.List(pl.Int64)), "scale": pl.Int64},
    )
    result = df.select(
        pl.col("lst").list.eval(pl.element().list.eval(pl.element() * pl.col("scale")))
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"lst": [[[10, 20], [30, 40]], [[500, 600]]]},
            schema={"lst": pl.List(pl.List(pl.Int64))},
        ),
    )


def test_nested_list_eval_list_eval_col_ref_with_nulls() -> None:
    df = pl.DataFrame(
        {
            "lst": [[[1, 2], None, [3, 4]], None, [[5, 6]]],
            "scale": [10, 20, 100],
        },
        schema={"lst": pl.List(pl.List(pl.Int64)), "scale": pl.Int64},
    )
    result = df.select(
        pl.col("lst").list.eval(pl.element().list.eval(pl.element() * pl.col("scale")))
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"lst": [[[10, 20], None, [30, 40]], None, [[500, 600]]]},
            schema={"lst": pl.List(pl.List(pl.Int64))},
        ),
    )


def test_deeply_nested_list_eval_col_ref_three_levels() -> None:
    df = pl.DataFrame(
        {
            "x": [[[[1.0, 2.0], [3.0]], [[4.0]]], [[[5.0, 6.0]]]],
            "scale": [2.0, 3.0],
        },
        schema={"x": pl.List(pl.List(pl.List(pl.Float64))), "scale": pl.Float64},
    )
    result = df.select(
        pl.col("x").list.eval(
            pl.element().list.eval(
                pl.element().list.agg(pl.element().sum() * pl.col("scale"))
            )
        )
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"x": [[[6.0, 6.0], [8.0]], [[33.0]]]},
            schema={"x": pl.List(pl.List(pl.Float64))},
        ),
    )


def test_deeply_nested_list_eval_col_ref_three_levels_with_nulls() -> None:
    df = pl.DataFrame(
        {
            "x": [
                [[[1.0, 2.0], None, [3.0]], None],
                None,
                [[[4.0]]],
            ],
            "scale": [2.0, 5.0, 3.0],
        },
        schema={"x": pl.List(pl.List(pl.List(pl.Float64))), "scale": pl.Float64},
    )
    result = df.select(
        pl.col("x").list.eval(
            pl.element().list.eval(
                pl.element().list.agg(pl.element().sum() * pl.col("scale"))
            )
        )
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"x": [[[6.0, None, 6.0], None], None, [[12.0]]]},
            schema={"x": pl.List(pl.List(pl.Float64))},
        ),
    )


def test_deeply_nested_list_eval_col_ref_four_levels() -> None:
    df = pl.DataFrame(
        {
            "x": [[[[[1.0, 2.0], [3.0]]]], [[[[4.0, 5.0]]]]],
            "scale": [10.0, 100.0],
        },
        schema={
            "x": pl.List(pl.List(pl.List(pl.List(pl.Float64)))),
            "scale": pl.Float64,
        },
    )
    result = df.select(
        pl.col("x").list.eval(
            pl.element().list.eval(
                pl.element().list.eval(
                    pl.element().list.agg(pl.element().sum() * pl.col("scale"))
                )
            )
        )
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"x": [[[[30.0, 30.0]]], [[[900.0]]]]},
            schema={"x": pl.List(pl.List(pl.List(pl.Float64)))},
        ),
    )


@pytest.mark.parametrize(
    "f",
    [lambda a, b: a.list.eval(b), lambda a, b: a.list.agg(b)],
)
def test_scalar_shift(f: Callable[[pl.Expr, pl.Expr], pl.Expr]) -> None:
    df = pl.DataFrame({"lst": [[1, 2], [3, 4], [5, 6, 7]], "n": [1, 0, 2]})
    df = df.select(f(pl.col.lst, pl.element().shift(pl.col.n)))
    assert_frame_equal(df, pl.DataFrame({"lst": [[None, 1], [3, 4], [None, None, 5]]}))

    df = pl.DataFrame({"lst": [[1, 2], [3, 4], None], "n": [1, 0, 2]})
    df = df.select(f(pl.col.lst, pl.element().shift(pl.col.n)))
    assert_frame_equal(df, pl.DataFrame({"lst": [[None, 1], [3, 4], None]}))


def test_list_eval_col_ref_compare_7210() -> None:
    # The canonical use case from #7210: compare list elements to another column.
    df = pl.DataFrame({"a": [[1, 5, 3], [4, 2, 6]], "b": [3, 3]})
    assert_frame_equal(
        df.select(pl.col("a").list.eval(pl.element() > pl.col("b")).alias("gt")),
        pl.DataFrame({"gt": [[False, True, False], [True, False, True]]}),
    )


def test_list_agg_col_ref_count_above_7210() -> None:
    # #7210: count how many list elements exceed a per-row threshold column.
    df = pl.DataFrame({"vals": [[1, 5, 3, 8], [2, 9, 1]], "threshold": [3, 5]})
    assert_frame_equal(
        df.select(
            pl.col("vals")
            .list.agg((pl.element() > pl.col("threshold")).sum())
            .alias("n")
        ),
        pl.DataFrame({"n": [2, 1]}, schema={"n": pl.UInt32}),
    )


def test_list_eval_len1_broadcast_nonempty() -> None:
    # A length-1 list literal is broadcast to the height of the referenced column.
    df = pl.DataFrame({"y": [10, 20, 30]})
    assert_frame_equal(
        df.select(
            pl.lit([1, 2, 3], dtype=pl.List(pl.Int64))
            .list.eval(pl.element() + pl.col("y"))
            .alias("r")
        ),
        pl.DataFrame({"r": [[11, 12, 13], [21, 22, 23], [31, 32, 33]]}),
    )


def test_list_eval_len1_broadcast_nonempty_group_by() -> None:
    df = pl.DataFrame({"g": [1, 1, 2], "y": [10, 20, 30]})
    result = df.group_by("g", maintain_order=True).agg(
        pl.lit([1, 2], dtype=pl.List(pl.Int64))
        .list.eval(pl.element() + pl.col("y"))
        .alias("r")
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"g": [1, 2], "r": [[[11, 12], [21, 22]], [[31, 32]]]},
            schema={"g": pl.Int64, "r": pl.List(pl.List(pl.Int64))},
        ),
    )


def test_list_eval_col_ref_only_no_element() -> None:
    # A scalar column reference without `element` yields a single-element list per row.
    df = pl.DataFrame({"x": [[1, 2], [3, 4, 5]], "y": [10, 20]})
    assert_frame_equal(
        df.select(pl.col("x").list.eval(pl.col("y")).alias("r")),
        pl.DataFrame({"r": [[10], [20]]}),
    )


def test_list_eval_col_ref_scalar_expr() -> None:
    # A scalar expression mixing element, a named column and a literal.
    df = pl.DataFrame({"x": [[1, 2], [3, 4]], "k": [2, 3]})
    assert_frame_equal(
        df.select(pl.col("x").list.eval(pl.element() * pl.col("k") + 100).alias("r")),
        pl.DataFrame({"r": [[102, 104], [109, 112]]}),
    )


def test_list_agg_col_ref_group_by() -> None:
    df = pl.DataFrame(
        {"g": [1, 1, 2], "x": [[1, 2], [3, 4], [5, 6]], "y": [10, 20, 30]}
    )
    result = df.group_by("g", maintain_order=True).agg(
        pl.col("x").list.agg((pl.element() + pl.col("y")).sum()).alias("r")
    )
    assert_frame_equal(
        result,
        pl.DataFrame({"g": [1, 2], "r": [[23, 47], [71]]}),
    )


def test_list_eval_col_ref_append() -> None:
    # A length-changing evaluation (`append`) referencing an outer column. Named columns
    # inside `list.eval` used to raise; they now resolve to the per-row outer value.
    df = pl.DataFrame({"A": ["a", "b"], "B": [["a", "b"], ["c", "d"]]})
    assert_frame_equal(
        df.select(pl.col("B").list.eval(pl.element().append(pl.col("A")))),
        pl.DataFrame({"B": [["a", "b", "a"], ["c", "d", "b"]]}),
    )


def test_list_eval_col_ref_string() -> None:
    df = pl.DataFrame({"x": [["a", "b"], ["c"]], "y": ["_1", "_2"]})
    assert_frame_equal(
        df.select(pl.col("x").list.eval(pl.element() + pl.col("y")).alias("r")),
        pl.DataFrame({"r": [["a_1", "b_1"], ["c_2"]]}),
    )


def test_list_eval_col_ref_projection_pushdown() -> None:
    # A column referenced only inside `eval` must survive projection pushdown; if it
    # were pruned, `collect()` would fail or produce wrong results.
    lf = pl.LazyFrame(
        {"x": [[1, 2, 3], [4, 5]], "y": [10, 20], "unused": [999, 888]}
    ).select(pl.col("x").list.eval(pl.element() + pl.col("y")).alias("r"))
    assert_frame_equal(
        lf.collect(),
        pl.DataFrame({"r": [[11, 12, 13], [24, 25]]}),
    )
    # `y` is referenced only inside the eval; pushdown must keep it.
    assert "y" in lf.explain()


def test_list_eval_col_ref_predicate_and_projection_pushdown() -> None:
    lf = (
        pl.LazyFrame(
            {
                "x": [[1, 2], [3, 4], [5, 6]],
                "y": [10, 20, 30],
                "g": [1, 2, 1],
            }
        )
        .select("g", r=pl.col("x").list.eval(pl.element() + pl.col("y")).list.sum())
        .filter(pl.col("g") == 1)
    )
    assert_frame_equal(
        lf.collect(),
        pl.DataFrame({"g": [1, 1], "r": [23, 71]}),
    )


def test_list_eval_col_ref_derived_column() -> None:
    # The referenced column is produced upstream (not present in the source).
    lf = (
        pl.LazyFrame({"x": [[1, 2], [3, 4]], "y": [10, 20]})
        .with_columns(y2=pl.col("y") * 100)
        .select(pl.col("x").list.eval(pl.element() + pl.col("y2")).alias("r"))
    )
    assert_frame_equal(
        lf.collect(),
        pl.DataFrame({"r": [[1001, 1002], [2003, 2004]]}),
    )


def test_list_eval_nested_col_ref_group_by() -> None:
    # Nested list.eval referencing an outer column, evaluated inside a group_by agg.
    df = pl.DataFrame(
        {
            "g": [1, 1, 2],
            "lst": [[[1, 2], [3]], [[4]], [[5, 6]]],
            "s": [10, 20, 30],
        },
        schema={"g": pl.Int64, "lst": pl.List(pl.List(pl.Int64)), "s": pl.Int64},
    )
    result = df.group_by("g", maintain_order=True).agg(
        pl.col("lst")
        .list.eval(pl.element().list.agg(pl.element().sum() * pl.col("s")))
        .alias("r")
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {"g": [1, 2], "r": [[[30, 30], [80]], [[330]]]},
            schema={"g": pl.Int64, "r": pl.List(pl.List(pl.Int64))},
        ),
    )

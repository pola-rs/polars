from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.exceptions import (
    StructFieldNotFoundError,
)
from polars.testing import assert_frame_equal, assert_series_equal


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
    result = df.select(pl.col("a").list.eval(pl.lit("")))
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
        (pl.element().null_count(), True),
        (pl.element().rank().null_count(), True),
        (pl.element().rank(), False),
        (pl.element() + pl.lit(1), False),
        (pl.element().filter(pl.element() != 0), False),
        (pl.element().drop_nulls(), False),
        (pl.element().n_unique(), True),
    ],
)
def test_list_agg_parametric(expr: pl.Expr, is_scalar: bool) -> None:
    def test_case(s: pl.Series) -> None:
        out = s.list.agg(expr)

        for i, v in enumerate(s):
            if v is None:
                assert out[i] is None
                continue

            assert isinstance(v, pl.Series)

            v = v.rename("")
            v = v.to_frame().select(expr).to_series()

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

import re
from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis.strategies import booleans

import polars as pl
import polars.selectors as cs
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series


def test_top_k() -> None:
    # expression
    s = pl.Series("a", [3, 8, 1, 5, 2])

    assert_series_equal(s.top_k(3), pl.Series("a", [8, 5, 3]), check_order=False)
    assert_series_equal(s.bottom_k(4), pl.Series("a", [3, 2, 1, 5]), check_order=False)

    # 5886
    df = pl.DataFrame(
        {
            "test": [2, 4, 1, 3],
            "val": [2, 4, 9, 3],
            "bool_val": [False, True, True, False],
            "str_value": ["d", "b", "a", "c"],
        }
    )
    assert_frame_equal(
        df.select(pl.col("test").top_k(10)),
        pl.DataFrame({"test": [4, 3, 2, 1]}),
        check_row_order=False,
    )

    assert_frame_equal(
        df.select(
            top_k=pl.col("test").top_k(pl.col("val").min()).sort(),
            bottom_k=pl.col("test").bottom_k(pl.col("val").min()).sort(),
        ),
        pl.DataFrame({"top_k": [3, 4], "bottom_k": [1, 2]}),
    )

    assert_frame_equal(
        df.select(
            pl.col("bool_val").top_k(2).alias("top_k"),
            pl.col("bool_val").bottom_k(2).alias("bottom_k"),
        ),
        pl.DataFrame({"top_k": [True, True], "bottom_k": [False, False]}),
        check_row_order=False,
    )

    assert_frame_equal(
        df.select(pl.col("str_value").top_k(2)),
        pl.DataFrame({"str_value": ["d", "c"]}),
        check_row_order=False,
    )

    assert_frame_equal(
        df.select(pl.col("str_value").bottom_k(2)),
        pl.DataFrame({"str_value": ["a", "b"]}),
        check_row_order=False,
    )

    with pytest.raises(ComputeError):
        df.select(
            pl.col("bool_val").top_k(pl.lit(None)),
        )

    with pytest.raises(ComputeError):
        df.select(pl.col("test").top_k(pl.lit(pl.Series("s", [1, 2]))))

    # dataframe
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 2, 2, None],
            "b": [None, 2, 1, 4, 3, 2, None],
        }
    )

    assert_frame_equal(
        df.top_k(3, by=["a", "b"]),
        pl.DataFrame({"a": [4, 3, 2], "b": [4, 1, 3]}),
        check_row_order=False,
    )

    assert_frame_equal(
        df.top_k(3, by=["a", "b"], reverse=True),
        pl.DataFrame({"a": [1, 2, 2], "b": [None, 2, 2]}),
        check_row_order=False,
    )
    assert_frame_equal(
        df.bottom_k(4, by=["a", "b"], reverse=True),
        pl.DataFrame({"a": [4, 3, 2, 2], "b": [4, 1, 3, 2]}),
        check_row_order=False,
    )

    df2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [12, 11, 10, 9, 8, 7],
            "c": ["Apple", "Orange", "Apple", "Apple", "Banana", "Banana"],
        }
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b").top_k_by("a", 2).name.suffix("_top_by_a").sort(),
            pl.col("a", "b").top_k_by("b", 2).name.suffix("_top_by_b").sort(),
        ),
        pl.DataFrame(
            {
                "a_top_by_a": [5, 6],
                "b_top_by_a": [7, 8],
                "a_top_by_b": [1, 2],
                "b_top_by_b": [11, 12],
            }
        ),
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b")
            .top_k_by("a", 2, reverse=True)
            .name.suffix("_top_by_a")
            .sort(),
            pl.col("a", "b")
            .top_k_by("b", 2, reverse=True)
            .name.suffix("_top_by_b")
            .sort(),
        ),
        pl.DataFrame(
            {
                "a_top_by_a": [1, 2],
                "b_top_by_a": [11, 12],
                "a_top_by_b": [5, 6],
                "b_top_by_b": [7, 8],
            }
        ),
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b").bottom_k_by("a", 2).name.suffix("_bottom_by_a").sort(),
            pl.col("a", "b").bottom_k_by("b", 2).name.suffix("_bottom_by_b").sort(),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_a": [1, 2],
                "b_bottom_by_a": [11, 12],
                "a_bottom_by_b": [5, 6],
                "b_bottom_by_b": [7, 8],
            }
        ),
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b")
            .bottom_k_by("a", 2, reverse=True)
            .name.suffix("_bottom_by_a")
            .sort(),
            pl.col("a", "b")
            .bottom_k_by("b", 2, reverse=True)
            .name.suffix("_bottom_by_b")
            .sort(),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_a": [5, 6],
                "b_bottom_by_a": [7, 8],
                "a_bottom_by_b": [1, 2],
                "b_bottom_by_b": [11, 12],
            }
        ),
    )

    assert_frame_equal(
        df2.group_by("c", maintain_order=True)
        .agg(pl.all().top_k_by("a", 2))
        .explode(cs.all().exclude("c")),
        pl.DataFrame(
            {
                "c": ["Apple", "Apple", "Orange", "Banana", "Banana"],
                "a": [4, 3, 2, 6, 5],
                "b": [9, 10, 11, 7, 8],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.group_by("c", maintain_order=True)
        .agg(pl.all().bottom_k_by("a", 2))
        .explode(cs.all().exclude("c")),
        pl.DataFrame(
            {
                "c": ["Apple", "Apple", "Orange", "Banana", "Banana"],
                "a": [1, 3, 2, 5, 6],
                "b": [12, 10, 11, 8, 7],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c").top_k_by(["c", "a"], 2).name.suffix("_top_by_ca"),
            pl.col("a", "b", "c").top_k_by(["c", "b"], 2).name.suffix("_top_by_cb"),
        ),
        pl.DataFrame(
            {
                "a_top_by_ca": [2, 6],
                "b_top_by_ca": [11, 7],
                "c_top_by_ca": ["Orange", "Banana"],
                "a_top_by_cb": [2, 5],
                "b_top_by_cb": [11, 8],
                "c_top_by_cb": ["Orange", "Banana"],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .bottom_k_by(["c", "a"], 2)
            .name.suffix("_bottom_by_ca")
            .sort(),
            pl.col("a", "b", "c")
            .bottom_k_by(["c", "b"], 2)
            .name.suffix("_bottom_by_cb")
            .sort(),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_ca": [1, 3],
                "b_bottom_by_ca": [10, 12],
                "c_bottom_by_ca": ["Apple", "Apple"],
                "a_bottom_by_cb": [3, 4],
                "b_bottom_by_cb": [9, 10],
                "c_bottom_by_cb": ["Apple", "Apple"],
            }
        ),
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .top_k_by(["c", "a"], 2, reverse=[True, False])
            .name.suffix("_top_by_ca")
            .sort(),
            pl.col("a", "b", "c")
            .top_k_by(["c", "b"], 2, reverse=[True, False])
            .name.suffix("_top_by_cb")
            .sort(),
        ),
        pl.DataFrame(
            {
                "a_top_by_ca": [3, 4],
                "b_top_by_ca": [9, 10],
                "c_top_by_ca": ["Apple", "Apple"],
                "a_top_by_cb": [1, 3],
                "b_top_by_cb": [10, 12],
                "c_top_by_cb": ["Apple", "Apple"],
            }
        ),
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .bottom_k_by(["c", "a"], 2, reverse=[True, False])
            .name.suffix("_bottom_by_ca")
            .sort(),
            pl.col("a", "b", "c")
            .bottom_k_by(["c", "b"], 2, reverse=[True, False])
            .name.suffix("_bottom_by_cb")
            .sort(),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_ca": [2, 5],
                "b_bottom_by_ca": [8, 11],
                "c_bottom_by_ca": ["Banana", "Orange"],
                "a_bottom_by_cb": [2, 6],
                "b_bottom_by_cb": [7, 11],
                "c_bottom_by_cb": ["Banana", "Orange"],
            }
        ),
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .top_k_by(["c", "a"], 2, reverse=[False, True])
            .name.suffix("_top_by_ca")
            .sort(),
            pl.col("a", "b", "c")
            .top_k_by(["c", "b"], 2, reverse=[False, True])
            .name.suffix("_top_by_cb")
            .sort(),
        ),
        pl.DataFrame(
            {
                "a_top_by_ca": [2, 5],
                "b_top_by_ca": [8, 11],
                "c_top_by_ca": ["Banana", "Orange"],
                "a_top_by_cb": [2, 6],
                "b_top_by_cb": [7, 11],
                "c_top_by_cb": ["Banana", "Orange"],
            }
        ),
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .top_k_by(["c", "a"], 2, reverse=[False, True])
            .name.suffix("_bottom_by_ca")
            .sort(),
            pl.col("a", "b", "c")
            .top_k_by(["c", "b"], 2, reverse=[False, True])
            .name.suffix("_bottom_by_cb")
            .sort(),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_ca": [2, 5],
                "b_bottom_by_ca": [8, 11],
                "c_bottom_by_ca": ["Banana", "Orange"],
                "a_bottom_by_cb": [2, 6],
                "b_bottom_by_cb": [7, 11],
                "c_bottom_by_cb": ["Banana", "Orange"],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"the length of `reverse` \(2\) does not match the length of `by` \(1\)",
    ):
        df2.select(pl.all().top_k_by("a", 2, reverse=[True, False]))

    with pytest.raises(
        ValueError,
        match=r"the length of `reverse` \(2\) does not match the length of `by` \(1\)",
    ):
        df2.select(pl.all().bottom_k_by("a", 2, reverse=[True, False]))


def test_top_k_reverse() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.top_k(1, by=["a", "b"], reverse=True)
    expected = pl.DataFrame({"a": [1], "b": [4]})
    assert_frame_equal(result, expected, check_row_order=False)
    result = df.top_k(1, by=["a", "b"], reverse=[True, True])
    assert_frame_equal(result, expected, check_row_order=False)
    with pytest.raises(
        ValueError,
        match=r"the length of `reverse` \(1\) does not match the length of `by` \(2\)",
    ):
        df.top_k(1, by=["a", "b"], reverse=[True])


def test_top_k_9385() -> None:
    lf = pl.LazyFrame({"b": [True, False]})
    result = lf.sort(["b"]).slice(0, 1)
    assert result.collect()["b"].to_list() == [False]


def test_top_k_empty() -> None:
    df = pl.DataFrame({"test": []})

    assert_frame_equal(df.select([pl.col("test").top_k(2)]), df)


@given(s=series(excluded_dtypes=[pl.Null, pl.Struct]), should_sort=booleans())
def test_top_k_nulls(s: pl.Series, should_sort: bool) -> None:
    if should_sort:
        s = s.sort()

    valid_count = s.len() - s.null_count()
    result = s.top_k(valid_count)
    assert result.null_count() == 0

    result = s.top_k(s.len())
    assert result.null_count() == s.null_count()

    result = s.top_k(s.len() * 2)
    assert_series_equal(result, s, check_order=False)


@given(s=series(excluded_dtypes=[pl.Null, pl.Struct]), should_sort=booleans())
def test_bottom_k_nulls(s: pl.Series, should_sort: bool) -> None:
    if should_sort:
        s = s.sort()

    valid_count = s.len() - s.null_count()

    result = s.bottom_k(valid_count)
    assert result.null_count() == 0

    result = s.bottom_k(s.len())
    assert result.null_count() == s.null_count()

    result = s.bottom_k(s.len() * 2)
    assert_series_equal(result, s, check_order=False)


def test_top_k_descending_deprecated() -> None:
    with pytest.deprecated_call():
        pl.col("a").top_k_by("b", descending=True)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("sort_function"),
    [
        lambda x: x,
        lambda x: x.sort("a", descending=False, maintain_order=True),
        lambda x: x.sort("a", descending=True, maintain_order=True),
    ],
)
@pytest.mark.parametrize(
    ("df", "df2"),
    [
        (
            pl.LazyFrame({"a": [3, 4, 1, 2, 5]}),
            pl.LazyFrame({"a": [1, None, None, 4, 5]}),
        ),
        (
            pl.LazyFrame({"a": [3, 4, 1, 2, 5], "b": [1, 2, 3, 4, 5]}),
            pl.LazyFrame({"a": [1, None, None, 4, 5], "b": [1, 2, 3, 4, 5]}),
        ),
    ],
)
def test_top_k_df(
    sort_function: Callable[[pl.LazyFrame], pl.LazyFrame],
    df: pl.LazyFrame,
    df2: pl.LazyFrame,
) -> None:
    df = sort_function(df)
    expected = [5, 4, 3]
    assert df.sort("a", descending=True).limit(3).collect()["a"].to_list() == expected
    assert df.top_k(3, by="a").collect()["a"].to_list() == expected
    expected = [1, 2, 3]
    assert df.sort("a", descending=False).limit(3).collect()["a"].to_list() == expected
    assert df.bottom_k(3, by="a").collect()["a"].to_list() == expected

    df = sort_function(df2)
    expected2 = [5, 4, 1, None]
    assert (
        df.sort("a", descending=True, nulls_last=True).limit(4).collect()["a"].to_list()
        == expected2
    )
    assert df.top_k(4, by="a").collect()["a"].to_list() == expected2
    expected2 = [1, 4, 5, None]
    assert (
        df.sort("a", descending=False, nulls_last=True)
        .limit(4)
        .collect()["a"]
        .to_list()
        == expected2
    )
    assert df.bottom_k(4, by="a").collect()["a"].to_list() == expected2

    assert df.sort("a", descending=False, nulls_last=False).limit(4).collect()[
        "a"
    ].to_list() == [None, None, 1, 4]
    assert df.sort("a", descending=True, nulls_last=False).limit(4).collect()[
        "a"
    ].to_list() == [None, None, 5, 4]


@pytest.mark.parametrize("descending", [True, False])
def test_sorted_top_k_20719(descending: bool) -> None:
    df = pl.DataFrame(
        [
            {"a": 1, "b": 1},
            {"a": 5, "b": 5},
            {"a": 9, "b": 9},
            {"a": 10, "b": 20},
        ]
    ).sort(by="a", descending=descending)

    # Note: Output stability is guaranteed by the input sortedness as an
    # implementation detail.

    for func, reverse in [
        [pl.DataFrame.top_k, False],
        [pl.DataFrame.bottom_k, True],
    ]:
        assert_frame_equal(
            df.pipe(func, 2, by="a", reverse=reverse),  # type: ignore[arg-type]
            pl.DataFrame(
                [
                    {"a": 10, "b": 20},
                    {"a": 9, "b": 9},
                ]
            ),
        )

    for func, reverse in [
        [pl.DataFrame.top_k, True],
        [pl.DataFrame.bottom_k, False],
    ]:
        assert_frame_equal(
            df.pipe(func, 2, by="a", reverse=reverse),  # type: ignore[arg-type]
            pl.DataFrame(
                [
                    {"a": 1, "b": 1},
                    {"a": 5, "b": 5},
                ]
            ),
        )


@pytest.mark.parametrize(
    ("func", "reverse", "expect"),
    [
        (pl.DataFrame.top_k, False, pl.DataFrame({"a": [2, 2]})),
        (pl.DataFrame.bottom_k, True, pl.DataFrame({"a": [2, 2]})),
        (pl.DataFrame.top_k, True, pl.DataFrame({"a": [1, 2]})),
        (pl.DataFrame.bottom_k, False, pl.DataFrame({"a": [1, 2]})),
    ],
)
@pytest.mark.parametrize("descending", [True, False])
def test_sorted_top_k_duplicates(
    func: Callable[[pl.DataFrame], pl.DataFrame],
    reverse: bool,
    expect: pl.DataFrame,
    descending: bool,
) -> None:
    assert_frame_equal(
        pl.DataFrame({"a": [1, 2, 2]})  # type: ignore[call-arg]
        .sort("a", descending=descending)
        .pipe(func, 2, by="a", reverse=reverse),
        expect,
    )


def test_top_k_list_dtype() -> None:
    s = pl.Series([[1, 2], [3, 4], [], [0]], dtype=pl.List(pl.Int64))
    expected = pl.Series([[1, 2], [3, 4]], dtype=pl.List(pl.Int64))
    assert_series_equal(s.top_k(2), expected, check_order=False)

    s = pl.Series([[[1, 2], [3]], [[4], []], [[0]]], dtype=pl.List(pl.List(pl.Int64)))
    expected = pl.Series([[[4], []], [[1, 2], [3]]], dtype=pl.List(pl.List(pl.Int64)))
    assert_series_equal(s.top_k(2), expected, check_order=False)


def test_top_k_sorted_21260() -> None:
    s = pl.Series([1, 2, 3, 4, 5])
    assert s.top_k(3).sort().to_list() == [3, 4, 5]
    assert s.sort(descending=False).top_k(3).sort().to_list() == [3, 4, 5]
    assert s.sort(descending=True).top_k(3).sort().to_list() == [3, 4, 5]

    assert s.bottom_k(3).sort().to_list() == [1, 2, 3]
    assert s.sort(descending=False).bottom_k(3).sort().to_list() == [1, 2, 3]
    assert s.sort(descending=True).bottom_k(3).sort().to_list() == [1, 2, 3]


def test_top_k_by() -> None:
    # expression
    s = pl.Series("a", [3, 8, 1, 5, 2])

    assert_series_equal(
        s.top_k_by("a", 3), pl.Series("a", [8, 5, 3]), check_order=False
    )


def test_bottom_k_by() -> None:
    # expression
    s = pl.Series("a", [3, 8, 1, 5, 2])

    assert_series_equal(
        s.bottom_k_by("a", 4), pl.Series("a", [3, 2, 1, 5]), check_order=False
    )


def test_sort_head_maintain_order() -> None:
    df = pl.DataFrame(
        {"x": [2, 0, 8, 0, 0, 0, 7, 0, 9, 0], "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    )
    expected = pl.DataFrame({"x": [0, 0, 0, 0], "y": [1, 3, 4, 5]})
    q = df.lazy().sort(by="x", maintain_order=True).head(4)
    assert_frame_equal(q.collect(), expected)


def test_top_k_non_elementwise_by_24163() -> None:
    query = pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8]}).top_k(
        2, by=(pl.when(pl.len() == 8).then(pl.col.a).otherwise(-pl.col.a))
    )

    expected = pl.DataFrame({"a": [7, 8]})
    assert_frame_equal(expected, query.collect(), check_row_order=False)


def test_top_k_by_non_uniq_name_25072() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.sort(by=[pl.col.a, pl.col.a]).head(2)
    expected = pl.DataFrame({"a": [1, 2], "b": [4, 5]})
    assert_frame_equal(result, expected)


def test_top_k_union_null() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.Int64})
    df2 = pl.DataFrame({"a": [None, None, None]}, schema={"a": pl.Null})
    out = (
        pl.concat([df1.lazy().join(df1.lazy(), on="a"), df2.lazy()])
        .bottom_k(5, by="a")
        .collect(engine="streaming")
    )
    assert_frame_equal(
        out,
        pl.DataFrame({"a": [1, 2, 3, None, None]}, schema={"a": pl.Int64}),
        check_row_order=False,
    )


def test_top_k_dyn_pred_pushdown() -> None:
    df = pl.DataFrame({"x": [1], "y": [1]})
    plan = df.lazy().with_columns(pl.col.x * pl.col.x).sort("y").head(3).explain()

    pred = re.search(r"FILTER.*dynamic_predicate", plan)
    with_cols = re.search(r"WITH_COLUMNS", plan)
    assert pred is not None
    assert with_cols is not None
    assert pred.start() > with_cols.start()

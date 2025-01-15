from typing import Callable

import pytest
from hypothesis import given
from hypothesis.strategies import booleans

import polars as pl
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
            top_k=pl.col("test").top_k(pl.col("val").min()),
            bottom_k=pl.col("test").bottom_k(pl.col("val").min()),
        ),
        pl.DataFrame({"top_k": [4, 3], "bottom_k": [1, 2]}),
        check_row_order=False,
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
        df.select(
            pl.col("str_value").top_k(2).alias("top_k"),
            pl.col("str_value").bottom_k(2).alias("bottom_k"),
        ),
        pl.DataFrame({"top_k": ["d", "c"], "bottom_k": ["a", "b"]}),
        check_row_order=False,
    )

    with pytest.raises(ComputeError, match="`k` must be set for `top_k`"):
        df.select(
            pl.col("bool_val").top_k(pl.lit(None)),
        )

    with pytest.raises(ComputeError, match="`k` must be a single value for `top_k`."):
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
            pl.col("a", "b").top_k_by("a", 2).name.suffix("_top_by_a"),
            pl.col("a", "b").top_k_by("b", 2).name.suffix("_top_by_b"),
        ),
        pl.DataFrame(
            {
                "a_top_by_a": [6, 5],
                "b_top_by_a": [7, 8],
                "a_top_by_b": [1, 2],
                "b_top_by_b": [12, 11],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b").top_k_by("a", 2, reverse=True).name.suffix("_top_by_a"),
            pl.col("a", "b").top_k_by("b", 2, reverse=True).name.suffix("_top_by_b"),
        ),
        pl.DataFrame(
            {
                "a_top_by_a": [1, 2],
                "b_top_by_a": [12, 11],
                "a_top_by_b": [6, 5],
                "b_top_by_b": [7, 8],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b").bottom_k_by("a", 2).name.suffix("_bottom_by_a"),
            pl.col("a", "b").bottom_k_by("b", 2).name.suffix("_bottom_by_b"),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_a": [1, 2],
                "b_bottom_by_a": [12, 11],
                "a_bottom_by_b": [6, 5],
                "b_bottom_by_b": [7, 8],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b")
            .bottom_k_by("a", 2, reverse=True)
            .name.suffix("_bottom_by_a"),
            pl.col("a", "b")
            .bottom_k_by("b", 2, reverse=True)
            .name.suffix("_bottom_by_b"),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_a": [6, 5],
                "b_bottom_by_a": [7, 8],
                "a_bottom_by_b": [1, 2],
                "b_bottom_by_b": [12, 11],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.group_by("c", maintain_order=True)
        .agg(pl.all().top_k_by("a", 2))
        .explode(pl.all().exclude("c")),
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
        .explode(pl.all().exclude("c")),
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
            .name.suffix("_bottom_by_ca"),
            pl.col("a", "b", "c")
            .bottom_k_by(["c", "b"], 2)
            .name.suffix("_bottom_by_cb"),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_ca": [1, 3],
                "b_bottom_by_ca": [12, 10],
                "c_bottom_by_ca": ["Apple", "Apple"],
                "a_bottom_by_cb": [4, 3],
                "b_bottom_by_cb": [9, 10],
                "c_bottom_by_cb": ["Apple", "Apple"],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .top_k_by(["c", "a"], 2, reverse=[True, False])
            .name.suffix("_top_by_ca"),
            pl.col("a", "b", "c")
            .top_k_by(["c", "b"], 2, reverse=[True, False])
            .name.suffix("_top_by_cb"),
        ),
        pl.DataFrame(
            {
                "a_top_by_ca": [4, 3],
                "b_top_by_ca": [9, 10],
                "c_top_by_ca": ["Apple", "Apple"],
                "a_top_by_cb": [1, 3],
                "b_top_by_cb": [12, 10],
                "c_top_by_cb": ["Apple", "Apple"],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .bottom_k_by(["c", "a"], 2, reverse=[True, False])
            .name.suffix("_bottom_by_ca"),
            pl.col("a", "b", "c")
            .bottom_k_by(["c", "b"], 2, reverse=[True, False])
            .name.suffix("_bottom_by_cb"),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_ca": [2, 5],
                "b_bottom_by_ca": [11, 8],
                "c_bottom_by_ca": ["Orange", "Banana"],
                "a_bottom_by_cb": [2, 6],
                "b_bottom_by_cb": [11, 7],
                "c_bottom_by_cb": ["Orange", "Banana"],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .top_k_by(["c", "a"], 2, reverse=[False, True])
            .name.suffix("_top_by_ca"),
            pl.col("a", "b", "c")
            .top_k_by(["c", "b"], 2, reverse=[False, True])
            .name.suffix("_top_by_cb"),
        ),
        pl.DataFrame(
            {
                "a_top_by_ca": [2, 5],
                "b_top_by_ca": [11, 8],
                "c_top_by_ca": ["Orange", "Banana"],
                "a_top_by_cb": [2, 6],
                "b_top_by_cb": [11, 7],
                "c_top_by_cb": ["Orange", "Banana"],
            }
        ),
        check_row_order=False,
    )

    assert_frame_equal(
        df2.select(
            pl.col("a", "b", "c")
            .top_k_by(["c", "a"], 2, reverse=[False, True])
            .name.suffix("_bottom_by_ca"),
            pl.col("a", "b", "c")
            .top_k_by(["c", "b"], 2, reverse=[False, True])
            .name.suffix("_bottom_by_cb"),
        ),
        pl.DataFrame(
            {
                "a_bottom_by_ca": [2, 5],
                "b_bottom_by_ca": [11, 8],
                "c_bottom_by_ca": ["Orange", "Banana"],
                "a_bottom_by_cb": [2, 6],
                "b_bottom_by_cb": [11, 7],
                "c_bottom_by_cb": ["Orange", "Banana"],
            }
        ),
        check_row_order=False,
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

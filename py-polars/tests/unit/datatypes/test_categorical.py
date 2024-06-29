from __future__ import annotations

import contextlib
import io
import operator
from typing import TYPE_CHECKING, Any, Callable, Literal

import pytest

import polars as pl
from polars import StringCache
from polars.exceptions import (
    CategoricalRemappingWarning,
    ComputeError,
    StringCacheMismatchError,
)
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@StringCache()
def test_categorical_full_outer_join() -> None:
    df1 = pl.DataFrame(
        [
            pl.Series("key1", [42]),
            pl.Series("key2", ["bar"], dtype=pl.Categorical),
            pl.Series("val1", [1]),
        ]
    ).lazy()

    df2 = pl.DataFrame(
        [
            pl.Series("key1", [42]),
            pl.Series("key2", ["bar"], dtype=pl.Categorical),
            pl.Series("val2", [2]),
        ]
    ).lazy()

    expected = pl.DataFrame(
        {
            "key1": [42],
            "key2": ["bar"],
            "val1": [1],
            "key1_right": [42],
            "key2_right": ["bar"],
            "val2": [2],
        },
        schema_overrides={"key2": pl.Categorical, "key2_right": pl.Categorical},
    )

    out = df1.join(df2, on=["key1", "key2"], how="full").collect()
    assert_frame_equal(out, expected)

    dfa = pl.DataFrame(
        [
            pl.Series("key", ["foo", "bar"], dtype=pl.Categorical),
            pl.Series("val1", [3, 1]),
        ]
    )
    dfb = pl.DataFrame(
        [
            pl.Series("key", ["bar", "baz"], dtype=pl.Categorical),
            pl.Series("val2", [6, 8]),
        ]
    )

    df = dfa.join(dfb, on="key", how="full")
    # the cast is important to test the rev map
    assert df["key"].cast(pl.String).to_list() == ["bar", None, "foo"]
    assert df["key_right"].cast(pl.String).to_list() == ["bar", "baz", None]


def test_read_csv_categorical() -> None:
    f = io.BytesIO()
    f.write(b"col1,col2,col3,col4,col5,col6\n'foo',2,3,4,5,6\n'bar',8,9,10,11,12")
    f.seek(0)
    df = pl.read_csv(f, has_header=True, schema_overrides={"col1": pl.Categorical})
    assert df["col1"].dtype == pl.Categorical


def test_cat_to_dummies() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3, 4], "bar": ["a", "b", "a", "c"]})
    df = df.with_columns(pl.col("bar").cast(pl.Categorical))
    assert df.to_dummies().to_dict(as_series=False) == {
        "foo_1": [1, 0, 0, 0],
        "foo_2": [0, 1, 0, 0],
        "foo_3": [0, 0, 1, 0],
        "foo_4": [0, 0, 0, 1],
        "bar_a": [1, 0, 1, 0],
        "bar_b": [0, 1, 0, 0],
        "bar_c": [0, 0, 0, 1],
    }


@StringCache()
def test_categorical_is_in_list() -> None:
    # this requires type coercion to cast.
    # we should not cast within the function as this would be expensive within a
    # group by context that would be a cast per group
    df = pl.DataFrame(
        {"a": [1, 2, 3, 1, 2], "b": ["a", "b", "c", "d", "e"]}
    ).with_columns(pl.col("b").cast(pl.Categorical))

    cat_list = ("a", "b", "c")
    assert df.filter(pl.col("b").is_in(cat_list)).to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
    }


@StringCache()
def test_unset_sorted_on_append() -> None:
    df1 = pl.DataFrame(
        [
            pl.Series("key", ["a", "b", "a", "b"], dtype=pl.Categorical),
            pl.Series("val", [1, 2, 3, 4]),
        ]
    ).sort("key")
    df2 = pl.DataFrame(
        [
            pl.Series("key", ["a", "b", "a", "b"], dtype=pl.Categorical),
            pl.Series("val", [5, 6, 7, 8]),
        ]
    ).sort("key")
    df = pl.concat([df1, df2], rechunk=False)
    assert df.group_by("key").len()["len"].to_list() == [4, 4]


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.eq, pl.Series([True, True, True, False, None, None])),
        (operator.ne, pl.Series([False, False, False, True, None, None])),
        (pl.Series.ne_missing, pl.Series([False, False, False, True, True, True])),
        (pl.Series.eq_missing, pl.Series([True, True, True, False, False, False])),
    ],
)
def test_categorical_equality(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series(["a", "b", "c", "c", None, None], dtype=pl.Categorical)
    s2 = pl.Series("b_cat", ["a", "b", "c", "a", "b", "c"], dtype=pl.Categorical)
    assert_series_equal(op(s, s2), expected)
    assert_series_equal(op(s, s2.cast(pl.String)), expected)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.eq, pl.Series([False, False, False, False, None, None])),
        (operator.ne, pl.Series([True, True, True, True, None, None])),
        (pl.Series.eq_missing, pl.Series([False, False, False, False, False, False])),
        (pl.Series.ne_missing, pl.Series([True, True, True, True, True, True])),
    ],
)
@StringCache()
def test_categorical_equality_global_fastpath(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series(["a", "b", "c", "c", None, None], dtype=pl.Categorical)
    s2 = pl.Series(["d"], dtype=pl.Categorical)
    assert_series_equal(op(s, s2), expected)
    assert_series_equal(op(s, s2.cast(pl.String)), expected)


@pytest.mark.parametrize(
    ("op", "expected_phys", "expected_lexical"),
    [
        (
            operator.le,
            pl.Series([True, True, True, True, False]),
            pl.Series([False, True, True, False, True]),
        ),
        (
            operator.lt,
            pl.Series([True, False, False, True, False]),
            pl.Series([False, False, False, False, True]),
        ),
        (
            operator.ge,
            pl.Series([False, True, True, False, True]),
            pl.Series([True, True, True, True, False]),
        ),
        (
            operator.gt,
            pl.Series([False, False, False, False, True]),
            pl.Series([True, False, False, True, False]),
        ),
    ],
)
@StringCache()
def test_categorical_global_ordering(
    op: Callable[[pl.Series, pl.Series], pl.Series],
    expected_phys: pl.Series,
    expected_lexical: pl.Series,
) -> None:
    s = pl.Series(["z", "b", "c", "c", "a"], dtype=pl.Categorical)
    s2 = pl.Series("b_cat", ["a", "b", "c", "a", "c"], dtype=pl.Categorical)
    assert_series_equal(op(s, s2), expected_phys)

    s = s.cast(pl.Categorical("lexical"))
    s2 = s2.cast(pl.Categorical("lexical"))
    assert_series_equal(op(s, s2), expected_lexical)


@pytest.mark.parametrize(
    ("op", "expected_phys", "expected_lexical"),
    [
        (operator.le, pl.Series([True, True, False]), pl.Series([False, True, False])),
        (
            operator.lt,
            pl.Series([True, False, False]),
            pl.Series([False, False, False]),
        ),
        (operator.ge, pl.Series([False, True, True]), pl.Series([True, True, True])),
        (operator.gt, pl.Series([False, False, True]), pl.Series([True, False, True])),
    ],
)
@StringCache()
def test_categorical_global_ordering_broadcast_rhs(
    op: Callable[[pl.Series, pl.Series], pl.Series],
    expected_phys: pl.Series,
    expected_lexical: pl.Series,
) -> None:
    s = pl.Series(["c", "a", "b"], dtype=pl.Categorical)
    s2 = pl.Series("b_cat", ["a"], dtype=pl.Categorical)
    assert_series_equal(op(s, s2), expected_phys)

    s = s.cast(pl.Categorical("lexical"))
    s2 = s2.cast(pl.Categorical("lexical"))
    assert_series_equal(op(s, s2), expected_lexical)
    assert_series_equal(op(s, s2.cast(pl.String)), expected_lexical)


@pytest.mark.parametrize(
    ("op", "expected_phys", "expected_lexical"),
    [
        (operator.le, pl.Series([True, True, True]), pl.Series([True, False, True])),
        (operator.lt, pl.Series([True, True, False]), pl.Series([True, False, False])),
        (operator.ge, pl.Series([False, False, True]), pl.Series([False, True, True])),
        (
            operator.gt,
            pl.Series([False, False, False]),
            pl.Series([False, True, False]),
        ),
    ],
)
@StringCache()
def test_categorical_global_ordering_broadcast_lhs(
    op: Callable[[pl.Series, pl.Series], pl.Series],
    expected_phys: pl.Series,
    expected_lexical: pl.Series,
) -> None:
    s = pl.Series(["b"], dtype=pl.Categorical)
    s2 = pl.Series(["c", "a", "b"], dtype=pl.Categorical)
    assert_series_equal(op(s, s2), expected_phys)

    s = s.cast(pl.Categorical("lexical"))
    s2 = s2.cast(pl.Categorical("lexical"))
    assert_series_equal(op(s, s2), expected_lexical)
    assert_series_equal(op(s, s2.cast(pl.String)), expected_lexical)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.le, pl.Series([True, True, True, False, True, True])),
        (operator.lt, pl.Series([False, False, False, False, True, False])),
        (operator.ge, pl.Series([True, True, True, True, False, True])),
        (operator.gt, pl.Series([False, False, False, True, False, False])),
    ],
)
def test_categorical_ordering(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series(["a", "b", "c", "c", "a", "b"], dtype=pl.Categorical)
    s2 = pl.Series("b_cat", ["a", "b", "c", "a", "c", "b"], dtype=pl.Categorical)
    assert_series_equal(op(s, s2), expected)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.le, pl.Series([None, True, True, True, True, True])),
        (operator.lt, pl.Series([None, False, False, False, True, True])),
        (operator.ge, pl.Series([None, True, True, True, False, False])),
        (operator.gt, pl.Series([None, False, False, False, False, False])),
    ],
)
def test_compare_categorical(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series([None, "a", "b", "c", "b", "a"], dtype=pl.Categorical)
    s2 = pl.Series([None, "a", "b", "c", "c", "b"])

    assert_series_equal(op(s, s2), expected)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.le, pl.Series([None, True, True, False, True, True])),
        (operator.lt, pl.Series([None, True, False, False, False, True])),
        (operator.ge, pl.Series([None, False, True, True, True, False])),
        (operator.gt, pl.Series([None, False, False, True, False, False])),
        (operator.eq, pl.Series([None, False, True, False, True, False])),
        (operator.ne, pl.Series([None, True, False, True, False, True])),
        (pl.Series.eq_missing, pl.Series([False, False, True, False, True, False])),
        (pl.Series.ne_missing, pl.Series([True, True, False, True, False, True])),
    ],
)
def test_compare_categorical_single(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series([None, "a", "b", "c", "b", "a"], dtype=pl.Categorical)
    s2 = "b"

    assert_series_equal(op(s, s2), expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.le, pl.Series([None, True, True, True, True, True])),
        (operator.lt, pl.Series([None, True, True, True, True, True])),
        (operator.ge, pl.Series([None, False, False, False, False, False])),
        (operator.gt, pl.Series([None, False, False, False, False, False])),
        (operator.eq, pl.Series([None, False, False, False, False, False])),
        (operator.ne, pl.Series([None, True, True, True, True, True])),
        (pl.Series.ne_missing, pl.Series([True, True, True, True, True, True])),
        (pl.Series.eq_missing, pl.Series([False, False, False, False, False, False])),
    ],
)
@StringCache()
def test_compare_categorical_single_non_existent(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series([None, "a", "b", "c", "b", "a"], dtype=pl.Categorical)
    s2 = "d"
    assert_series_equal(op(s, s2), expected)  # type: ignore[arg-type]
    s_cat = pl.Series(["d"], dtype=pl.Categorical)
    assert_series_equal(op(s, s_cat), expected)
    assert_series_equal(op(s, s_cat.cast(pl.String)), expected)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (
            operator.le,
            pl.Series([None, None, None, None, None, None], dtype=pl.Boolean),
        ),
        (
            operator.lt,
            pl.Series([None, None, None, None, None, None], dtype=pl.Boolean),
        ),
        (
            operator.ge,
            pl.Series([None, None, None, None, None, None], dtype=pl.Boolean),
        ),
        (
            operator.gt,
            pl.Series([None, None, None, None, None, None], dtype=pl.Boolean),
        ),
        (
            operator.eq,
            pl.Series([None, None, None, None, None, None], dtype=pl.Boolean),
        ),
        (
            operator.ne,
            pl.Series([None, None, None, None, None, None], dtype=pl.Boolean),
        ),
        (pl.Series.ne_missing, pl.Series([False, True, True, True, True, True])),
        (pl.Series.eq_missing, pl.Series([True, False, False, False, False, False])),
    ],
)
@StringCache()
def test_compare_categorical_single_none(
    op: Callable[[pl.Series, pl.Series], pl.Series], expected: pl.Series
) -> None:
    s = pl.Series([None, "a", "b", "c", "b", "a"], dtype=pl.Categorical)
    s2 = pl.Series([None], dtype=pl.Categorical)
    assert_series_equal(op(s, s2), expected)
    assert_series_equal(op(s, s2.cast(pl.String)), expected)


def test_categorical_error_on_local_cmp() -> None:
    df_cat = pl.DataFrame(
        [
            pl.Series("a_cat", ["c", "a", "b", "c", "b"], dtype=pl.Categorical),
            pl.Series("b_cat", ["F", "G", "E", "G", "G"], dtype=pl.Categorical),
        ]
    )
    with pytest.raises(
        StringCacheMismatchError,
        match="cannot compare categoricals coming from different sources",
    ):
        df_cat.filter(pl.col("a_cat") == pl.col("b_cat"))


def test_cast_null_to_categorical() -> None:
    assert pl.DataFrame().with_columns(
        pl.lit(None).cast(pl.Categorical).alias("nullable_enum")
    ).dtypes == [pl.Categorical]


@StringCache()
def test_merge_lit_under_global_cache_4491() -> None:
    df = pl.DataFrame(
        [
            pl.Series("label", ["foo", "bar"], dtype=pl.Categorical),
            pl.Series("value", [3, 9]),
        ]
    )
    assert df.with_columns(
        pl.when(pl.col("value") > 5)
        .then(pl.col("label"))
        .otherwise(pl.lit(None, pl.Categorical))
    ).to_dict(as_series=False) == {"label": [None, "bar"], "value": [3, 9]}


def test_nested_cache_composition() -> None:
    # very artificial example/test, but validates the behaviour
    # of nested StringCache scopes, which we want to play well
    # with each other when composing more complex pipelines.

    assert pl.using_string_cache() is False

    # function representing a composable stage of a pipeline; it implements
    # an inner scope for the case where it is called by itself, but when
    # called as part of a larger series of ops it should not invalidate
    # the string cache (eg: the outermost scope should be respected).
    def create_lazy(data: dict) -> pl.LazyFrame:  # type: ignore[type-arg]
        with pl.StringCache():
            df = pl.DataFrame({"a": ["foo", "bar", "ham"], "b": [1, 2, 3]})
            lf = df.with_columns(pl.col("a").cast(pl.Categorical)).lazy()

        # confirm that scope-exit does NOT invalidate the
        # cache yet, as an outer context is still active
        assert pl.using_string_cache() is True
        return lf

    # this outer scope should be respected
    with pl.StringCache():
        lf1 = create_lazy({"a": ["foo", "bar", "ham"], "b": [1, 2, 3]})
        lf2 = create_lazy({"a": ["spam", "foo", "eggs"], "c": [3, 2, 2]})

        res = lf1.join(lf2, on="a", how="inner").collect().rows()
        assert sorted(res) == [("bar", 2, 2), ("foo", 1, 1), ("ham", 3, 3)]

    # no other scope active; NOW we expect the cache to have been invalidated
    assert pl.using_string_cache() is False


def test_categorical_in_struct_nulls() -> None:
    s = pl.Series(
        "job", ["doctor", "waiter", None, None, None, "doctor"], pl.Categorical
    )
    df = pl.DataFrame([s])
    s = (df.select(pl.col("job").value_counts(sort=True)))["job"]

    assert s[0] == {"job": None, "count": 3}
    assert s[1] == {"job": "doctor", "count": 2}
    assert s[2] == {"job": "waiter", "count": 1}


def test_cast_inner_categorical() -> None:
    dtype = pl.List(pl.Categorical)
    out = pl.Series("foo", [["a"], ["a", "b"]]).cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == [["a"], ["a", "b"]]

    with pytest.raises(
        ComputeError, match=r"casting to categorical not allowed in `list.eval`"
    ):
        pl.Series("foo", [["a", "b"], ["a", "b"]]).list.eval(
            pl.element().cast(pl.Categorical)
        )


@pytest.mark.slow()
def test_stringcache() -> None:
    N = 1_500
    with pl.StringCache():
        # create a large enough column that the categorical map is reallocated
        df = pl.DataFrame({"cats": pl.arange(0, N, eager=True)}).select(
            pl.col("cats").cast(pl.String).cast(pl.Categorical)
        )
        assert df.filter(pl.col("cats").is_in(["1", "2"])).to_dict(as_series=False) == {
            "cats": ["1", "2"]
        }


@pytest.mark.parametrize(
    ("dtype", "outcome"),
    [
        (pl.Categorical, ["foo", "bar", "baz"]),
        (pl.Categorical("physical"), ["foo", "bar", "baz"]),
        (pl.Categorical("lexical"), ["bar", "baz", "foo"]),
    ],
)
def test_categorical_sort_order_by_parameter(
    dtype: PolarsDataType, outcome: list[str]
) -> None:
    s = pl.Series(["foo", "bar", "baz"], dtype=dtype)
    df = pl.DataFrame({"cat": s})
    assert df.sort(["cat"])["cat"].to_list() == outcome


@StringCache()
@pytest.mark.parametrize("row_fmt_sort_enabled", [False, True])
def test_categorical_sort_order(row_fmt_sort_enabled: bool, monkeypatch: Any) -> None:
    # create the categorical ordering first
    pl.Series(["foo", "bar", "baz"], dtype=pl.Categorical)
    df = pl.DataFrame(
        {
            "n": [0, 0, 0],
            # use same categories in different order
            "x": pl.Series(["baz", "bar", "foo"], dtype=pl.Categorical),
        }
    )

    if row_fmt_sort_enabled:
        monkeypatch.setenv("POLARS_ROW_FMT_SORT", "1")

    result = df.sort(["n", "x"])
    assert result["x"].to_list() == ["foo", "bar", "baz"]

    result = df.with_columns(pl.col("x").cast(pl.Categorical("lexical"))).sort("n", "x")
    assert result["x"].to_list() == ["bar", "baz", "foo"]


def test_err_on_categorical_asof_join_by_arg() -> None:
    df1 = pl.DataFrame(
        [
            pl.Series("cat", ["a", "foo", "bar", "foo", "bar"], dtype=pl.Categorical),
            pl.Series("time", [-10, 0, 10, 20, 30], dtype=pl.Int32),
        ]
    )
    df2 = pl.DataFrame(
        [
            pl.Series(
                "cat",
                ["bar", "bar", "bar", "bar", "foo", "foo", "foo", "foo"],
                dtype=pl.Categorical,
            ),
            pl.Series("time", [-5, 5, 15, 25] * 2, dtype=pl.Int32),
            pl.Series("x", [1, 2, 3, 4] * 2, dtype=pl.Int32),
        ]
    )
    with pytest.raises(
        StringCacheMismatchError,
        match="cannot compare categoricals coming from different sources",
    ):
        df1.join_asof(df2, on=pl.col("time").set_sorted(), by="cat")


def test_categorical_list_get_item() -> None:
    out = pl.Series([["a"]]).cast(pl.List(pl.Categorical)).item()
    assert isinstance(out, pl.Series)
    assert out.dtype == pl.Categorical


def test_nested_categorical_aggregation_7848() -> None:
    # a double categorical aggregation
    assert pl.DataFrame(
        {
            "group": [1, 1, 2, 2, 2, 3, 3],
            "letter": ["a", "b", "c", "d", "e", "f", "g"],
        }
    ).with_columns([pl.col("letter").cast(pl.Categorical)]).group_by(
        "group", maintain_order=True
    ).all().with_columns(pl.col("letter").list.len().alias("c_group")).group_by(
        ["c_group"], maintain_order=True
    ).agg(pl.col("letter")).to_dict(as_series=False) == {
        "c_group": [2, 3],
        "letter": [[["a", "b"], ["f", "g"]], [["c", "d", "e"]]],
    }


def test_nested_categorical_cast() -> None:
    values = [["x"], ["y"], ["x"]]
    dtype = pl.List(pl.Categorical)
    s = pl.Series(values).cast(dtype)
    assert s.dtype == dtype
    assert s.to_list() == values


def test_struct_categorical_nesting() -> None:
    # this triggers a lot of materialization
    df = pl.DataFrame(
        {"cats": ["Value1", "Value2", "Value1"]},
        schema_overrides={"cats": pl.Categorical},
    )
    s = df.select(pl.struct(pl.col("cats")))["cats"].implode()
    assert s.dtype == pl.List(pl.Struct([pl.Field("cats", pl.Categorical)]))
    # triggers recursive conversion
    assert s.to_list() == [[{"cats": "Value1"}, {"cats": "Value2"}, {"cats": "Value1"}]]
    # triggers different recursive conversion
    assert len(s.to_arrow()) == 1


def test_categorical_fill_null_existing_category() -> None:
    # ensure physical types align
    df = pl.DataFrame({"col": ["a", None, "a"]}, schema={"col": pl.Categorical})
    result = df.fill_null("a").with_columns(pl.col("col").to_physical().alias("code"))
    expected = {"col": ["a", "a", "a"], "code": [0, 0, 0]}
    assert result.to_dict(as_series=False) == expected


@StringCache()
def test_categorical_fill_null_stringcache() -> None:
    df = pl.LazyFrame(
        {"index": [1, 2, 3], "cat": ["a", "b", None]},
        schema={"index": pl.Int64(), "cat": pl.Categorical()},
    )
    a = df.select(pl.col("cat").fill_null("hi")).collect()

    assert a.to_dict(as_series=False) == {"cat": ["a", "b", "hi"]}
    assert a.dtypes == [pl.Categorical]


def test_fast_unique_flag_from_arrow() -> None:
    df = pl.DataFrame(
        {
            "colB": ["1", "2", "3", "4", "5", "5", "5", "5"],
        }
    ).with_columns([pl.col("colB").cast(pl.Categorical)])

    filtered = df.to_arrow().filter([True, False, True, True, False, True, True, True])
    assert pl.from_arrow(filtered).select(pl.col("colB").n_unique()).item() == 4  # type: ignore[union-attr]


def test_construct_with_null() -> None:
    # Example from https://github.com/pola-rs/polars/issues/7188
    df = pl.from_dicts([{"A": None}, {"A": "foo"}], schema={"A": pl.Categorical})
    assert df.to_series().to_list() == [None, "foo"]

    s = pl.Series([{"struct_A": None}], dtype=pl.Struct({"struct_A": pl.Categorical}))
    assert s.to_list() == [{"struct_A": None}]


def test_categorical_concat_string_cached() -> None:
    with pl.StringCache():
        df1 = pl.DataFrame({"x": ["A"]}).with_columns(pl.col("x").cast(pl.Categorical))
        df2 = pl.DataFrame({"x": ["B"]}).with_columns(pl.col("x").cast(pl.Categorical))

    out = pl.concat([df1, df2])
    assert out.dtypes == [pl.Categorical]
    assert out["x"].to_list() == ["A", "B"]


def test_list_builder_different_categorical_rev_maps() -> None:
    with pl.StringCache():
        # built with different values, so different rev-map
        s1 = pl.Series(["a", "b"], dtype=pl.Categorical)
        s2 = pl.Series(["c", "d"], dtype=pl.Categorical)

    assert pl.DataFrame({"c": [s1, s2]}).to_dict(as_series=False) == {
        "c": [["a", "b"], ["c", "d"]]
    }


def test_categorical_collect_11408() -> None:
    df = pl.DataFrame(
        data={"groups": ["a", "b", "c"], "cats": ["a", "b", "c"], "amount": [1, 2, 3]},
        schema={"groups": pl.String, "cats": pl.Categorical, "amount": pl.Int8},
    )

    assert df.group_by("groups").agg(
        pl.col("cats").filter(pl.col("amount") == pl.col("amount").min()).first()
    ).sort("groups").to_dict(as_series=False) == {
        "groups": ["a", "b", "c"],
        "cats": ["a", "b", "c"],
    }


def test_categorical_nested_cast_unchecked() -> None:
    s = pl.Series("cat", [["cat"]]).cast(pl.List(pl.Categorical))
    assert pl.Series([s]).to_list() == [[["cat"]]]


def test_categorical_update_lengths() -> None:
    with pl.StringCache():
        s1 = pl.Series(["", ""], dtype=pl.Categorical)
        s2 = pl.Series([None, "", ""], dtype=pl.Categorical)

    s = pl.concat([s1, s2], rechunk=False)
    assert s.null_count() == 1
    assert s.len() == 5


def test_categorical_zip_append_local_different_rev_map() -> None:
    s1 = pl.Series(["cat1", "cat2", "cat1"], dtype=pl.Categorical)
    s2 = pl.Series(["cat2", "cat2", "cat3"], dtype=pl.Categorical)
    with pytest.warns(
        CategoricalRemappingWarning,
        match="Local categoricals have different encodings",
    ):
        s3 = s1.append(s2)
    categories = s3.cat.get_categories()
    assert len(categories) == 3
    assert set(categories) == {"cat1", "cat2", "cat3"}


def test_categorical_zip_extend_local_different_rev_map() -> None:
    s1 = pl.Series(["cat1", "cat2", "cat1"], dtype=pl.Categorical)
    s2 = pl.Series(["cat2", "cat2", "cat3"], dtype=pl.Categorical)
    with pytest.warns(
        CategoricalRemappingWarning,
        match="Local categoricals have different encodings",
    ):
        s3 = s1.extend(s2)
    categories = s3.cat.get_categories()
    assert len(categories) == 3
    assert set(categories) == {"cat1", "cat2", "cat3"}


def test_categorical_zip_with_local_different_rev_map() -> None:
    s1 = pl.Series(["cat1", "cat2", "cat1"], dtype=pl.Categorical)
    mask = pl.Series([True, False, False])
    s2 = pl.Series(["cat2", "cat2", "cat3"], dtype=pl.Categorical)
    with pytest.warns(
        CategoricalRemappingWarning,
        match="Local categoricals have different encodings",
    ):
        s3 = s1.zip_with(mask, s2)
    categories = s3.cat.get_categories()
    assert len(categories) == 3
    assert set(categories) == {"cat1", "cat2", "cat3"}


def test_categorical_vstack_with_local_different_rev_map() -> None:
    df1 = pl.DataFrame({"a": pl.Series(["a", "b", "c"], dtype=pl.Categorical)})
    df2 = pl.DataFrame({"a": pl.Series(["d", "e", "f"], dtype=pl.Categorical)})
    with pytest.warns(
        CategoricalRemappingWarning,
        match="Local categoricals have different encodings",
    ):
        df3 = df1.vstack(df2)
    assert df3.get_column("a").cat.get_categories().to_list() == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]
    assert df3.get_column("a").cast(pl.UInt32).to_list() == [0, 1, 2, 3, 4, 5]


def test_shift_over_13041() -> None:
    df = pl.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 1],
            "cat_col": pl.Series(["a", "b", "c", "d", "e", "f"], dtype=pl.Categorical),
        }
    )
    result = df.with_columns(pl.col("cat_col").shift(2).over("id"))

    assert result.to_dict(as_series=False) == {
        "id": [0, 0, 0, 1, 1, 1],
        "cat_col": [None, None, "a", None, None, "d"],
    }


@pytest.mark.parametrize("context", [pl.StringCache(), contextlib.nullcontext()])
@pytest.mark.parametrize("ordering", ["physical", "lexical"])
def test_sort_categorical_retain_none(
    context: contextlib.AbstractContextManager,  # type: ignore[type-arg]
    ordering: Literal["physical", "lexical"],
) -> None:
    with context:
        df = pl.DataFrame(
            [
                pl.Series(
                    "e",
                    ["foo", None, "bar", "ham", None],
                    dtype=pl.Categorical(ordering=ordering),
                )
            ]
        )

        df_sorted = df.with_columns(pl.col("e").sort())
        assert (
            df_sorted.get_column("e").null_count()
            == df.get_column("e").null_count()
            == 2
        )
        if ordering == "lexical":
            assert df_sorted.get_column("e").to_list() == [
                None,
                None,
                "bar",
                "foo",
                "ham",
            ]


def test_cast_from_cat_to_numeric() -> None:
    cat_series = pl.Series(
        "cat_series",
        ["0.69845702", "0.69317475", "2.43642724", "-0.95303469", "0.60684237"],
    ).cast(pl.Categorical)
    maximum = cat_series.cast(pl.Float32).max()
    assert abs(maximum - 2.43642724) < 1e-6  # type: ignore[operator]

    s = pl.Series(["1", "2", "3"], dtype=pl.Categorical)
    assert s.cast(pl.UInt8).sum() == 6


def test_cat_preserve_lexical_ordering_on_clear() -> None:
    s = pl.Series("a", ["a", "b"], dtype=pl.Categorical(ordering="lexical"))
    s2 = s.clear()
    assert s.dtype == s2.dtype


def test_cat_preserve_lexical_ordering_on_concat() -> None:
    dtype = pl.Categorical(ordering="lexical")

    df = pl.DataFrame({"x": ["b", "a", "c"]}).with_columns(pl.col("x").cast(dtype))
    df2 = pl.concat([df, df])
    assert df2["x"].dtype == dtype


def test_cat_append_lexical_sorted_flag() -> None:
    df = pl.DataFrame({"x": [0, 1, 1], "y": ["B", "B", "A"]}).with_columns(
        pl.col("y").cast(pl.Categorical(ordering="lexical"))
    )
    df2 = pl.concat([part.sort("y") for part in df.partition_by("x")])

    assert not (df2["y"].is_sorted())


def test_get_cat_categories_multiple_chunks() -> None:
    df = pl.DataFrame(
        [
            pl.Series("e", ["a", "b"], pl.Enum(["a", "b"])),
        ]
    )
    df = pl.concat(
        [df for _ in range(100)], how="vertical", rechunk=False, parallel=True
    )
    df_cat = df.lazy().select(pl.col("e").cat.get_categories()).collect()
    assert len(df_cat) == 2

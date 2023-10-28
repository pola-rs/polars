from __future__ import annotations

import io
from typing import Any

import pytest

import polars as pl
from polars import StringCache
from polars.exceptions import StringCacheMismatchError
from polars.testing import assert_frame_equal


@StringCache()
def test_categorical_outer_join() -> None:
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
        {"key1": [42], "key2": ["bar"], "val1": [1], "val2": [2]},
        schema_overrides={"key2": pl.Categorical},
    )

    out = df1.join(df2, on=["key1", "key2"], how="outer").collect()
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

    df = dfa.join(dfb, on="key", how="outer")
    # the cast is important to test the rev map
    assert df["key"].cast(pl.Utf8).to_list() == ["bar", "baz", "foo"]


def test_read_csv_categorical() -> None:
    f = io.BytesIO()
    f.write(b"col1,col2,col3,col4,col5,col6\n'foo',2,3,4,5,6\n'bar',8,9,10,11,12")
    f.seek(0)
    df = pl.read_csv(f, has_header=True, dtypes={"col1": pl.Categorical})
    assert df["col1"].dtype == pl.Categorical


def test_cat_to_dummies() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3, 4], "bar": ["a", "b", "a", "c"]})
    df = df.with_columns(pl.col("bar").cast(pl.Categorical))
    assert df.to_dummies().to_dict(False) == {
        "foo_1": [1, 0, 0, 0],
        "foo_2": [0, 1, 0, 0],
        "foo_3": [0, 0, 1, 0],
        "foo_4": [0, 0, 0, 1],
        "bar_a": [1, 0, 1, 0],
        "bar_b": [0, 1, 0, 0],
        "bar_c": [0, 0, 0, 1],
    }


def test_categorical_describe_3487() -> None:
    # test if we don't err
    df = pl.DataFrame({"cats": ["a", "b"]})
    df = df.with_columns(pl.col("cats").cast(pl.Categorical))
    df.describe()


@StringCache()
def test_categorical_is_in_list() -> None:
    # this requires type coercion to cast.
    # we should not cast within the function as this would be expensive within a
    # group by context that would be a cast per group
    df = pl.DataFrame(
        {"a": [1, 2, 3, 1, 2], "b": ["a", "b", "c", "d", "e"]}
    ).with_columns(pl.col("b").cast(pl.Categorical))

    cat_list = ("a", "b", "c")
    assert df.filter(pl.col("b").is_in(cat_list)).to_dict(False) == {
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
    assert df.group_by("key").count()["count"].to_list() == [4, 4]


def test_categorical_error_on_local_cmp() -> None:
    df_cat = pl.DataFrame(
        [
            pl.Series("a_cat", ["c", "a", "b", "c", "b"], dtype=pl.Categorical),
            pl.Series("b_cat", ["F", "G", "E", "G", "G"], dtype=pl.Categorical),
        ]
    )
    with pytest.raises(
        pl.ComputeError,
        match=(
            "cannot compare categoricals originating from different sources; consider"
            " setting a global string cache"
        ),
    ):
        df_cat.filter(pl.col("a_cat") == pl.col("b_cat"))


def test_cast_null_to_categorical() -> None:
    assert pl.DataFrame().with_columns(
        [pl.lit(None).cast(pl.Categorical).alias("nullable_enum")]
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
    ).to_dict(False) == {"label": [None, "bar"], "value": [3, 9]}


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


def test_categorical_max_null_5437() -> None:
    assert (
        pl.DataFrame({"strings": ["c", "b", "a", "c"], "values": [0, 1, 2, 3]})
        .with_columns(pl.col("strings").cast(pl.Categorical).alias("cats"))
        .select(pl.all().max())
    ).to_dict(False) == {"strings": ["c"], "values": [3], "cats": [None]}


def test_categorical_in_struct_nulls() -> None:
    s = pl.Series(
        "job", ["doctor", "waiter", None, None, None, "doctor"], pl.Categorical
    )
    df = pl.DataFrame([s])
    s = (df.select(pl.col("job").value_counts(sort=True)))["job"]

    assert s[0] == {"job": None, "counts": 3}
    assert s[1] == {"job": "doctor", "counts": 2}
    assert s[2] == {"job": "waiter", "counts": 1}


def test_cast_inner_categorical() -> None:
    dtype = pl.List(pl.Categorical)
    out = pl.Series("foo", [["a"], ["a", "b"]]).cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == [["a"], ["a", "b"]]

    with pytest.raises(
        pl.ComputeError, match=r"casting to categorical not allowed in `list.eval`"
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
            [pl.col("cats").cast(pl.Utf8).cast(pl.Categorical)]
        )
        assert df.filter(pl.col("cats").is_in(["1", "2"])).to_dict(False) == {
            "cats": ["1", "2"]
        }


@StringCache()
def test_categorical_sort_order(monkeypatch: Any) -> None:
    # create the categorical ordering first
    pl.Series(["foo", "bar", "baz"], dtype=pl.Categorical)
    df = pl.DataFrame(
        {
            "n": [0, 0, 0],
            # use same categories in different order
            "x": pl.Series(["baz", "bar", "foo"], dtype=pl.Categorical),
        }
    )

    assert df.sort(["n", "x"])["x"].to_list() == ["foo", "bar", "baz"]
    assert df.with_columns(pl.col("x").cat.set_ordering("lexical")).sort(["n", "x"])[
        "x"
    ].to_list() == ["bar", "baz", "foo"]
    monkeypatch.setenv("POLARS_ROW_FMT_SORT", "1")
    assert df.sort(["n", "x"])["x"].to_list() == ["foo", "bar", "baz"]
    assert df.with_columns(pl.col("x").cat.set_ordering("lexical")).sort(["n", "x"])[
        "x"
    ].to_list() == ["bar", "baz", "foo"]


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
        maintain_order=True, by=["group"]
    ).all().with_columns(pl.col("letter").list.len().alias("c_group")).group_by(
        by=["c_group"], maintain_order=True
    ).agg(pl.col("letter")).to_dict(False) == {
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
    assert pl.DataFrame(
        {"col": ["a", None, "a"]}, schema={"col": pl.Categorical}
    ).fill_null("a").with_columns(pl.col("col").to_physical().alias("code")).to_dict(
        False
    ) == {
        "col": ["a", "a", "a"],
        "code": [0, 0, 0],
    }


@StringCache()
def test_categorical_fill_null_stringcache() -> None:
    df = pl.LazyFrame(
        {"index": [1, 2, 3], "cat": ["a", "b", None]},
        schema={"index": pl.Int64(), "cat": pl.Categorical()},
    )
    a = df.select(pl.col("cat").fill_null("hi")).collect()

    assert a.to_dict(False) == {"cat": ["a", "b", "hi"]}
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

    assert pl.DataFrame({"c": [s1, s2]}).to_dict(False) == {
        "c": [["a", "b"], ["c", "d"]]
    }


def test_categorical_collect_11408() -> None:
    df = pl.DataFrame(
        data={"groups": ["a", "b", "c"], "cats": ["a", "b", "c"], "amount": [1, 2, 3]},
        schema={"groups": pl.Utf8, "cats": pl.Categorical, "amount": pl.Int8},
    )

    assert df.group_by("groups").agg(
        pl.col("cats").filter(pl.col("amount") == pl.col("amount").min()).first()
    ).sort("groups").to_dict(False) == {
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

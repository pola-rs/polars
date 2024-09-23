from typing import Iterator

import pytest

import polars as pl
from polars.exceptions import CategoricalRemappingWarning
from polars.testing import assert_frame_equal


@pytest.fixture(autouse=True)
def _disable_string_cache() -> Iterator[None]:
    """Fixture to make sure the string cache is disabled before and after each test."""
    pl.disable_string_cache()
    yield
    pl.disable_string_cache()


def sc(set: bool) -> None:
    """Short syntax for asserting whether the global string cache is being used."""
    assert pl.using_string_cache() is set


def test_string_cache_enable_disable() -> None:
    sc(False)
    pl.enable_string_cache()
    sc(True)
    pl.disable_string_cache()
    sc(False)


def test_string_cache_enable_disable_repeated() -> None:
    sc(False)
    pl.enable_string_cache()
    sc(True)
    pl.enable_string_cache()
    sc(True)
    pl.disable_string_cache()
    sc(False)
    pl.disable_string_cache()
    sc(False)


def test_string_cache_context_manager() -> None:
    sc(False)
    with pl.StringCache():
        sc(True)
    sc(False)


def test_string_cache_context_manager_nested() -> None:
    sc(False)
    with pl.StringCache():
        sc(True)
        with pl.StringCache():
            sc(True)
        sc(True)
    sc(False)


def test_string_cache_context_manager_mixed_with_enable_disable() -> None:
    sc(False)
    with pl.StringCache():
        sc(True)
        pl.enable_string_cache()
        sc(True)
    sc(True)

    with pl.StringCache():
        sc(True)
    sc(True)

    with pl.StringCache():
        sc(True)
        with pl.StringCache():
            sc(True)
            pl.disable_string_cache()
            sc(True)
        sc(True)
    sc(False)

    with pl.StringCache():
        sc(True)
        pl.disable_string_cache()
        sc(True)
    sc(False)


def test_string_cache_decorator() -> None:
    @pl.StringCache()
    def my_function() -> None:
        sc(True)

    sc(False)
    my_function()
    sc(False)


def test_string_cache_decorator_mixed_with_enable() -> None:
    @pl.StringCache()
    def my_function() -> None:
        sc(True)
        pl.enable_string_cache()
        sc(True)

    sc(False)
    my_function()
    sc(True)


def test_string_cache_join() -> None:
    df1 = pl.DataFrame({"a": ["foo", "bar", "ham"], "b": [1, 2, 3]})
    df2 = pl.DataFrame({"a": ["eggs", "spam", "foo"], "c": [2, 2, 3]})

    # ensure cache is off when casting to categorical; the join will fail
    pl.disable_string_cache()
    assert pl.using_string_cache() is False

    with pytest.warns(
        CategoricalRemappingWarning,
        match="Local categoricals have different encodings",
    ):
        df1a = df1.with_columns(pl.col("a").cast(pl.Categorical))
        df2a = df2.with_columns(pl.col("a").cast(pl.Categorical))
        out = df1a.join(df2a, on="a", how="inner")

    expected = pl.DataFrame(
        {"a": ["foo"], "b": [1], "c": [3]}, schema_overrides={"a": pl.Categorical}
    )

    # Can not do equality checks on local categoricals with different categories
    assert_frame_equal(out, expected, categorical_as_str=True)

    # now turn on the cache
    pl.enable_string_cache()
    assert pl.using_string_cache() is True

    df1b = df1.with_columns(pl.col("a").cast(pl.Categorical))
    df2b = df2.with_columns(pl.col("a").cast(pl.Categorical))
    out = df1b.join(df2b, on="a", how="inner")

    expected = pl.DataFrame(
        {"a": ["foo"], "b": [1], "c": [3]}, schema_overrides={"a": pl.Categorical}
    )
    assert_frame_equal(out, expected)


def test_string_cache_eager_lazy() -> None:
    # tests if the global string cache is really global and not interfered by the lazy
    # execution. first the global settings was thread-local and this breaks with the
    # parallel execution of lazy
    with pl.StringCache():
        df1 = pl.DataFrame(
            {"region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"]}
        ).select([pl.col("region_ids").cast(pl.Categorical)])

        df2 = pl.DataFrame(
            {"seq_name": ["reg4", "reg2", "reg1"], "score": [3.0, 1.0, 2.0]}
        ).select([pl.col("seq_name").cast(pl.Categorical), pl.col("score")])

        expected = pl.DataFrame(
            {
                "region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"],
                "score": [2.0, 1.0, None, 3.0, None],
            }
        ).with_columns(pl.col("region_ids").cast(pl.Categorical))

        result = df1.join(df2, left_on="region_ids", right_on="seq_name", how="left")
        assert_frame_equal(result, expected)

        # also check row-wise categorical insert.
        # (column-wise is preferred, but this shouldn't fail)
        for params in (
            {"schema": [("region_ids", pl.Categorical)]},
            {
                "schema": ["region_ids"],
                "schema_overrides": {"region_ids": pl.Categorical},
            },
        ):
            df3 = pl.DataFrame(
                data=[["reg1"], ["reg2"], ["reg3"], ["reg4"], ["reg5"]],
                orient="row",
                **params,
            )
            assert_frame_equal(df1, df3)

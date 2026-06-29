import pytest
from hypothesis import given

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal
from polars.testing.parametric.strategies.core import dataframes


@pytest.mark.parametrize("key1", ["col0", "col1"])
@pytest.mark.parametrize("key2", ["col0", "col1"])
@pytest.mark.parametrize("mo1", [False, True])
@pytest.mark.parametrize("mo2", [False, True])
@given(df=dataframes(min_cols=2, max_cols=2))
def test_sort_node_collapse(
    df: pl.DataFrame, mo1: bool, mo2: bool, key1: str, key2: str
) -> None:
    q = (
        df.with_row_index()
        .lazy()
        .sort(key1, maintain_order=mo1)
        .sort(key2, maintain_order=mo2)
        .select(pl.col("index"))
    )
    lp = q.explain()
    lp_expect = "SORT BY [maintain_order: true]" if mo1 and mo2 else "SORT BY"
    assert lp.count("SORT BY") == 1
    if not mo2:
        assert f'{lp_expect} [col("{key2}")]' in lp
    elif key1 == key2:
        assert f'{lp_expect} [col("{key1}")]' in lp
    else:
        assert f'{lp_expect} [col("{key2}"), col("{key1}")]' in lp
    actual = q.collect()
    expected = (
        df.with_row_index()
        .sort(key1, maintain_order=mo1)
        .sort(key2, maintain_order=mo2)
        .select(pl.col("index"))
    )
    assert_frame_equal(actual, expected, check_row_order=mo1 and mo2)


@pytest.mark.parametrize("mo1", [False, True])
def test_sort_node_collapse_multiple(mo1: bool) -> None:
    df = pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4]})
    for q in [
        df.lazy().sort("a", "b", maintain_order=mo1).sort("a", maintain_order=True),
        df.lazy().sort("a", maintain_order=mo1).sort("a", "b", maintain_order=True),
    ]:
        assert q.explain().count("SORT BY") == 1
        if mo1:
            assert 'SORT BY [maintain_order: true] [col("a"), col("b")]' in q.explain()
        else:
            assert 'SORT BY [col("a"), col("b")]' in q.explain()
        actual = q.collect()
        expected = df.sort("a", "b", maintain_order=mo1)
        assert_frame_equal(actual, expected, check_row_order=mo1)


@pytest.mark.parametrize("key1", ["col0", "col1"])
@pytest.mark.parametrize("key2", ["col0", "col1"])
@pytest.mark.parametrize("maintain_order", [False, True])
@given(df=dataframes(min_cols=2, max_cols=2))
def test_sort_node_prune_hint(
    df: pl.DataFrame, key1: str, key2: str, maintain_order: bool
) -> None:
    q = (
        df.sort(key1)
        .with_row_index("idx")
        .lazy()
        .set_sorted(key1)
        .sort(key2, maintain_order=maintain_order)
        .select(pl.col("idx"))
    )
    lp = q.explain()
    if key1 == key2:
        assert "SORT BY" not in lp
    else:
        assert "SORT BY" in lp
    actual = q.collect()
    expected = (
        df.sort(key1)
        .with_row_index("idx")
        .sort(key2, maintain_order=maintain_order)
        .select(pl.col("idx"))
    )
    assert_frame_equal(actual, expected, check_row_order=maintain_order)


def test_sort_node_prune_hint_multiple() -> None:
    df = pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4]}).with_row_index("idx")
    q = df.lazy().set_sorted("a", "b").sort("a").select(pl.col("idx"))
    assert "SORT BY" not in q.explain()
    q = (
        df.lazy()
        .set_sorted("a")
        .sort("a", "b", maintain_order=False)
        .select(pl.col("idx"))
    )
    assert 'SORT BY [col("a"), col("b")]' in q.explain()
    q = (
        df.lazy()
        .set_sorted("a")
        .sort("a", "b", maintain_order=True)
        .select(pl.col("idx"))
    )
    assert 'SORT BY [maintain_order: true] [col("a"), col("b")]' in q.explain()

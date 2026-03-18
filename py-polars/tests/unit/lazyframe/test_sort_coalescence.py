import pytest
from hypothesis import given

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal
from polars.testing.parametric.strategies.core import dataframes


@pytest.mark.parametrize("mo1", [False, True])
@pytest.mark.parametrize("mo2", [False, True])
@pytest.mark.parametrize("key1", ["col0", "col1"])
@pytest.mark.parametrize("key2", ["col0", "col1"])
@given(df=dataframes(min_cols=2, max_cols=2))
def test_sort_node_coalescence(
    df: pl.DataFrame, mo1: bool, mo2: bool, key1: str, key2: str
) -> None:
    q = df.lazy().sort(key1, maintain_order=mo1).sort(key2, maintain_order=mo2)
    lp = q.explain()
    if not mo2:
        assert lp.count("SORT BY") == 1
        assert "[maintain_order: true]" not in lp
    elif key1 == key2:
        assert lp.count("SORT BY") == 1
        if mo1 and mo2:
            assert "[maintain_order: true]" in lp
        else:
            assert "[maintain_order: true]" not in lp
    else:
        assert lp.count("SORT BY") == 2
    actual = q.collect()
    expected = df.sort(key1, maintain_order=mo1).sort(key2, maintain_order=mo2)
    assert_frame_equal(actual, expected, check_row_order=mo1 and mo2)


@pytest.mark.parametrize("mo1", [False, True])
@pytest.mark.parametrize("mo2", [False, True])
def test_sort_node_coalescence_multiple(mo1: bool, mo2: bool) -> None:
    df = pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4]})
    for q in [
        df.lazy().sort("a", "b", maintain_order=mo1).sort("a", maintain_order=mo2),
        df.lazy().sort("a", maintain_order=mo1).sort("a", "b", maintain_order=mo2),
    ]:
        assert q.explain().count("SORT BY") == 1
        if mo1 and mo2:
            assert 'SORT BY [maintain_order: true] [col("a"), col("b")]' in q.explain()
        else:
            assert 'SORT BY [col("a"), col("b")]' in q.explain()
        actual = q.collect()
        expected = df.sort("a", "b", maintain_order=mo1 and mo2)
        assert_frame_equal(actual, expected, check_row_order=mo1 and mo2)

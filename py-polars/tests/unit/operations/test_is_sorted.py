import polars as pl


def test_is_sorted() -> None:
    assert not pl.Series([1, 2, 5, None, 2, None]).is_sorted()
    assert pl.Series([1, 2, 4, None, None]).is_sorted(nulls_last=True)
    assert pl.Series([None, None, 1, 2, 4]).is_sorted(nulls_last=False)
    assert not pl.Series([None, 1, None, 2, 4]).is_sorted()
    assert not pl.Series([None, 1, 2, 3, -1, 4]).is_sorted(nulls_last=False)
    assert not pl.Series([1, 2, 3, -1, 4, None, None]).is_sorted(nulls_last=True)
    assert not pl.Series([1, 2, 3, -1, 4]).is_sorted()
    assert pl.Series([1, 2, 3, 4]).is_sorted()
    assert pl.Series([5, 2, 1, 1, -1]).is_sorted(descending=True)
    assert pl.Series([None, None, 5, 2, 1, 1, -1]).is_sorted(
        descending=True, nulls_last=False
    )
    assert pl.Series([5, 2, 1, 1, -1, None, None]).is_sorted(
        descending=True, nulls_last=True
    )
    assert not pl.Series([5, None, 2, 1, 1, -1, None, None]).is_sorted(
        descending=True, nulls_last=True
    )
    assert not pl.Series([5, 2, 1, 10, 1, -1, None, None]).is_sorted(
        descending=True, nulls_last=True
    )

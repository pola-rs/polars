import polars as pl


def test_cat_uses_lexical_ordering() -> None:
    s = pl.Series(["a", "b", None, "b"]).cast(pl.Categorical)
    assert s.cat.uses_lexical_ordering() is False

    s = s.cat.set_ordering("lexical")
    assert s.cat.uses_lexical_ordering() is True

    s = s.cat.set_ordering("physical")
    assert s.cat.uses_lexical_ordering() is False

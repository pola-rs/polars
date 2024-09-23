import pytest

import polars as pl


def test_list_constructor_strictness() -> None:
    with pytest.raises(TypeError, match="setting `strict=False`"):
        pl.Series([[1], ["two"]], strict=True)

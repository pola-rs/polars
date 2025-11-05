import pytest

import polars as pl


def test_catalog_require_https() -> None:
    with pytest.raises(ValueError):
        pl.Catalog("http://")

    pl.Catalog("https://")
    pl.Catalog("http://", require_https=False)

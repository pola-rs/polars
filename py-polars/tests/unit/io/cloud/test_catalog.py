import re

import pytest

import polars as pl
from polars.exceptions import ArgumentRemovedError


def test_catalog_require_https() -> None:
    with pytest.raises(ValueError):
        pl.Catalog("http://")

    pl.Catalog("https://")
    pl.Catalog("http://", require_https=False)


def test_catalog_scan_table_retries_removed() -> None:
    msg = 'Pass {"max_retries": n} via `storage_options` instead.'
    with pytest.raises(ArgumentRemovedError, match=re.escape(msg)):
        pl.Catalog("https://").scan_table("c", "n", "t", retries=3)  # type: ignore[call-arg]

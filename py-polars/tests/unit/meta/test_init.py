import importlib
import importlib.metadata
import re

import pytest

import polars as pl
from polars.exceptions import AttributeRemovedError


def test_init_nonexistent_attribute() -> None:
    with pytest.raises(
        AttributeError, match="module 'polars' has no attribute 'stroopwafel'"
    ):
        pl.stroopwafel  # type: ignore[attr-defined]


def test_init_exceptions_not_found() -> None:
    msg = "accessing `ComputeError` from the top-level `polars` module was deprecated in version 1.0.0"
    with pytest.raises(AttributeRemovedError, match=re.escape(msg)):
        pl.ComputeError  # type: ignore[attr-defined]


def test_dtype_groups_not_found() -> None:
    msg = "`INTEGER_DTYPES` was deprecated in version 1.0.0"
    with pytest.raises(AttributeRemovedError, match=re.escape(msg)):
        pl.INTEGER_DTYPES  # type: ignore[attr-defined]


def test_import_all() -> None:
    exec("from polars import *")


def test_version() -> None:
    # This has already gone wrong once (#23940), preventing future problems.
    lhs = (
        pl.__version__.replace("-alpha.", "a")
        .replace("-beta.", "b")
        .replace("-rc.", "rc")
    )
    rhs = importlib.metadata.version("polars")

    assert lhs == rhs, (
        f"`static PYPOLARS_VERSION` ({lhs}) at `crates/polars-python/src/c_api/mod.rs` "
        f"does not match importlib package metadata version ({rhs})"
    )


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("arctan2d", "use `arctan2` followed by `.degrees()` instead."),
        ("groups", "use `df.with_row_index().group_by(...).agg(pl.col('index'))`"),
        ("read_csv_batched", "use `scan_csv` instead"),
        ("threadpool_size", "it was renamed; use `thread_pool_size` instead."),
    ],
)
def test_removed_functions(name: str, match: str) -> None:
    with pytest.raises(AttributeRemovedError, match=re.escape(match)):
        getattr(pl, name)

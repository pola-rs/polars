from __future__ import annotations

import copy
import os.path
from typing import cast

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_copy() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    assert_frame_equal(copy.copy(df), df)
    assert_frame_equal(copy.deepcopy(df), df)

    a = pl.Series("a", [1, 2])
    assert_series_equal(copy.copy(a), a)
    assert_series_equal(copy.deepcopy(a), a)


def test_categorical_round_trip() -> None:
    df = pl.DataFrame({"ints": [1, 2, 3], "cat": ["a", "b", "c"]})
    df = df.with_columns(pl.col("cat").cast(pl.Categorical))

    tbl = df.to_arrow()
    assert "dictionary" in str(tbl["cat"].type)

    df2 = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert df2.dtypes == [pl.Int64, pl.Categorical]


def test_from_different_chunks() -> None:
    s0 = pl.Series("a", [1, 2, 3, 4, None])
    s1 = pl.Series("b", [1, 2])
    s11 = pl.Series("b", [1, 2, 3])
    s1.append(s11)

    # check we don't panic
    df = pl.DataFrame([s0, s1])
    df.to_arrow()
    df = pl.DataFrame([s0, s1])
    out = df.to_pandas()
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (5, 2)


def test_unit_io_subdir_has_no_init() -> None:
    # --------------------------------------------------------------------------------
    # If this test fails it means an '__init__.py' was added to 'tests/unit/io'.
    # See https://github.com/pola-rs/polars/pull/6889 for why this can cause issues.
    # --------------------------------------------------------------------------------
    # TLDR: it can mask the builtin 'io' module, causing a fatal python error.
    # --------------------------------------------------------------------------------
    io_dir = os.path.dirname(__file__)
    assert io_dir.endswith(f"unit{os.path.sep}io")
    assert not os.path.exists(
        f"{io_dir}{os.path.sep}__init__.py"
    ), "Found undesirable '__init__.py' in the 'unit.io' tests subdirectory"

from __future__ import annotations

import io
import operator
import pickle
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


TestExtension = pl.Extension(
    name="testing.test_extension",
    storage=pl.Int8,
    metadata="A test extension type",
)

pl.register_extension_type("testing.test_extension", pl.Extension)


class PythonTestExtension(pl.datatypes.BaseExtension):
    """A test extension type defined in Python."""

    def __init__(self, storage: pl.DataType) -> None:
        super().__init__(name="testing.python_test_extension", storage=storage)

    def __repr__(self) -> str:
        return "PythonTestExtension"

    def _string_repr(self) -> str:
        return "pythontestext"


pl.register_extension_type("testing.python_test_extension", PythonTestExtension)

pl.register_extension_type("testing.test_storage_extension", as_storage=True)


def test_extension_df_constructor() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"]},
        schema={"a": TestExtension, "b": PythonTestExtension(pl.String)},
    )

    assert_frame_equal(
        df.select(pl.all().ext.storage()),
        pl.DataFrame(
            {"a": [1, 2, 3], "b": ["a", "b", "c"]},
            schema={"a": pl.Int8, "b": pl.String},
        ),
    )


def test_extension_series_to_storage_and_back() -> None:
    s = pl.Series("a", [10, 20, 30], dtype=TestExtension)

    storage = s.ext.storage()
    assert_series_equal(storage, pl.Series("a", [10, 20, 30], dtype=pl.Int8))

    ext = storage.ext.to(TestExtension)
    assert_series_equal(ext, pl.Series("a", [10, 20, 30], dtype=TestExtension))


def test_extension_parquet_roundtrip() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [0.1, 0.2, 0.3],
            "d": [True, False, True],
            "e": ["foo", "bar", "foo"],
        },
        schema={
            "a": TestExtension,
            "b": PythonTestExtension(pl.String),
            "c": PythonTestExtension(pl.Float64),
            "d": PythonTestExtension(pl.Boolean),
            "e": PythonTestExtension(pl.Categorical),
        },
    )

    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    df_read = pl.read_parquet(buffer)

    assert_frame_equal(df, df_read)

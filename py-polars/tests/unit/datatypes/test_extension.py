from __future__ import annotations

import io
import pickle
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import JoinStrategy, PolarsDataType

TestExtension = pl.Extension(
    name="testing.test_extension",
    storage=pl.Int8,
    metadata="A test extension type",
)

pl.register_extension_type("testing.test_extension", pl.Extension)


class PythonTestExtension(pl.datatypes.BaseExtension):
    """A test extension type defined in Python."""

    def __init__(self, storage: PolarsDataType) -> None:
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


ROUNDTRIP_DF = pl.DataFrame(
    {
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
        "c": [0.1, 0.2, float("nan")],
        "d": [True, False, True],
        "e": ["foo", "bar", "foo"],
        "f": [b"foo", b"bar", b"foo"],
        "g": [[1, 2, 3], [4, 5], [6, 7, 2**100]],
        "h": [[1, 2, 3], [4, 5, 6], [100, 1200, -1230]],
        "i": [{"a": 1, "b": ["foo"]}, {"a": 1, "b": ["bar"]}, {"a": 1, "b": ["foo"]}],
    },
    schema={
        "a": TestExtension,
        "b": PythonTestExtension(pl.String),
        "c": PythonTestExtension(pl.Float64),
        "d": PythonTestExtension(pl.Boolean),
        "e": PythonTestExtension(pl.Categorical),
        "f": PythonTestExtension(pl.Binary),
        "g": PythonTestExtension(pl.List(pl.UInt128)),
        "h": PythonTestExtension(pl.Array(pl.Int64, 3)),
        "i": PythonTestExtension(
            pl.Struct(
                {
                    "a": pl.Int8,
                    "b": PythonTestExtension(pl.List(pl.Categorical)),
                }
            )
        ),
    },
)


def test_extension_parquet_roundtrip() -> None:
    df = ROUNDTRIP_DF
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    df_read = pl.read_parquet(buffer)
    assert_frame_equal(df, df_read)


def test_extension_pickle_roundtrip() -> None:
    df = ROUNDTRIP_DF
    df_pickled = pickle.loads(pickle.dumps(df))
    assert_frame_equal(df, df_pickled)


def test_extension_ipc_roundtrip() -> None:
    df = ROUNDTRIP_DF
    buffer = io.BytesIO()
    df.write_ipc(buffer)
    buffer.seek(0)
    df_read = pl.read_ipc(buffer)
    assert_frame_equal(df, df_read)


def test_to_from_storage_roundtrip() -> None:
    df = ROUNDTRIP_DF
    df_storage = df.select(pl.all().ext.storage())
    df_ext = df_storage.select([pl.col(c).ext.to(df.schema[c]) for c in df.columns])
    assert_frame_equal(df, df_ext)


def test_extension_gather() -> None:
    df = ROUNDTRIP_DF
    df_storage = df.select(pl.all().ext.storage())
    result = df.select(pl.all().gather([2, 0]))
    expected = df_storage.select(pl.all().gather([2, 0])).select(
        [pl.col(c).ext.to(df.schema[c]) for c in df.columns]
    )
    assert_frame_equal(result, expected)


def test_extension_when_then_otherwise() -> None:
    df = ROUNDTRIP_DF.with_columns(pl.Series("mask", [True, False, None]))
    df_storage = df.select(pl.all().ext.storage())
    result = df.select(
        pl.when(pl.col("mask")).then(c).otherwise(pl.col(c).reverse())
        for c in df.columns
        if c != "mask"
    )
    expected = df_storage.select(
        pl.when(pl.col("mask")).then(c).otherwise(pl.col(c).reverse())
        for c in df.columns
        if c != "mask"
    ).select([pl.col(c).ext.to(df.schema[c]) for c in df.columns if c != "mask"])
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
def test_extension_join_payload(how: JoinStrategy) -> None:
    df = ROUNDTRIP_DF.with_columns(pl.Series("key", [1, 2, 3]))
    df_storage = df.select(pl.all().ext.storage())
    print(df)
    print(df_storage)
    other = pl.DataFrame({"key": [1, 3, 1, 1, 2, 3, 1, 4, 1]})

    result = df.join(other, on="key", how=how)
    expected = df_storage.join(other, on="key", how=how).with_columns(
        [pl.col(c).ext.to(df.schema[c]) for c in df.columns if c != "key"]
    )
    print(result, expected)
    assert_frame_equal(result, expected, check_row_order=False)

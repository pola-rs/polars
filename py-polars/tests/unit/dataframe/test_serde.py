from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


def test_df_serialize() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).sort("a")
    result = df.serialize()
    expected = '{"columns":[{"name":"a","datatype":"Int64","bit_settings":"SORTED_ASC","values":[1,2,3]},{"name":"b","datatype":"Int64","bit_settings":"","values":[4,5,6]}]}'
    assert result == expected


@pytest.mark.parametrize("buf", [io.BytesIO(), io.StringIO()])
def test_to_from_buffer(df: pl.DataFrame, buf: io.IOBase) -> None:
    df.serialize(buf)
    buf.seek(0)
    read_df = pl.DataFrame.deserialize(buf)
    assert_frame_equal(df, read_df, categorical_as_str=True)


@pytest.mark.write_disk()
def test_to_from_file(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.json"
    df.serialize(file_path)
    out = pl.DataFrame.deserialize(file_path)

    assert_frame_equal(df, out, categorical_as_str=True)


def test_write_json_to_string() -> None:
    # Tests if it runs if no arg given
    df = pl.DataFrame({"a": [1, 2, 3]})
    expected_str = '{"columns":[{"name":"a","datatype":"Int64","bit_settings":"","values":[1,2,3]}]}'
    assert df.serialize() == expected_str


def test_write_json(df: pl.DataFrame) -> None:
    # Text-based conversion loses time info
    df = df.select(pl.all().exclude(["cat", "time"]))
    s = df.serialize()
    f = io.BytesIO()
    f.write(s.encode())
    f.seek(0)
    out = pl.DataFrame.deserialize(f)
    assert_frame_equal(out, df)

    file = io.BytesIO()
    df.serialize(file)
    file.seek(0)
    out = pl.DataFrame.deserialize(file)
    assert_frame_equal(out, df)


def test_df_serde_enum() -> None:
    dtype = pl.Enum(["foo", "bar", "ham"])
    df = pl.DataFrame([pl.Series("e", ["foo", "bar", "ham"], dtype=dtype)])
    buf = io.StringIO()
    df.serialize(buf)
    buf.seek(0)
    df_in = pl.DataFrame.deserialize(buf)
    assert df_in.schema["e"] == dtype


@pytest.mark.parametrize(
    ("data", "dtype"),
    [
        ([[1, 2, 3], [None, None, None], [1, None, 3]], pl.Array(pl.Int32(), shape=3)),
        ([["a", "b"], [None, None]], pl.Array(pl.Utf8, shape=2)),
        ([[True, False, None], [None, None, None]], pl.Array(pl.Utf8, shape=3)),
        (
            [[[1, 2, 3], [4, None, 5]], None, [[None, None, 2]]],
            pl.List(pl.Array(pl.Int32(), shape=3)),
        ),
        (
            [
                [datetime(1991, 1, 1), datetime(1991, 1, 1), None],
                [None, None, None],
            ],
            pl.Array(pl.Datetime, shape=3),
        ),
    ],
)
def test_write_read_json_array(data: Any, dtype: pl.DataType) -> None:
    df = pl.DataFrame({"foo": data}, schema={"foo": dtype})
    buf = io.StringIO()
    df.serialize(buf)
    buf.seek(0)
    deserialized_df = pl.DataFrame.deserialize(buf)
    assert_frame_equal(deserialized_df, df)


@pytest.mark.parametrize(
    ("data", "dtype"),
    [
        (
            [
                [
                    datetime(1997, 10, 1),
                    datetime(2000, 1, 2, 10, 30, 1),
                ],
                [None, None],
            ],
            pl.Array(pl.Datetime, shape=2),
        ),
        (
            [[date(1997, 10, 1), date(2000, 1, 1)], [None, None]],
            pl.Array(pl.Date, shape=2),
        ),
        (
            [
                [timedelta(seconds=1), timedelta(seconds=10)],
                [None, None],
            ],
            pl.Array(pl.Duration, shape=2),
        ),
    ],
)
def test_write_read_json_array_logical_inner_type(
    data: Any, dtype: pl.DataType
) -> None:
    df = pl.DataFrame({"foo": data}, schema={"foo": dtype})
    buf = io.StringIO()
    df.serialize(buf)
    buf.seek(0)
    deserialized_df = pl.DataFrame.deserialize(buf)
    assert deserialized_df.dtypes == df.dtypes
    assert deserialized_df.to_dict(as_series=False) == df.to_dict(as_series=False)


def test_json_deserialize_empty_list_10458() -> None:
    schema = {"LIST_OF_STRINGS": pl.List(pl.String)}
    serialized_schema = pl.DataFrame(schema=schema).serialize()
    df = pl.DataFrame.deserialize(io.StringIO(serialized_schema))
    assert df.schema == schema


def test_serde_validation() -> None:
    f = io.StringIO(
        """
    {
      "columns": [
        {
          "name": "a",
          "datatype": "Int64",
          "values": [
            1,
            2
          ]
        },
        {
          "name": "b",
          "datatype": "Int64",
          "values": [
            1
          ]
        }
      ]
    }
    """
    )
    with pytest.raises(
        pl.ComputeError,
        match=r"lengths don't match",
    ):
        pl.DataFrame.deserialize(f)

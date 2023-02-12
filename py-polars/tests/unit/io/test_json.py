from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_frame_equal_local_categoricals


@pytest.mark.parametrize("buf", [io.BytesIO(), io.StringIO()])
def test_to_from_buffer(df: pl.DataFrame, buf: io.IOBase) -> None:
    df.write_json(buf)
    buf.seek(0)
    read_df = pl.read_json(buf)
    assert_frame_equal_local_categoricals(df, read_df)


def test_to_from_file(df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "small.json"
        df.write_json(file_path)
        out = pl.read_json(file_path)

    assert_frame_equal_local_categoricals(df, out)


def test_write_json_to_string() -> None:
    # Tests if it runs if no arg given
    df = pl.DataFrame({"a": [1, 2, 3]})
    expected_str = '{"columns":[{"name":"a","datatype":"Int64","values":[1,2,3]}]}'
    assert df.write_json() == expected_str


def test_write_json(df: pl.DataFrame) -> None:
    # Text-based conversion loses time info
    df = df.select(pl.all().exclude(["cat", "time"]))
    s = df.write_json()
    f = io.BytesIO()
    f.write(s.encode())
    f.seek(0)
    out = pl.read_json(f)
    assert_frame_equal(out, df)

    file = io.BytesIO()
    df.write_json(file)
    file.seek(0)
    out = pl.read_json(file)
    assert_frame_equal(out, df)


def test_write_json_row_oriented() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    out = df.write_json(row_oriented=True)
    assert out == '[{"a":1,"b":"a"},{"a":2,"b":"b"},{"a":3,"b":null}]'

    # Test round trip
    f = io.BytesIO()
    f.write(out.encode())
    f.seek(0)
    result = pl.read_json(f)
    assert_frame_equal(result, df)


def test_write_ndjson() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    out = df.write_ndjson()
    assert out == '{"a":1,"b":"a"}\n{"a":2,"b":"b"}\n{"a":3,"b":null}\n'

    # Test round trip
    f = io.BytesIO()
    f.write(out.encode())
    f.seek(0)
    result = pl.read_ndjson(f)
    assert_frame_equal(result, df)


def test_write_ndjson_with_trailing_newline() -> None:
    input = """{"Column1":"Value1"}\n"""
    df = pl.read_ndjson(io.StringIO(input))

    expected = pl.DataFrame({"Column1": ["Value1"]})
    assert_frame_equal(df, expected)


def test_read_ndjson_empty_array() -> None:
    assert pl.read_ndjson(io.StringIO("""{"foo": {"bar": []}}""")).to_dict(False) == {
        "foo": [{"": None}]
    }


def test_ndjson_nested_null() -> None:
    payload = """{"foo":{"bar":[{}]}}"""
    assert pl.read_ndjson(io.StringIO(payload)).to_dict(False) == {
        "foo": [{"bar": [{"": None}]}]
    }


def test_ndjson_nested_utf8_int() -> None:
    ndjson = """{"Accumulables":[{"Value":32395888},{"Value":"539454"}]}"""
    assert pl.read_ndjson(io.StringIO(ndjson)).to_dict(False) == {
        "Accumulables": [[{"Value": "32395888"}, {"Value": "539454"}]]
    }

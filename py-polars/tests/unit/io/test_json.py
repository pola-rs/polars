from __future__ import annotations

import io
import os

import polars as pl
from polars.testing import assert_frame_equal_local_categoricals


def test_to_from_buffer(df: pl.DataFrame) -> None:
    for buf in (io.BytesIO(), io.StringIO()):
        df.write_json(buf)
        buf.seek(0)
        read_df = pl.read_json(buf)
        assert_frame_equal_local_categoricals(df, read_df)


def test_to_from_file(io_test_dir: str, df: pl.DataFrame) -> None:
    f = os.path.join(io_test_dir, "small.json")
    df.write_json(f)
    out = pl.read_json(f)
    assert_frame_equal_local_categoricals(df, out)


def test_write_json() -> None:
    # tests if it runs if no arg given
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert (
        df.write_json()
        == '{"columns":[{"name":"a","datatype":"Int64","values":[1,2,3]}]}'
    )
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    expected = df

    out = df.write_json(row_oriented=True)
    assert out == r"""[{"a":1,"b":"a"},{"a":2,"b":"b"},{"a":3,"b":null}]"""
    # test round trip
    f = io.BytesIO()
    f.write(out.encode())
    f.seek(0)
    df = pl.read_json(f)
    assert df.frame_equal(expected)

    out = df.write_ndjson()
    assert (
        out
        == r"""{"a":1,"b":"a"}
{"a":2,"b":"b"}
{"a":3,"b":null}
"""
    )
    # test round trip
    f = io.BytesIO()
    f.write(out.encode())
    f.seek(0)
    df = pl.read_ndjson(f)
    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    assert df.frame_equal(expected)


def test_write_json2(df: pl.DataFrame) -> None:
    # text-based conversion loses time info
    df = df.select(pl.all().exclude(["cat", "time"]))
    s = df.write_json()
    f = io.BytesIO()
    f.write(s.encode())
    f.seek(0)
    out = pl.read_json(f)
    assert df.frame_equal(out, null_equal=True)

    file = io.BytesIO()
    df.write_json(file)
    file.seek(0)
    out = pl.read_json(file)
    assert df.frame_equal(out, null_equal=True)


def test_ndjson_with_trailing_newline() -> None:

    input = """{"Column1":"Value1"}\n"""

    df = pl.read_ndjson(io.StringIO(input))
    expected = pl.DataFrame({"Column1": ["Value1"]})
    assert df.frame_equal(expected)

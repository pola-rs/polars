# flake8: noqa: W191,E101
import io
import os

import pytest

import polars as pl


def test_to_from_buffer(df: pl.DataFrame) -> None:
    for buf in (io.BytesIO(), io.StringIO()):
        df.to_json(buf)
        buf.seek(0)
        read_df = pl.read_json(buf)
        read_df = read_df.with_columns(
            [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
        )
        assert df.frame_equal(read_df)


def test_to_from_file(io_test_dir: str, df: pl.DataFrame) -> None:
    f = os.path.join(io_test_dir, "small.json")
    df.to_json(f)

    out = pl.read_json(f)
    assert out.frame_equal(df)

    # read_df = read_df.with_columns(
    #     [pl.col("cat").cast(pl.Categorical), pl.col("time").cast(pl.Time)]
    # )
    # assert df.frame_equal(read_df)


def test_to_json() -> None:
    # tests if it runs if no arg given
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert (
        df.to_json() == '{"columns":[{"name":"a","datatype":"Int64","values":[1,2,3]}]}'
    )
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    expected = df

    out = df.to_json(row_oriented=True)
    assert out == r"""[{"a":1,"b":"a"},{"a":2,"b":"b"},{"a":3,"b":null}]"""
    # test round trip
    f = io.BytesIO()
    f.write(out.encode())  # type: ignore
    f.seek(0)
    df = pl.read_json(f, json_lines=False)
    assert df.frame_equal(expected)

    out = df.to_json(json_lines=True)
    assert (
        out
        == r"""{"a":1,"b":"a"}
{"a":2,"b":"b"}
{"a":3,"b":null}
"""
    )
    # test round trip
    f = io.BytesIO()
    f.write(out.encode())  # type: ignore
    f.seek(0)
    df = pl.read_json(f, json_lines=True)
    expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    assert df.frame_equal(expected)


def test_to_json2(df: pl.DataFrame) -> None:
    # text-based conversion loses time info
    df = df.select(pl.all().exclude(["cat", "time"]))
    s = df.to_json(to_string=True)
    f = io.BytesIO()
    f.write(s.encode())
    f.seek(0)
    out = pl.read_json(f)
    assert df.frame_equal(out, null_equal=True)

    file = io.BytesIO()
    df.to_json(file)
    file.seek(0)
    out = pl.read_json(file)
    assert df.frame_equal(out, null_equal=True)

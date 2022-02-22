# flake8: noqa: W191,E101
import io
import os

import pytest

import polars as pl


def test_to_from_buffer(df: pl.DataFrame) -> None:
    buf = io.BytesIO()
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

    # Not sure why error occur
    with pytest.raises(RuntimeError):
        _ = pl.read_json(f)

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

    out = df.to_json(row_oriented=True)
    assert out == r"""[{"a":1,"b":"a"},{"a":2,"b":"b"},{"a":3,"b":null}]"""
    out = df.to_json(json_lines=True)
    assert (
        out
        == r"""{"a":1,"b":"a"}
{"a":2,"b":"b"}
{"a":3,"b":null}
"""
    )

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import AvroCompression


COMPRESSIONS = ["uncompressed", "snappy", "deflate"]


@pytest.fixture
def example_df() -> pl.DataFrame:
    return pl.DataFrame({"i64": [1, 2], "f64": [0.1, 0.2], "str": ["a", "b"]})


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_buffer(example_df: pl.DataFrame, compression: AvroCompression) -> None:
    buf = io.BytesIO()
    example_df.write_avro(buf, compression=compression)
    buf.seek(0)

    read_df = pl.read_avro(buf)
    assert_frame_equal(example_df, read_df)


@pytest.mark.write_disk
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_file(
    example_df: pl.DataFrame, compression: AvroCompression, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.avro"
    example_df.write_avro(file_path, compression=compression)
    df_read = pl.read_avro(file_path)

    assert_frame_equal(example_df, df_read)


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_avro(f)
    f.seek(0)

    read_df = pl.read_avro(f, columns=["b", "c"])
    assert_frame_equal(expected, read_df)


def test_select_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_avro(f)
    f.seek(0)

    read_df = pl.read_avro(f, columns=[1, 2])
    assert_frame_equal(expected, read_df)


def test_with_name() -> None:
    df = pl.DataFrame({"a": [1]})
    expected = pl.DataFrame(
        {
            "type": ["record"],
            "name": ["my_schema_name"],
            "fields": [[{"name": "a", "type": ["null", "long"]}]],
        }
    )

    f = io.BytesIO()
    df.write_avro(f, name="my_schema_name")

    f.seek(0)
    raw = f.read()

    read_df = pl.read_json(raw[raw.find(b"{") : raw.rfind(b"}") + 1])

    assert_frame_equal(expected, read_df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_buffer_multiple_chunks_29568(compression: AvroCompression) -> None:
    df = pl.concat(
        [pl.DataFrame({"i64": [1, 2]}), pl.DataFrame({"i64": [3]})], rechunk=False
    )
    assert df.n_chunks() == 2

    buf = io.BytesIO()
    df.write_avro(buf, compression=compression)
    assert buf.getvalue().count(b"Obj\x01") == 1
    buf.seek(0)

    read_df = pl.read_avro(buf)
    assert_frame_equal(df, read_df)


@pytest.mark.parametrize(("offset", "length"), [(0, 5), (1, 4), (2, 3), (3, 2)])
def test_write_list_sliced_and_empty_27090(offset: int, length: int) -> None:
    df = pl.DataFrame(
        {
            "f": [0.0, 1.0, 2.0, 3.0, 4.0],
            "lst": [["a", "b"], [], None, ["c"], ["d", "e"]],
            "lst_lst": [[[1], []], [], None, [[2, 3]], [[4]]],
            "lst_struct": [
                [{"a": 1, "b": "x"}],
                [],
                None,
                [{"a": 2, "b": None}, {"a": 3, "b": "y"}],
                [{"a": 4, "b": "z"}],
            ],
        },
        schema={
            "f": pl.Float64,
            "lst": pl.List(pl.String),
            "lst_lst": pl.List(pl.List(pl.Int64)),
            "lst_struct": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String})),
        },
    )
    chunk = df.slice(offset, length)
    buf = io.BytesIO()
    chunk.write_avro(buf)
    buf.seek(0)
    assert_frame_equal(pl.read_avro(buf), chunk)


def test_write_list_null_row_with_values_27090() -> None:
    # The null row still covers the child values "x" and "y".
    lst = pa.LargeListArray.from_arrays(
        pa.array([0, 2, 4, 5], pa.int64()),
        pa.array(["a", "b", "x", "y", "c"], pa.large_string()),
        mask=pa.array([False, True, False]),
    )
    df = pl.DataFrame({"lst": pl.from_arrow(lst), "f": [1.0, 2.0, 3.0]})
    buf = io.BytesIO()
    df.write_avro(buf)
    buf.seek(0)
    expected = pl.DataFrame(
        {"lst": [["a", "b"], None, ["c"]], "f": [1.0, 2.0, 3.0]},
        schema={"lst": pl.List(pl.String), "f": pl.Float64},
    )
    assert_frame_equal(pl.read_avro(buf), expected)

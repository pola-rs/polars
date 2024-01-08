from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def foods_ndjson_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.ndjson"


def test_scan_ndjson(foods_ndjson_path: Path) -> None:
    df = pl.scan_ndjson(foods_ndjson_path, row_count_name="row_count").collect()
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_ndjson(foods_ndjson_path, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_ndjson(foods_ndjson_path, row_count_name="row_count")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_scan_ndjson_with_schema(foods_ndjson_path: Path) -> None:
    schema = {
        "category": pl.Categorical,
        "calories": pl.Int64,
        "fats_g": pl.Float64,
        "sugars_g": pl.Int64,
    }
    df = pl.scan_ndjson(foods_ndjson_path, schema=schema).collect()
    assert df["category"].dtype == pl.Categorical
    assert df["calories"].dtype == pl.Int64
    assert df["fats_g"].dtype == pl.Float64
    assert df["sugars_g"].dtype == pl.Int64

    schema["sugars_g"] = pl.Float64
    df = pl.scan_ndjson(foods_ndjson_path, schema=schema).collect()
    assert df["sugars_g"].dtype == pl.Float64


@pytest.mark.write_disk()
def test_scan_with_projection(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    json = r"""
{"text": "\"hello", "id": 1}
{"text": "\n{\n\t\t\"inner\": \"json\n}\n", "id": 10}
{"id": 0, "text":"\"","date":"2013-08-03 15:17:23"}
{"id": 1, "text":"\"123\"","date":"2009-05-19 21:07:53"}
{"id": 2, "text":"/....","date":"2009-05-19 21:07:53"}
{"id": 3, "text":"\n\n..","date":"2"}
{"id": 4, "text":"\"'/\n...","date":"2009-05-19 21:07:53"}
{"id": 5, "text":".h\"h1hh\\21hi1e2emm...","date":"2009-05-19 21:07:53"}
{"id": 6, "text":"xxxx....","date":"2009-05-19 21:07:53"}
{"id": 7, "text":".\"quoted text\".","date":"2009-05-19 21:07:53"}
"""
    json_bytes = bytes(json, "utf-8")

    file_path = tmp_path / "escape_chars.json"
    file_path.write_bytes(json_bytes)

    actual = pl.scan_ndjson(file_path).select(["id", "text"]).collect()

    expected = pl.DataFrame(
        {
            "id": [1, 10, 0, 1, 2, 3, 4, 5, 6, 7],
            "text": [
                '"hello',
                '\n{\n\t\t"inner": "json\n}\n',
                '"',
                '"123"',
                "/....",
                "\n\n..",
                "\"'/\n...",
                '.h"h1hh\\21hi1e2emm...',
                "xxxx....",
                '."quoted text".',
            ],
        }
    )
    assert_frame_equal(actual, expected)


def test_glob_n_rows(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.ndjson"
    df = pl.scan_ndjson(file_path, n_rows=40).collect()

    # 27 rows from foods1.ndjson and 13 from foods2.ndjson
    assert df.shape == (40, 4)

    # take first and last rows
    assert df[[0, 39]].to_dict(as_series=False) == {
        "category": ["vegetables", "seafood"],
        "calories": [45, 146],
        "fats_g": [0.5, 6.0],
        "sugars_g": [2, 2],
    }


# See #10661.
def test_json_no_unicode_truncate() -> None:
    assert pl.read_ndjson(rb'{"field": "\ufffd1234"}')[0, 0] == "\ufffd1234"


def test_ndjson_list_arg(io_files_path: Path) -> None:
    first = io_files_path / "foods1.ndjson"
    second = io_files_path / "foods2.ndjson"

    df = pl.scan_ndjson(source=[first, second]).collect()
    assert df.shape == (54, 4)
    assert df.row(-1) == ("seafood", 194, 12.0, 1)
    assert df.row(0) == ("vegetables", 45, 0.5, 2)


def test_anonymous_scan_explain(io_files_path: Path) -> None:
    file = io_files_path / "foods1.ndjson"
    q = pl.scan_ndjson(source=file)
    assert "Anonymous" in q.explain()
    assert "Anonymous" in q.show_graph(raw_output=True)  # type: ignore[operator]


def test_sink_ndjson_should_write_same_data(
    io_files_path: Path, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)
    # Arrange
    source_path = io_files_path / "foods1.csv"
    target_path = tmp_path / "foods_test.ndjson"
    expected = pl.read_csv(source_path)
    lf = pl.scan_csv(source_path)
    # Act
    lf.sink_ndjson(target_path)
    df = pl.read_ndjson(target_path)
    # Assert
    assert_frame_equal(df, expected)

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
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


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
    with open(file_path, "wb") as f:
        f.write(json_bytes)
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
    assert df[[0, 39]].to_dict(False) == {
        "category": ["vegetables", "seafood"],
        "calories": [45, 146],
        "fats_g": [0.5, 6.0],
        "sugars_g": [2, 2],
    }

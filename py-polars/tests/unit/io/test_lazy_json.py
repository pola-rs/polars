from __future__ import annotations

from os import path

import polars as pl


def test_scan_ndjson(foods_ndjson: str) -> None:
    df = pl.scan_ndjson(foods_ndjson, row_count_name="row_count").collect()
    assert df["row_count"].to_list() == list(range(27))

    df = (
        pl.scan_ndjson(foods_ndjson, row_count_name="row_count")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_count"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_ndjson(foods_ndjson, row_count_name="row_count")
        .with_row_count("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_scan_with_projection() -> None:
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
    file = path.join(path.dirname(__file__), "escape_chars.json")
    with open(file, "wb") as f:
        f.write(
            json_bytes,
        )
    actual = pl.scan_ndjson(file).select(["id", "text"]).collect()
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
    assert actual.frame_equal(expected, null_equal=True)

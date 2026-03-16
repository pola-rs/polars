from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import NdjsonCompression
    from tests.conftest import PlMonkeyPatch


@pytest.fixture
def foods_ndjson_path(io_files_path: Path) -> Path:
    return io_files_path / "foods1.ndjson"


def test_scan_ndjson(foods_ndjson_path: Path) -> None:
    df = pl.scan_ndjson(foods_ndjson_path, row_index_name="row_index").collect()
    assert df["row_index"].to_list() == list(range(27))

    df = (
        pl.scan_ndjson(foods_ndjson_path, row_index_name="row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))
        .collect()
    )

    assert df["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    df = (
        pl.scan_ndjson(foods_ndjson_path, row_index_name="row_index")
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


def test_scan_ndjson_infer_0(foods_ndjson_path: Path) -> None:
    with pytest.raises(ValueError):
        pl.scan_ndjson(foods_ndjson_path, infer_schema_length=0)


def test_scan_ndjson_batch_size_zero() -> None:
    with pytest.raises(ValueError, match="invalid zero value"):
        pl.scan_ndjson("test.ndjson", batch_size=0)


@pytest.mark.write_disk
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


def test_projection_pushdown_ndjson(io_files_path: Path) -> None:
    file_path = io_files_path / "foods1.ndjson"
    df = pl.scan_ndjson(file_path).select(pl.col.calories)

    explain = df.explain()

    assert "simple π" not in explain
    assert "PROJECT 1/4 COLUMNS" in explain

    assert_frame_equal(df.collect(optimizations=pl.QueryOptFlags.none()), df.collect())


def test_predicate_pushdown_ndjson(io_files_path: Path) -> None:
    file_path = io_files_path / "foods1.ndjson"
    df = pl.scan_ndjson(file_path).filter(pl.col.calories > 80)

    explain = df.explain()

    assert "FILTER" not in explain
    assert """SELECTION: [(col("calories")) > (80)]""" in explain

    assert_frame_equal(df.collect(optimizations=pl.QueryOptFlags.none()), df.collect())


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


def test_glob_single_scan(io_files_path: Path) -> None:
    file_path = io_files_path / "foods*.ndjson"
    df = pl.scan_ndjson(file_path, n_rows=40)

    explain = df.explain()

    assert explain.count("SCAN") == 1
    assert "UNION" not in explain


def test_scan_ndjson_empty_lines_in_middle() -> None:
    assert_frame_equal(
        pl.scan_ndjson(
            f"""\
{{"a": 1}}
{"              "}
{{"a": 2}}{"              "}
{"              "}
{{"a": 3}}
""".encode()
        ).collect(),
        pl.DataFrame({"a": [1, 2, 3]}),
    )


@pytest.mark.parametrize("row_index_offset", [None, 0, 20])
def test_scan_ndjson_slicing(
    foods_ndjson_path: Path, row_index_offset: int | None
) -> None:
    lf = pl.scan_ndjson(foods_ndjson_path)

    if row_index_offset is not None:
        lf = lf.with_row_index(offset=row_index_offset)

    for q in [
        lf.head(5),
        lf.tail(5),
        lf.head(0),
        lf.tail(0),
        lf.slice(-999, 3),
        lf.slice(999, 3),
        lf.slice(-999, 0),
        lf.slice(999, 0),
        lf.slice(-999),
        lf.slice(-3, 999),
    ]:
        assert_frame_equal(
            q.collect(), q.collect(optimizations=pl.QueryOptFlags.none())
        )


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Boolean,
        pl.Int32,
        pl.Int64,
        pl.UInt64,
        pl.UInt32,
        pl.Float32,
        pl.Float64,
        pl.Datetime,
        pl.Date,
        pl.Null,
    ],
)
def test_scan_ndjson_raises_on_parse_error(dtype: pl.DataType) -> None:
    buf = b"""\
{"a": "AAAA"}
"""

    cx = (
        pytest.raises(
            pl.exceptions.ComputeError,
            match="got non-null value for NULL-typed column: AAAA",
        )
        if str(dtype) == "Null"
        else pytest.raises(
            pl.exceptions.ComputeError,
            match=re.escape("cannot parse 'AAAA' (string) as "),
        )
    )

    with cx:
        pl.scan_ndjson(
            buf,
            schema={"a": dtype},
        ).collect()

    assert_frame_equal(
        pl.scan_ndjson(buf, schema={"a": dtype}, ignore_errors=True).collect(),
        pl.DataFrame({"a": [None]}, schema={"a": dtype}),
    )


def test_scan_ndjson_parse_string() -> None:
    assert_frame_equal(
        pl.scan_ndjson(
            b"""\
{"a": "123"}
""",
            schema={"a": pl.String},
        ).collect(),
        pl.DataFrame({"a": "123"}),
    )


def test_scan_ndjson_raises_on_parse_error_nested() -> None:
    buf = b"""\
{"a": {"b": "AAAA"}}
"""
    q = pl.scan_ndjson(
        buf,
        schema={"a": pl.Struct({"b": pl.Int64})},
    )

    with pytest.raises(pl.exceptions.ComputeError):
        q.collect()

    q = pl.scan_ndjson(
        buf, schema={"a": pl.Struct({"b": pl.Int64})}, ignore_errors=True
    )

    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": [{"b": None}]}, schema={"a": pl.Struct({"b": pl.Int64})}),
    )


def test_scan_ndjson_nested_as_string() -> None:
    buf = b"""\
{"a": {"x": 1}, "b": [1,2,3], "c": {"y": null}, "d": [{"k": "abc"}, {"j": "123"}, {"l": 7, "m": 8}]}
"""

    df = pl.scan_ndjson(
        buf,
        schema={"a": pl.String, "b": pl.String, "c": pl.String, "d": pl.String},
    ).collect()

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "a": '{"x": 1}',
                "b": "[1, 2, 3]",
                "c": '{"y": null}',
                "d": '[{"k": "abc"}, {"j": "123"}, {"l": 7, "m": 8}]',
            }
        ),
    )


def test_scan_ndjson_nested_as_string_escape_chars() -> None:
    buf = b"""\
{"a": {"x": "\\"\\r\\n"}}
"""

    df = pl.scan_ndjson(
        buf,
        schema={"a": pl.String},
    ).collect()

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "a": '{"x": "\\"\\r\\n"}',
            }
        ),
    )

    assert_frame_equal(
        pl.scan_ndjson(
            buf,
            schema={"a": pl.String},
        )
        .collect()
        .with_columns(pl.col("a").str.json_decode(pl.Struct({"x": pl.String}))),
        pl.scan_ndjson(buf, schema={"a": pl.Struct({"x": pl.String})}).collect(),
    )


def test_scan_ndjson_schema_overwrite_22514() -> None:
    buf = b"""\
{"a": 1}
"""

    q = pl.scan_ndjson(buf)

    # Baseline: Infers as Int64
    assert q.collect_schema() == {"a": pl.Int64}
    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1}))

    q = pl.scan_ndjson(buf, schema_overrides={"a": pl.String})
    assert q.collect_schema() == {"a": pl.String}
    assert_frame_equal(q.collect(), pl.DataFrame({"a": "1"}))


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.parametrize("compression", ["uncompressed", "zstd", "gzip"])
def test_scan_ndjson_progressive_infer_schema_length(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path, compression: NdjsonCompression
) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "tt.ndjson"

    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    n_rows = 60_000
    df = (
        pl.DataFrame()
        .with_columns(pl.int_range(n_rows).alias("a"))
        .with_columns(pl.col.a.cast(pl.String).alias("b"))
    )

    df.lazy().sink_ndjson(file_path, compression=compression, check_extension=False)

    for infer_len in [1, 2, 100, n_rows - 1, n_rows, n_rows + 1, None]:
        out = pl.scan_ndjson(file_path, infer_schema_length=infer_len).collect()
        assert df.schema == out.schema

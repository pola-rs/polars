from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


def _read_csv_sql(path: Path, options: str) -> pl.DataFrame:
    args = f"'{path.as_posix()}'" + (f", {options}" if options else "")
    with pl.SQLContext() as ctx:
        return ctx.execute(f"SELECT * FROM read_csv({args})", eager=True)


_WIDE_ROW_LATE = "".join(f"{i},b\n" for i in range(1, 151)) + "151,b,late\n"


@pytest.mark.parametrize(
    ("contents", "options", "kwargs"),
    [
        # rows of different widths without a header (the issue's case)
        ("1,a\n2,b,extra\n3\n", "has_header => false", {"has_header": False}),
        (
            _WIDE_ROW_LATE,
            "has_header => false, infer_schema_length => NULL",
            {"has_header": False, "infer_schema_length": None},
        ),
        # layout
        (
            "a;b\n1,5;2\n",
            "separator => ';', decimal_comma => true",
            {"separator": ";", "decimal_comma": True},
        ),
        ("a\tb\n1\t2\n", "separator => E'\\t'", {"separator": "\t"}),
        ('a,b\n1,"x"\n', "quote_char => NULL", {"quote_char": None}),
        ('a,b\n1,"x"\n', "quote_char => ''", {"quote_char": ""}),
        ("a,b\n# note\n1,2\n", "comment_prefix => '#'", {"comment_prefix": "#"}),
        (
            "junk\njunk\na,b\n1,2\n3,4\n",
            "skip_rows => 2, skip_rows_after_header => 1",
            {"skip_rows": 2, "skip_rows_after_header": 1},
        ),
        ('title "open\na,b\n1,2\n', "skip_lines := 1", {"skip_lines": 1}),
        # names and types
        ("a,b\n1,2\n", "new_columns => ['x', 'y']", {"new_columns": ["x", "y"]}),
        (
            "1,2\n3,4\n",
            "has_header => false, schema => {'x': 'BIGINT', 'y': 'VARCHAR'}",
            {"has_header": False, "schema": {"x": pl.Int64, "y": pl.String}},
        ),
        (
            "a,b\n1,2\n",
            "schema_overrides => {'b': 'VARCHAR'}",
            {"schema_overrides": {"b": pl.String}},
        ),
        (
            "a,b\n1,2\n",
            "schema_overrides => ['VARCHAR', 'DOUBLE']",
            {"schema_overrides": [pl.String, pl.Float64]},
        ),
        ("a,b\n1,2\n", "infer_schema => false", {"infer_schema": False}),
        ("d\n2026-01-01\n", "", {}),
        ("d\n2026-01-01\n", "try_parse_dates => false", {"try_parse_dates": False}),
        # values and errors
        ("a,b\nNA,1\n", "null_values => 'NA'", {"null_values": "NA"}),
        ("a,b\nNA,-\n", "null_values => ['NA', '-']", {"null_values": ["NA", "-"]}),
        ("a,b\nNA,NA\n", "null_values => {'a': 'NA'}", {"null_values": {"a": "NA"}}),
        (
            "a\n1\nx\n",
            "schema_overrides => {'a': 'INT'}, ignore_errors => true",
            {"schema_overrides": {"a": pl.Int32}, "ignore_errors": True},
        ),
        # ragged rows, file paths, empty files
        (
            "a,b\n1,2,3\n",
            "truncate_ragged_lines => true",
            {"truncate_ragged_lines": True},
        ),
        ("a\n1\n", "include_file_paths => 'file'", {"include_file_paths": "file"}),
        (
            "",
            "has_header => false, schema => {'x': 'INT'}",
            {"has_header": False, "schema": {"x": pl.Int32}},
        ),
    ],
)
def test_read_csv_options_match_scan_csv(
    tmp_path: Path, contents: str, options: str, kwargs: dict[str, Any]
) -> None:
    path = tmp_path / "data.csv"
    path.write_text(contents)

    # SQL keeps `try_parse_dates=True` as its default; the same path spelling as in the
    # query keeps `include_file_paths` identical on every platform
    scan_kwargs: dict[str, Any] = {"try_parse_dates": True, **kwargs}
    expected = pl.scan_csv(path.as_posix(), **scan_kwargs).collect()
    assert_frame_equal(_read_csv_sql(path, options), expected)


def test_read_csv_options_lossy_encoding(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"a\n\xff\n")

    expected = pl.scan_csv(path.as_posix(), encoding="utf8-lossy").collect()
    assert_frame_equal(_read_csv_sql(path, "encoding => 'utf8-lossy'"), expected)
    assert expected["a"][0] == "�"


def test_read_csv_options_survive_plan_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("1;x\n2;y\n")

    q = pl.sql(
        f"SELECT column_1 FROM read_csv('{path.as_posix()}', "
        "has_header => false, separator => ';') WHERE column_0 > 1"
    )
    roundtripped = pl.LazyFrame.deserialize(q.serialize())
    assert_frame_equal(roundtripped.collect(), pl.DataFrame({"column_1": ["y"]}))


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        (
            "header => false",
            SQLInterfaceError,
            r"unknown `read_csv` option 'header'; valid options are: has_header, ",
        ),
        (
            "n_rows => 5",
            SQLInterfaceError,
            r"option 'n_rows' is not supported in SQL: use LIMIT",
        ),
        (
            "missing_columns => 'insert'",
            SQLInterfaceError,
            r"is not supported in SQL: it is unstable; .* infer_schema_length => NULL",
        ),
        (
            "has_header = false",
            SQLSyntaxError,
            r"options must be named with '=>' \(found: has_header = false\)",
        ),
        (
            "has_header => false, has_header => true",
            SQLSyntaxError,
            r"option 'has_header' is given more than once",
        ),
        (
            "has_header => *",
            SQLSyntaxError,
            r"option 'has_header' expects a value; found: \*",
        ),
        (
            "has_header => 'no'",
            SQLSyntaxError,
            r"option 'has_header' expects a boolean; found: 'no'",
        ),
        (
            "separator => '\\t'",
            SQLSyntaxError,
            r"expects a single-byte character; .* E'' string",
        ),
        (
            "schema => {'x': 'INT('}",
            SQLSyntaxError,
            r"option 'schema' has an invalid type name: 'INT\('",
        ),
    ],
)
def test_read_csv_option_errors(
    tmp_path: Path, options: str, error: type[Exception], match: str
) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(error, match=match):
        _read_csv_sql(path, options)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import (
    ComputeError,
    SchemaFieldNotFoundError,
    SQLInterfaceError,
    SQLSyntaxError,
)
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


def _read_csv_sql(path: Path, options: str, path_prefix: str = "") -> pl.DataFrame:
    args = f"{path_prefix}'{path.as_posix()}'" + (f", {options}" if options else "")
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
        # early rows narrower than the schema (#29970)
        (
            "C,NEMP.WORLD,PUBLIC_DAILY\nD,DREGION,1,NSW1\nD,DUNIT,1,UNIT_A,10.5,1\n",
            "has_header => false, skip_rows => 1, missing_columns => 'insert', schema => {"
            + ", ".join(f"'column_{i}': 'VARCHAR'" for i in range(6))
            + "}",
            {
                "has_header": False,
                "skip_rows": 1,
                "missing_columns": "insert",
                "schema": {f"column_{i}": pl.String for i in range(6)},
            },
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
        # option names are case-insensitive
        ('title "open\na,b\n1,2\n', "SKIP_LINES := 1", {"skip_lines": 1}),
        # names and types
        ("a,b\n1,2\n", "new_columns => ['x', 'y']", {"new_columns": ["x", "y"]}),
        (
            "1,2\n3,4\n",
            "has_header => false, schema => {'x': 'BIGINT', 'y': 'VARCHAR'}",
            {"has_header": False, "schema": {"x": pl.Int64, "y": pl.String}},
        ),
        (
            "a,b,c\n1,x,1.5\n",
            "schema => {'a': 'BIGINT', 'b': 'VARCHAR'}, extra_columns => 'ignore'",
            {"schema": {"a": pl.Int64, "b": pl.String}, "extra_columns": "ignore"},
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
        # implies `truncate_ragged_lines`; values are case-insensitive
        ("a,b\n1,2,3\n4,5\n", "extra_columns => 'IGNORE'", {"extra_columns": "ignore"}),
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


@pytest.mark.parametrize(
    ("files", "options", "kwargs"),
    [
        # different headers: the union of the columns, or those of the first file
        (
            ["a,b\n1,x\n", "a,b,c\n2,y,1.5\n"],
            "missing_columns => 'insert'",
            {"missing_columns": "insert"},
        ),
        (
            ["a,b\n1,x\n", "a,b,c\n2,y,1.5\n"],
            "extra_columns => 'ignore', infer_schema_files => 1",
            {"extra_columns": "ignore", "infer_schema_files": 1},
        ),
        # the path of each file
        (
            ["a\n1\n", "a\n2\n"],
            "include_file_paths => 'file'",
            {"include_file_paths": "file"},
        ),
        # a wider type in the eleventh file, past the default of ten
        (
            ["a\n1.5\n" if i == 10 else "a\n1\n" for i in range(12)],
            "infer_schema_files => 11",
            {"infer_schema_files": 11},
        ),
    ],
)
def test_read_csv_options_across_files_match_scan_csv(
    tmp_path: Path, files: list[str], options: str, kwargs: dict[str, Any]
) -> None:
    for i, contents in enumerate(files):
        (tmp_path / f"part_{i:02}.csv").write_text(contents)
    glob = tmp_path / "part_*.csv"

    scan_kwargs: dict[str, Any] = {"try_parse_dates": True, **kwargs}
    expected = pl.scan_csv(glob.as_posix(), **scan_kwargs).collect()
    assert_frame_equal(_read_csv_sql(glob, options), expected)


def test_read_csv_options_lossy_encoding(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"a\n\xff\n")

    expected = pl.scan_csv(path.as_posix(), encoding="utf8-lossy").collect()
    assert_frame_equal(_read_csv_sql(path, "encoding => 'utf8-lossy'"), expected)
    assert expected["a"][0] == "�"


def test_read_csv_path_as_escaped_string(tmp_path: Path) -> None:
    path = tmp_path / "data.tsv"
    path.write_text("a\tb\n1\t2\n")

    # forward slashes keep the path free of escape sequences
    expected = pl.scan_csv(path.as_posix(), separator="\t").collect()
    assert_frame_equal(
        _read_csv_sql(path, "separator => E'\\t'", path_prefix="E"), expected
    )


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
            "credential_provider => 'auto'",
            SQLInterfaceError,
            r"is not supported in SQL: cloud credentials are read from the environment",
        ),
        (
            "missing_columns => 'fill'",
            SQLSyntaxError,
            r"option 'missing_columns' expects 'insert' or 'raise'; found: 'fill'",
        ),
        # Python's `None` only means "the default" here, so NULL is rejected, as for
        # `truncate_ragged_lines`; accepting it later would not break anything
        (
            "missing_columns => NULL",
            SQLSyntaxError,
            r"option 'missing_columns' expects 'insert' or 'raise'; found: NULL",
        ),
        (
            "extra_columns => 'ignore', truncate_ragged_lines => false",
            SQLSyntaxError,
            r"cannot set truncate_ragged_lines => false with extra_columns => 'ignore'",
        ),
        (
            "infer_schema_files => 0",
            SQLSyntaxError,
            r"option 'infer_schema_files' expects a positive integer; found: 0",
        ),
        # an excluded option without an alternative gets no trailing reason
        (
            "glob => false",
            SQLInterfaceError,
            r"option 'glob' is not supported in SQL$",
        ),
        (
            "columns => ['a']",
            SQLInterfaceError,
            r"option 'columns' is not supported in SQL: select the columns in the SELECT "
            r"list; to set their types, use schema",
        ),
        (
            "has_header = false",
            SQLSyntaxError,
            r"options must be named with '=>' \(found: has_header = false\)",
        ),
        (
            "false",
            SQLSyntaxError,
            r"options must be named with '=>' \(found: false\)",
        ),
        (
            "has_header => false, HAS_HEADER => true",
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
            "skip_rows => -1",
            SQLSyntaxError,
            r"option 'skip_rows' expects a non-negative integer; found: -1",
        ),
        (
            "encoding => 'latin1'",
            SQLSyntaxError,
            r"option 'encoding' expects 'utf8' or 'utf8-lossy'; found: 'latin1'",
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
        (
            "schema => ['INT']",
            SQLSyntaxError,
            r"option 'schema' expects a dictionary such as \{'a': 'INT'\}; found: \['INT'\]",
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


@pytest.mark.parametrize("args", ["", "has_header => false"])
def test_read_csv_expects_a_path_first(args: str) -> None:
    with pytest.raises(
        SQLSyntaxError, match=r"`read_csv` expects a file path as its first argument"
    ):
        pl.sql(f"SELECT * FROM read_csv({args})").collect()


def test_read_csv_override_that_does_not_fit(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a\n1\nx\n")
    with pytest.raises(ComputeError, match=r"could not parse"):
        _read_csv_sql(path, "schema_overrides => {'a': 'INT'}")


def _read_ndjson_sql(
    path: Path, options: str, function: str = "read_ndjson"
) -> pl.DataFrame:
    args = f"'{path.as_posix()}'" + (f", {options}" if options else "")
    with pl.SQLContext() as ctx:
        return ctx.execute(f"SELECT * FROM {function}({args})", eager=True)


_NDJSON_ROWS = (
    '{"a": 1, "b": "x", "s": {"x": 1}, "l": [1, 2], "d": "2026-01-02"}\n'
    '{"a": "oops", "b": "y", "s": {"x": 2}, "l": [3], "d": "2026-01-03"}\n'
)
_NDJSON_KEY_LATE = (
    "".join(f'{{"a": {i}}}\n' for i in range(150)) + '{"a": 150, "b": "late"}\n'
)


@pytest.mark.parametrize(
    ("contents", "options", "kwargs"),
    [
        (_NDJSON_ROWS, "", {}),
        # a key first seen after the inference window is dropped, unless inference
        # covers it
        (_NDJSON_KEY_LATE, "", {}),
        (
            _NDJSON_KEY_LATE,
            "infer_schema_length => NULL",
            {"infer_schema_length": None},
        ),
        # option names are case-insensitive, and `:=` works like `=>`
        (
            _NDJSON_KEY_LATE,
            "INFER_SCHEMA_LENGTH := 200",
            {"infer_schema_length": 200},
        ),
        (
            _NDJSON_ROWS,
            "schema => {'b': 'VARCHAR', 'c': 'INT'}",
            {"schema": {"b": pl.String, "c": pl.Int32}},
        ),
        (
            _NDJSON_ROWS,
            "schema_overrides => {'s': 'VARCHAR', 'l': 'INT[]', 'd': 'DATE'}",
            {
                "schema_overrides": {
                    "s": pl.String,
                    "l": pl.List(pl.Int32),
                    "d": pl.Date,
                }
            },
        ),
        (
            _NDJSON_ROWS,
            "schema_overrides => {'a': 'INT'}, ignore_errors => true",
            {"schema_overrides": {"a": pl.Int32}, "ignore_errors": True},
        ),
        (
            _NDJSON_ROWS,
            "include_file_paths => 'path'",
            {"include_file_paths": "path"},
        ),
    ],
)
def test_read_ndjson_options_match_scan_ndjson(
    tmp_path: Path, contents: str, options: str, kwargs: dict[str, Any]
) -> None:
    path = tmp_path / "data.ndjson"
    path.write_text(contents)

    # the same path spelling as in the query keeps `include_file_paths` identical
    expected = pl.scan_ndjson(path.as_posix(), **kwargs).collect()
    assert_frame_equal(_read_ndjson_sql(path, options), expected)


def test_read_ndjson_include_file_paths_over_a_glob(tmp_path: Path) -> None:
    (tmp_path / "part_1.ndjson").write_text('{"a": 1}\n')
    (tmp_path / "part_2.ndjson").write_text('{"a": 2}\n')
    glob = tmp_path / "part_*.ndjson"

    expected = pl.scan_ndjson(glob.as_posix(), include_file_paths="path").collect()
    assert_frame_equal(_read_ndjson_sql(glob, "include_file_paths => 'path'"), expected)
    assert expected["path"].n_unique() == 2


def test_read_json_takes_only_a_path(tmp_path: Path) -> None:
    path = tmp_path / "data.ndjson"
    path.write_text(_NDJSON_ROWS)

    assert_frame_equal(
        _read_ndjson_sql(path, "", function="read_json"),
        _read_ndjson_sql(path, ""),
    )
    with pytest.raises(
        SQLSyntaxError,
        match=r"`read_json` expects one file path or a list of file paths; found 2 arguments \(for options, use `read_ndjson`\)",
    ):
        _read_ndjson_sql(path, "ignore_errors => true", function="read_json")


def test_read_ndjson_options_survive_plan_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data.ndjson"
    path.write_text(_NDJSON_KEY_LATE)

    q = pl.sql(
        f"SELECT b FROM read_ndjson('{path.as_posix()}', "
        "infer_schema_length => NULL) WHERE b IS NOT NULL"
    )
    roundtripped = pl.LazyFrame.deserialize(q.serialize())
    assert_frame_equal(roundtripped.collect(), pl.DataFrame({"b": ["late"]}))


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        (
            "infer_schema_length => 0",
            SQLSyntaxError,
            r"option 'infer_schema_length' expects a positive integer or NULL; found: 0",
        ),
        (
            "header => true",
            SQLInterfaceError,
            r"unknown `read_ndjson` option 'header'; valid options are: schema, ",
        ),
        (
            "n_rows => 5",
            SQLInterfaceError,
            r"option 'n_rows' is not supported in SQL: use LIMIT",
        ),
        (
            "storage_options => {'a': 'b'}",
            SQLInterfaceError,
            r"not supported in SQL: cloud credentials are read from the environment",
        ),
        # an excluded option without an alternative gets no trailing reason
        (
            "batch_size => 10",
            SQLInterfaceError,
            r"option 'batch_size' is not supported in SQL$",
        ),
        (
            "ignore_errors = true",
            SQLSyntaxError,
            r"options must be named with '=>' \(found: ignore_errors = true\)",
        ),
        (
            "ignore_errors => 'no'",
            SQLSyntaxError,
            r"option 'ignore_errors' expects a boolean; found: 'no'",
        ),
        (
            "ignore_errors => true, IGNORE_ERRORS => false",
            SQLSyntaxError,
            r"option 'ignore_errors' is given more than once",
        ),
        (
            "schema => {'x': 'INT('}",
            SQLSyntaxError,
            r"option 'schema' has an invalid type name: 'INT\('",
        ),
        # an override for a key the file doesn't have names it
        (
            "schema_overrides => {'zz': 'INT'}",
            SchemaFieldNotFoundError,
            r"zz",
        ),
        # a value that doesn't fit an override fails, unless `ignore_errors` is set
        (
            "schema_overrides => {'a': 'INT'}",
            ComputeError,
            r"cannot parse 'oops' \(string\) as Int32",
        ),
    ],
)
def test_read_ndjson_option_errors(
    tmp_path: Path, options: str, error: type[Exception], match: str
) -> None:
    path = tmp_path / "data.ndjson"
    path.write_text(_NDJSON_ROWS)
    with pytest.raises(error, match=match):
        _read_ndjson_sql(path, options)


@pytest.mark.parametrize(
    ("function", "ext", "write", "scan"),
    [
        ("read_csv", "csv", pl.DataFrame.write_csv, pl.scan_csv),
        ("read_parquet", "parquet", pl.DataFrame.write_parquet, pl.scan_parquet),
        ("read_ipc", "ipc", pl.DataFrame.write_ipc, pl.scan_ipc),
        ("read_ndjson", "ndjson", pl.DataFrame.write_ndjson, pl.scan_ndjson),
        ("read_json", "ndjson", pl.DataFrame.write_ndjson, pl.scan_ndjson),
    ],
)
def test_read_list_of_paths(
    tmp_path: Path, function: str, ext: str, write: Any, scan: Any
) -> None:
    paths = [(tmp_path / f"{name}.{ext}").as_posix() for name in "abc"]
    for i, path in enumerate(paths):
        write(pl.DataFrame({"x": [i]}), path)
    # a glob inside the list is expanded, as in Python
    chosen = [paths[2], (tmp_path / f"a*.{ext}").as_posix()]

    sql_list = ", ".join(f"'{p}'" for p in chosen)
    result = pl.sql(f"SELECT * FROM {function}([{sql_list}])").collect()
    assert_frame_equal(result, scan(chosen).collect())
    assert result["x"].to_list() == [2, 0]

    # a list names the relation after the function
    result = pl.sql(f"SELECT {function}.x FROM {function}([{sql_list}])").collect()
    assert result["x"].to_list() == [2, 0]


def test_read_csv_list_of_paths_with_options(tmp_path: Path) -> None:
    # rows narrower than the schema, in files read by list (#29969)
    files = [tmp_path / f"{name}.csv" for name in "abc"]
    for file, contents in zip(files, ["1\n", "2,x\n", "3,y,z\n"], strict=True):
        file.write_text(contents)
    paths = [file.as_posix() for file in files]
    schema = {f"column_{i}": pl.String for i in range(3)}

    sql_schema = ", ".join(f"'{name}': 'VARCHAR'" for name in schema)
    result = pl.sql(
        f"SELECT * FROM read_csv(['{paths[0]}', '{paths[1]}'], has_header => false, "
        f"schema => {{{sql_schema}}}, missing_columns => 'insert', "
        "include_file_paths => 'file')"
    ).collect()
    expected = pl.scan_csv(
        paths[:2],
        has_header=False,
        schema=schema,
        missing_columns="insert",
        include_file_paths="file",
    ).collect()
    assert_frame_equal(result, expected)
    assert result["file"].to_list() == paths[:2]


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ("['a.csv', 1]", r"a list of them; found: \['a.csv', 1\]"),
        ("1", r"a list of them; found: 1"),
    ],
)
def test_read_csv_list_of_paths_errors(args: str, match: str) -> None:
    with pytest.raises(SQLSyntaxError, match=match):
        pl.sql(f"SELECT * FROM read_csv({args})").collect()


def test_read_csv_empty_list_of_paths() -> None:
    # as in Python: an empty list needs a schema
    result = pl.sql("SELECT * FROM read_csv([], schema => {'x': 'INT'})").collect()
    assert_frame_equal(result, pl.DataFrame(schema={"x": pl.Int32}))
    with pytest.raises(ComputeError, match=r"empty input"):
        pl.sql("SELECT * FROM read_csv([])").collect()


@pytest.mark.parametrize(
    ("function", "write"),
    [
        ("read_parquet", pl.DataFrame.write_parquet),
        ("read_ipc", pl.DataFrame.write_ipc),
    ],
)
def test_read_hive_partitions_list_of_paths(
    tmp_path: Path, function: str, write: Any
) -> None:
    for part in (1, 2):
        (tmp_path / f"p={part}").mkdir()
        write(pl.DataFrame({"x": [part]}), tmp_path / f"p={part}" / "data")
    files = [(tmp_path / f"p={part}" / "data").as_posix() for part in (1, 2)]

    # as in Python: a list of files doesn't read the partitions, one path does
    sql_list = ", ".join(f"'{f}'" for f in files)
    result = pl.sql(f"SELECT * FROM {function}([{sql_list}])").collect()
    assert result.columns == ["x"]
    result = pl.sql(f"SELECT * FROM {function}('{files[0]}')").collect()
    assert result.columns == ["x", "p"]

    # a single directory in a list does
    result = pl.sql(f"SELECT * FROM {function}(['{tmp_path.as_posix()}/'])").collect()
    assert_frame_equal(
        result, pl.DataFrame({"x": [1, 2], "p": [1, 2]}), check_row_order=False
    )


def _read_parquet_sql(path: Path, options: str) -> pl.DataFrame:
    args = f"'{path.as_posix()}'" + (f", {options}" if options else "")
    with pl.SQLContext() as ctx:
        return ctx.execute(f"SELECT * FROM read_parquet({args})", eager=True)


def _write_hive_parquet(root: Path) -> Path:
    # two hive partitions with date-like values; returns a glob over them
    for d in ("2026-01-01", "2026-01-02"):
        (root / f"d={d}").mkdir()
        pl.DataFrame({"a": [1, 2]}).write_parquet(root / f"d={d}" / "data.parquet")
    return root / "*" / "data.parquet"


@pytest.mark.parametrize(
    ("options", "kwargs"),
    [
        # unlike scan_parquet, hive partitions are read by default, for a glob too
        ("", {}),
        ("hive_partitioning => false", {"hive_partitioning": False}),
        ("try_parse_hive_dates => false", {"try_parse_hive_dates": False}),
        ("hive_schema => {'d': 'VARCHAR'}", {"hive_schema": {"d": pl.String}}),
        ("include_file_paths => 'file'", {"include_file_paths": "file"}),
    ],
)
def test_read_parquet_hive_options_match_scan_parquet(
    tmp_path: Path, options: str, kwargs: dict[str, Any]
) -> None:
    glob = _write_hive_parquet(tmp_path)
    scan_kwargs: dict[str, Any] = {"hive_partitioning": True, **kwargs}
    expected = pl.scan_parquet(glob.as_posix(), **scan_kwargs).collect()
    assert_frame_equal(_read_parquet_sql(glob, options), expected)


def test_read_parquet_hive_column_of_a_single_file(tmp_path: Path) -> None:
    # unlike scan_parquet's automatic choice, a single file in a hive folder gets it too
    path = tmp_path / "d=2026-01-01" / "data.parquet"
    path.parent.mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(path)

    assert _read_parquet_sql(path, "").schema == {"a": pl.Int64, "d": pl.Date}
    assert _read_parquet_sql(path, "hive_partitioning => false").schema == {
        "a": pl.Int64
    }


@pytest.mark.parametrize(
    ("options", "kwargs"),
    [
        # the first file defines the columns; later files lack one, or have another
        (
            "missing_columns => 'insert', extra_columns => 'ignore'",
            {"missing_columns": "insert", "extra_columns": "ignore"},
        ),
        # the union of the columns, from a schema; values are case-insensitive
        (
            "schema => {'a': 'BIGINT', 'b': 'VARCHAR', 'c': 'DOUBLE'}, "
            "missing_columns => 'INSERT'",
            {
                "schema": {"a": pl.Int64, "b": pl.String, "c": pl.Float64},
                "missing_columns": "insert",
            },
        ),
    ],
)
def test_read_parquet_column_options_match_scan_parquet(
    tmp_path: Path, options: str, kwargs: dict[str, Any]
) -> None:
    pl.DataFrame({"a": [1], "b": ["x"]}).write_parquet(tmp_path / "part_1.parquet")
    pl.DataFrame({"a": [2]}).write_parquet(tmp_path / "part_2.parquet")
    pl.DataFrame({"a": [3], "b": ["z"], "c": [1.5]}).write_parquet(
        tmp_path / "part_3.parquet"
    )
    glob = tmp_path / "part_*.parquet"

    scan_kwargs: dict[str, Any] = {"hive_partitioning": True, **kwargs}
    expected = pl.scan_parquet(glob.as_posix(), **scan_kwargs).collect()
    assert_frame_equal(_read_parquet_sql(glob, options), expected)


def test_read_parquet_options_survive_plan_roundtrip(tmp_path: Path) -> None:
    glob = _write_hive_parquet(tmp_path)
    q = pl.sql(
        f"SELECT * FROM read_parquet('{glob.as_posix()}', hive_partitioning => false) "
        "WHERE a > 1"
    )
    roundtripped = pl.LazyFrame.deserialize(q.serialize())
    assert_frame_equal(roundtripped.collect(), pl.DataFrame({"a": [2, 2]}))


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        # true or false only: NULL would not mean scan_parquet's automatic choice
        (
            "hive_partitioning => NULL",
            SQLSyntaxError,
            r"option 'hive_partitioning' expects a boolean; found: NULL",
        ),
        (
            "missing_columns => 'fill'",
            SQLSyntaxError,
            r"option 'missing_columns' expects 'insert' or 'raise'; found: 'fill'",
        ),
        (
            "header => true",
            SQLInterfaceError,
            r"unknown `read_parquet` option 'header'; valid options are: schema, ",
        ),
        (
            "n_rows => 5",
            SQLInterfaceError,
            r"option 'n_rows' is not supported in SQL: use LIMIT",
        ),
        (
            "columns => ['a']",
            SQLInterfaceError,
            r"option 'columns' is not supported in SQL: select the columns in the SELECT",
        ),
        (
            "cast_options => NULL",
            SQLInterfaceError,
            r"option 'cast_options' is not supported in SQL$",
        ),
    ],
)
def test_read_parquet_option_errors(
    tmp_path: Path, options: str, error: type[Exception], match: str
) -> None:
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)
    with pytest.raises(error, match=match):
        _read_parquet_sql(path, options)

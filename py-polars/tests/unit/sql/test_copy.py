from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1, 2, None], "s": ["a", "b,c|d", None]})


@pytest.mark.parametrize(
    ("ext", "read"),
    [
        ("csv", pl.read_csv),
        ("parquet", pl.read_parquet),
        ("ipc", pl.read_ipc),
        ("arrow", pl.read_ipc),
        ("feather", pl.read_ipc),
        ("ndjson", pl.read_ndjson),
        ("jsonl", pl.read_ndjson),
    ],
)
def test_copy_query_to_file(
    tmp_path: Path, df: pl.DataFrame, ext: str, read: Any
) -> None:
    path = tmp_path / f"out.{ext}"
    with pl.SQLContext(t=df) as ctx:
        # the file is written once the plan is collected
        plan = ctx.execute(f"COPY (SELECT * FROM t WHERE x > 0) TO '{path.as_posix()}'")
        assert not path.exists()
        assert plan.collect().shape == (0, 0)
        assert_frame_equal(read(path), df.head(2))

        # by the name of the format, whatever the file is called
        other = tmp_path / "out.dat"
        ctx.execute(f"COPY t TO '{other.as_posix()}' (FORMAT {ext})", eager=True)
        assert_frame_equal(read(other), df)


def test_copy_table_to_csv(tmp_path: Path, df: pl.DataFrame) -> None:
    path = tmp_path / "out.csv"
    path.write_text("an existing file is replaced\n")
    with pl.SQLContext(t=df) as ctx:
        ctx.execute(f"COPY t TO '{path.as_posix()}'", eager=True)
        assert path.read_text() == 'x,s\n1,a\n2,"b,c|d"\n,\n'
        # the file reads back by its path
        res = ctx.execute(f"SELECT * FROM '{path.as_posix()}'", eager=True)
        assert_frame_equal(res, df)

        ctx.execute(
            f"COPY t (s, x) TO '{path.as_posix()}' "
            "WITH (HEADER false, DELIMITER '|', NULL 'NA', QUOTE '''')",
            eager=True,
        )
        assert path.read_text() == "a|1\n'b,c|d'|2\nNA|NA\n"


def test_copy_frame_in_scope(tmp_path: Path, df: pl.DataFrame) -> None:
    path = tmp_path / "out.parquet"
    assert pl.sql(f"COPY df TO '{path.as_posix()}'").collect().shape == (0, 0)
    assert_frame_equal(pl.read_parquet(path), df)

    assert df.sql(f"COPY self (s) TO '{path.as_posix()}'").shape == (0, 0)
    assert_frame_equal(pl.read_parquet(path), df.select("s"))


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("COPY t FROM STDIN", "COPY FROM STDIN is not supported"),
        ("COPY t FROM '{}' (FORMAT csv)", "COPY FROM reads one file"),
        ("COPY t FROM '{}/in.parquet' (HEADER)", "HEADER applies to the CSV format"),
        ("COPY t TO STDOUT", "COPY TO STDOUT is not supported"),
        ("COPY t TO '{}/out.txt'", r"cannot tell the format of .*out\.txt.*csv"),
        ("COPY t TO '{}/out.json'", "cannot tell the format"),
        ("COPY t TO '{}/out.csv' (ESCAPE '!')", "COPY option ESCAPE is not supported"),
        ("COPY t TO '{}/out.parquet' (HEADER)", "HEADER applies to the CSV format"),
    ],
)
def test_copy_not_supported(
    tmp_path: Path, df: pl.DataFrame, statement: str, message: str
) -> None:
    with (
        pl.SQLContext(t=df) as ctx,
        pytest.raises(SQLInterfaceError, match=message),
    ):
        ctx.execute(statement.replace("{}", tmp_path.as_posix()))
    assert list(tmp_path.iterdir()) == []


def test_copy_with_file_writes_off(tmp_path: Path, df: pl.DataFrame) -> None:
    path = tmp_path / "out.csv"
    with (
        pl.SQLContext(t=df, allow_file_writes=False) as ctx,
        pytest.raises(SQLInterfaceError, match="writing files is disabled"),
    ):
        ctx.execute(f"COPY t TO '{path.as_posix()}'")
    assert not path.exists()


def test_copy_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    with pl.SQLContext() as ctx:
        ctx.execute("CREATE TABLE t (id INT, day DATE, note VARCHAR)")
        # as in Postgres: no header unless given, fields by position into the column
        # types, an unquoted empty field is NULL
        path.write_text("1,2024-01-02,a\n2,,\n")
        ctx.execute(f"COPY t FROM '{path.as_posix()}'")
        # with a NULL marker, an empty field is an empty string; HEADER true skips one
        # line
        path.write_text("x;y\n3;\n4;\\N\n")
        ctx.execute(
            f"COPY t (id, note) FROM '{path.as_posix()}' "
            "(FORMAT csv, HEADER true, DELIMITER ';', NULL '\\N')"
        )
        expected = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "day": [date(2024, 1, 2), None, None, None],
                "note": ["a", None, "", None],
            },
            schema={"id": pl.Int32, "day": pl.Date, "note": pl.String},
        )
        assert_frame_equal(ctx.execute("SELECT * FROM t", eager=True), expected)


def test_copy_from_round_trip(tmp_path: Path, df: pl.DataFrame) -> None:
    with pl.SQLContext(src=df) as ctx:
        ctx.execute("CREATE TABLE t (x BIGINT, s VARCHAR)")
        for ext, options in [("csv", "(HEADER true)"), ("parquet", "")]:
            path = (tmp_path / f"out.{ext}").as_posix()
            ctx.execute(f"COPY src TO '{path}'", eager=True)
            ctx.execute(f"COPY t FROM '{path}' {options}")
        assert_frame_equal(
            ctx.execute("SELECT * FROM t", eager=True), pl.concat([df, df])
        )


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        (None, FileNotFoundError),
        ("x\n", pl.exceptions.ComputeError),
    ],
)
def test_copy_from_fails_at_the_statement(
    tmp_path: Path, contents: str | None, error: type[Exception]
) -> None:
    path = tmp_path / "in.csv"
    if contents is not None:
        path.write_text(contents)
    with pl.SQLContext(t=pl.DataFrame({"a": [1]})) as ctx:
        with pytest.raises(error):
            ctx.execute(f"COPY t FROM '{path.as_posix()}'")
        assert ctx.execute("SELECT a FROM t", eager=True)["a"].to_list() == [1]

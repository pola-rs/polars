from __future__ import annotations

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
        ("COPY t FROM '{}/in.csv'", "COPY FROM is not supported; use CREATE TABLE"),
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

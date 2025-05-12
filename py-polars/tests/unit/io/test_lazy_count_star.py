from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from polars.lazyframe.frame import LazyFrame

import gzip
import re
from tempfile import NamedTemporaryFile

import pytest

import polars as pl
from polars.testing import assert_frame_equal


# Parameters
# * lf: COUNT(*) query
def assert_fast_count(
    lf: LazyFrame,
    expected_count: int,
    *,
    expected_name: str = "len",
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    capfd.readouterr()  # resets stderr
    result = lf.collect()
    capture = capfd.readouterr().err
    project_logs = set(re.findall(r"project: \d+", capture))

    # Logs current differ depending on file type / implementation dispatch
    if "FAST COUNT" in lf.explain():
        # * Should be no projections when fast count is enabled
        assert not project_logs
    else:
        # * Otherwise should have at least one `project: 0` (there is 1 per file).
        assert project_logs == {"project: 0"}

    assert result.schema == {expected_name: pl.get_index_type()}
    assert result.item() == expected_count

    # Test effect of the environment variable
    monkeypatch.setenv("POLARS_FAST_FILE_COUNT_DISPATCH", "0")

    capfd.readouterr()
    lf.collect()
    capture = capfd.readouterr().err
    project_logs = set(re.findall(r"project: \d+", capture))

    assert "FAST COUNT" not in lf.explain()
    assert project_logs == {"project: 0"}

    monkeypatch.setenv("POLARS_FAST_FILE_COUNT_DISPATCH", "1")

    capfd.readouterr()
    lf.collect()
    capture = capfd.readouterr().err
    project_logs = set(re.findall(r"project: \d+", capture))

    assert "FAST COUNT" in lf.explain()
    assert not project_logs


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.csv", 27), ("foods*.csv", 27 * 5)]
)
def test_count_csv(
    io_files_path: Path,
    path: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lf = pl.scan_csv(io_files_path / path).select(pl.len())

    assert_fast_count(lf, n_rows, capfd=capfd, monkeypatch=monkeypatch)


def test_count_csv_comment_char(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    q = pl.scan_csv(
        b"""
a,b
1,2

#
3,4
""",
        comment_prefix="#",
    )

    assert_frame_equal(
        q.collect(), pl.DataFrame({"a": [1, None, 3], "b": [2, None, 4]})
    )

    q = q.select(pl.len())
    assert_fast_count(q, 3, capfd=capfd, monkeypatch=monkeypatch)


def test_count_csv_no_newline_on_last_22564() -> None:
    data = b"""\
a,b
1,2
3,4
5,6"""

    assert pl.scan_csv(data).collect().height == 3
    assert pl.scan_csv(data, comment_prefix="#").collect().height == 3

    assert pl.scan_csv(data).select(pl.len()).collect().item() == 3
    assert pl.scan_csv(data, comment_prefix="#").select(pl.len()).collect().item() == 3


@pytest.mark.write_disk
def test_commented_csv(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with NamedTemporaryFile() as csv_a:
        csv_a.write(b"A,B\nGr1,A\nGr1,B\n# comment line\n")
        csv_a.seek(0)

        lf = pl.scan_csv(csv_a.name, comment_prefix="#").select(pl.len())
        assert_fast_count(lf, 2, capfd=capfd, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    ("pattern", "n_rows"), [("small.parquet", 4), ("foods*.parquet", 54)]
)
def test_count_parquet(
    io_files_path: Path,
    pattern: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lf = pl.scan_parquet(io_files_path / pattern).select(pl.len())
    assert_fast_count(lf, n_rows, capfd=capfd, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.ipc", 27), ("foods*.ipc", 27 * 2)]
)
def test_count_ipc(
    io_files_path: Path,
    path: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lf = pl.scan_ipc(io_files_path / path).select(pl.len())
    assert_fast_count(lf, n_rows, capfd=capfd, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.ndjson", 27), ("foods*.ndjson", 27 * 2)]
)
def test_count_ndjson(
    io_files_path: Path,
    path: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lf = pl.scan_ndjson(io_files_path / path).select(pl.len())
    assert_fast_count(lf, n_rows, capfd=capfd, monkeypatch=monkeypatch)


def test_count_compressed_csv_18057(
    io_files_path: Path,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_file = io_files_path / "gzipped.csv.gz"

    expected = pl.DataFrame(
        {"a": [1, 2, 3], "b": ["a", "b", "c"], "c": [1.0, 2.0, 3.0]}
    )
    lf = pl.scan_csv(csv_file, truncate_ragged_lines=True)
    out = lf.collect()
    assert_frame_equal(out, expected)
    # This also tests:
    # #18070 "CSV count_rows does not skip empty lines at file start"
    # as the file has an empty line at the beginning.

    q = lf.select(pl.len())
    assert_fast_count(q, 3, capfd=capfd, monkeypatch=monkeypatch)


def test_count_compressed_ndjson(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data.jsonl.gz"
    df = pl.DataFrame({"x": range(5)})

    with gzip.open(path, "wb") as f:
        df.write_ndjson(f)  # type: ignore[call-overload]

    lf = pl.scan_ndjson(path).select(pl.len())
    assert_fast_count(lf, 5, capfd=capfd, monkeypatch=monkeypatch)


def test_count_projection_pd(
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    df = pl.DataFrame({"a": range(3), "b": range(3)})

    q = (
        pl.scan_csv(df.write_csv().encode())
        .with_row_index()
        .select(pl.all())
        .select(pl.len())
    )

    # Manual assert, this is not converted to FAST COUNT but we will have
    # 0-width projections.

    monkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err
    project_logs = set(re.findall(r"project: \d+", capture))

    assert project_logs == {"project: 0"}
    assert result.item() == 3

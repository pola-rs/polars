from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from polars.lazyframe.frame import LazyFrame
    from tests.conftest import PlMonkeyPatch

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
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    capfd.readouterr()  # resets stderr

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
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

    # We disable the fast-count optimization to check that the normal scan
    # logic counts as expected.
    plmonkeypatch.setenv("POLARS_NO_FAST_FILE_COUNT", "1")

    capfd.readouterr()

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
        assert lf.collect().item() == expected_count

    capture = capfd.readouterr().err
    project_logs = set(re.findall(r"project: \d+", capture))

    assert "FAST COUNT" not in lf.explain()
    assert project_logs == {"project: 0"}

    plmonkeypatch.setenv("POLARS_NO_FAST_FILE_COUNT", "0")

    plan = lf.explain()
    if "Csv" not in plan:
        assert "FAST COUNT" not in plan
        return

    # CSV is the only format that uses a custom fast-count kernel, so we want
    # to make sure that the normal scan logic has the same count behavior. Here
    # we restore the default behavior that allows the fast-count optimization.
    assert "FAST COUNT" in plan

    capfd.readouterr()

    with plmonkeypatch.context() as cx:
        cx.setenv("POLARS_VERBOSE", "1")
        assert lf.collect().item() == expected_count

    capture = capfd.readouterr().err
    project_logs = set(re.findall(r"project: \d+", capture))

    assert not project_logs


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.csv", 27), ("foods*.csv", 27 * 5)]
)
def test_count_csv(
    io_files_path: Path,
    path: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    lf = pl.scan_csv(io_files_path / path).select(pl.len())

    assert_fast_count(lf, n_rows, capfd=capfd, plmonkeypatch=plmonkeypatch)


def test_count_csv_comment_char(
    capfd: pytest.CaptureFixture[str], plmonkeypatch: PlMonkeyPatch
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
    assert_fast_count(q, 3, capfd=capfd, plmonkeypatch=plmonkeypatch)


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
    capfd: pytest.CaptureFixture[str], plmonkeypatch: PlMonkeyPatch
) -> None:
    with NamedTemporaryFile() as csv_a:
        csv_a.write(b"A,B\nGr1,A\nGr1,B\n# comment line\n")
        csv_a.seek(0)

        lf = pl.scan_csv(csv_a.name, comment_prefix="#").select(pl.len())
        assert_fast_count(lf, 2, capfd=capfd, plmonkeypatch=plmonkeypatch)

    lf = pl.scan_csv(
        b"AAA",
        has_header=False,
        comment_prefix="#",
    ).select(pl.len())
    assert_fast_count(lf, 1, capfd=capfd, plmonkeypatch=plmonkeypatch)

    lf = pl.scan_csv(
        b"AAA\nBBB",
        has_header=False,
        comment_prefix="#",
    ).select(pl.len())
    assert_fast_count(lf, 2, capfd=capfd, plmonkeypatch=plmonkeypatch)

    lf = pl.scan_csv(
        b"AAA\n#comment\nBBB\n#comment",
        has_header=False,
        comment_prefix="#",
    ).select(pl.len())
    assert_fast_count(lf, 2, capfd=capfd, plmonkeypatch=plmonkeypatch)

    lf = pl.scan_csv(
        b"AAA\n#comment\nBBB\n#comment\nCCC\n#comment",
        has_header=False,
        comment_prefix="#",
    ).select(pl.len())
    assert_fast_count(lf, 3, capfd=capfd, plmonkeypatch=plmonkeypatch)

    lf = pl.scan_csv(
        b"AAA\n#comment\nBBB\n#comment\nCCC\n#comment\n",
        has_header=False,
        comment_prefix="#",
    ).select(pl.len())
    assert_fast_count(lf, 3, capfd=capfd, plmonkeypatch=plmonkeypatch)


@pytest.mark.parametrize(
    ("pattern", "n_rows"), [("small.parquet", 4), ("foods*.parquet", 54)]
)
def test_count_parquet(
    io_files_path: Path,
    pattern: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    lf = pl.scan_parquet(io_files_path / pattern).select(pl.len())
    assert_fast_count(lf, n_rows, capfd=capfd, plmonkeypatch=plmonkeypatch)


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.ipc", 27), ("foods*.ipc", 27 * 2)]
)
def test_count_ipc(
    io_files_path: Path,
    path: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    lf = pl.scan_ipc(io_files_path / path).select(pl.len())
    assert_fast_count(lf, n_rows, capfd=capfd, plmonkeypatch=plmonkeypatch)


@pytest.mark.parametrize(
    ("path", "n_rows"), [("foods1.ndjson", 27), ("foods*.ndjson", 27 * 2)]
)
def test_count_ndjson(
    io_files_path: Path,
    path: str,
    n_rows: int,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    lf = pl.scan_ndjson(io_files_path / path).select(pl.len())
    assert_fast_count(lf, n_rows, capfd=capfd, plmonkeypatch=plmonkeypatch)


def test_count_compressed_csv_18057(
    io_files_path: Path,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
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
    assert_fast_count(q, 3, capfd=capfd, plmonkeypatch=plmonkeypatch)


@pytest.mark.write_disk
def test_count_compressed_ndjson(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], plmonkeypatch: PlMonkeyPatch
) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data.jsonl.gz"
    df = pl.DataFrame({"x": range(5)})

    with gzip.open(path, "wb") as f:
        df.write_ndjson(f)  # type: ignore[call-overload]

    lf = pl.scan_ndjson(path).select(pl.len())
    assert_fast_count(lf, 5, capfd=capfd, plmonkeypatch=plmonkeypatch)


def test_count_projection_pd(
    capfd: pytest.CaptureFixture[str], plmonkeypatch: PlMonkeyPatch
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

    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err
    project_logs = set(re.findall(r"project: \d+", capture))

    assert project_logs == {"project: 0"}
    assert result.item() == 3


def test_csv_scan_skip_lines_len_22889(
    capfd: pytest.CaptureFixture[str], plmonkeypatch: PlMonkeyPatch
) -> None:
    bb = b"col\n1\n2\n3"
    lf = pl.scan_csv(bb, skip_lines=2).select(pl.len())
    assert_fast_count(lf, 1, capfd=capfd, plmonkeypatch=plmonkeypatch)

    # trigger multi-threading code path
    bb_10k = b"1\n2\n3\n4\n5\n6\n7\n8\n9\n0\n" * 1000
    lf = pl.scan_csv(bb_10k, skip_lines=1000, has_header=False).select(pl.len())
    assert_fast_count(lf, 9000, capfd=capfd, plmonkeypatch=plmonkeypatch)

    # for comparison
    out = pl.scan_csv(bb, skip_lines=2).collect().select(pl.len())
    expected = pl.DataFrame({"len": [1]}, schema={"len": pl.get_index_type()})
    assert_frame_equal(expected, out)


@pytest.mark.write_disk
@pytest.mark.slow
@pytest.mark.parametrize(
    "exec_str",
    [
        "pl.LazyFrame(height=n_rows).select(pl.len()).collect().item()",
        "pl.scan_parquet(parquet_file_path).select(pl.len()).collect().item()",
        "pl.scan_ipc(ipc_file_path).select(pl.len()).collect().item()",
        'pl.LazyFrame({"a": s, "b": s, "c": s}).select("c", "b").collect().height',
        """\
pl.collect_all(
    [
        pl.scan_parquet(parquet_file_path).select(pl.len()),
        pl.scan_ipc(ipc_file_path).select(pl.len()),
        pl.LazyFrame(height=n_rows).select(pl.len()),
    ]
)[0].item()""",
    ],
)
def test_streaming_fast_count_disables_morsel_split(
    tmp_path: Path, exec_str: str
) -> None:
    n_rows = (1 << 32) - 2
    parquet_file_path = tmp_path / "data.parquet"
    ipc_file_path = tmp_path / "data.ipc"

    script_args = [str(n_rows), str(parquet_file_path), str(ipc_file_path), exec_str]

    # We spawn 2 processes - the first process sets a huge ideal morsel size to
    # generate the data quickly. The 2nd process sets the ideal morsel size to 1,
    # making it so that if morsel splitting is performed it would exceed the
    # timeout of 5 seconds.

    assert (
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import os
import sys

os.environ["POLARS_IDEAL_MORSEL_SIZE"] = str(1_000_000_000)

import polars as pl

pl.Config.set_engine_affinity("streaming")

(
    _,
    n_rows,
    parquet_file_path,
    ipc_file_path,
    _,
) = sys.argv

n_rows = int(n_rows)

pl.LazyFrame(height=n_rows).sink_parquet(parquet_file_path, row_group_size=1_000_000_000)
pl.LazyFrame(height=n_rows).sink_ipc(ipc_file_path, record_batch_size=1_000_000_000)

print("OK", end="")
""",
                *script_args,
            ],
            timeout=5,
        )
        == b"OK"
    )

    assert (
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import os
import sys

os.environ["POLARS_IDEAL_MORSEL_SIZE"] = "1"

import polars as pl

pl.Config.set_engine_affinity("streaming")

(
    _,
    n_rows,
    parquet_file_path,
    ipc_file_path,
    exec_str,
) = sys.argv

n_rows = int(n_rows)

s = pl.Series([{}], dtype=pl.Struct({})).new_from_index(0, n_rows)
assert eval(exec_str) == n_rows

print("OK", end="")
""",
                *script_args,
            ],
            timeout=5,
        )
        == b"OK"
    )

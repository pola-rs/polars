from typing import Any
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from pathlib import Path


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("scan", "write"),
    [
        (pl.scan_ipc, pl.DataFrame.write_ipc),
        (pl.scan_parquet, pl.DataFrame.write_parquet),
        (pl.scan_csv, pl.DataFrame.write_csv),
        (pl.scan_ndjson, pl.DataFrame.write_ndjson),
    ],
)
def test_include_file_paths(tmp_path: Path, scan: Any, write: Any) -> None:
    a_path = tmp_path / "a"
    b_path = tmp_path / "b"

    write(pl.DataFrame({"a": [5, 10]}), a_path)
    write(pl.DataFrame({"a": [1996]}), b_path)

    out = scan([a_path, b_path], include_file_paths="f")

    assert_frame_equal(
        out.collect(),
        pl.DataFrame(
            {
                "a": [5, 10, 1996],
                "f": [str(a_path), str(a_path), str(b_path)],
            }
        ),
    )

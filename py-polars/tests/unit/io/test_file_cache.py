from __future__ import annotations

import sys
from pathlib import Path

import pytest

import polars as pl


@pytest.mark.slow()
@pytest.mark.skipif(sys.platform == "win32", reason="Paths differ on windows")
def test_file_cache_ttl(
    monkeypatch: pytest.MonkeyPatch, io_files_path: Path, tmp_path: Path
) -> None:
    file_cache_prefix = Path(pl._get_file_cache_prefix())
    io_files_path = io_files_path.absolute()
    monkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    from time import sleep

    from blake3 import blake3

    paths = [
        io_files_path / "foods1.csv",
        io_files_path / "foods2.csv",
    ]
    hashes = [blake3(bytes(x)).hexdigest()[:32] for x in paths]
    metadata_file_paths = [file_cache_prefix / f"m/{x}" for x in hashes]
    data_file_dir = file_cache_prefix / "d/"
    pl.scan_csv(paths).collect()
    pl.scan_csv(paths[1], file_cache_ttl=0).collect()

    has_data_file = [False, False]
    for data_file in data_file_dir.iterdir():
        for i, h in enumerate(hashes):
            has_data_file[i] |= data_file.name.startswith(h)

    assert all(x.exists() for x in metadata_file_paths)
    assert all(has_data_file)

    sleep(5)

    has_data_file = [False, False]
    for data_file in data_file_dir.iterdir():
        for i, h in enumerate(hashes):
            has_data_file[i] |= data_file.name.startswith(h)

    assert [x.exists() for x in metadata_file_paths] == [True, False]
    assert has_data_file == [True, False]

from __future__ import annotations

import sys
from pathlib import Path
from polars.testing import assert_frame_equal

import pytest

import polars as pl


@pytest.mark.slow()
@pytest.mark.skipif(sys.platform == "win32", reason="Paths differ on windows")
def test_file_cache_ttl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POLARS_FORCE_ASYNC", "1")

    file_cache_prefix = Path(pl._get_file_cache_prefix())
    tmp_path = tmp_path.absolute()
    tmp_path.mkdir(exist_ok=True)

    from time import sleep

    from blake3 import blake3

    paths = [
        tmp_path / "1.csv",
        tmp_path / "2.csv",
    ]
    df = pl.DataFrame({"a": 1})

    for p in paths:
        df.write_csv(p)

    hashes = [blake3(bytes(x)).hexdigest()[:32] for x in paths]
    metadata_file_paths = [file_cache_prefix / f"m/{x}" for x in hashes]
    data_file_dir = file_cache_prefix / "d/"
    pl.scan_csv(paths).collect()

    assert_frame_equal(
        pl.scan_csv(paths[1], schema=df.collect_schema(), file_cache_ttl=0).collect(),
        df,
    )

    sleep(5)

    has_data_file = [False, False]
    for data_file in data_file_dir.iterdir():
        for i, h in enumerate(hashes):
            has_data_file[i] |= data_file.name.startswith(h)

    assert [x.exists() for x in metadata_file_paths] == [True, False]
    assert has_data_file == [True, False]

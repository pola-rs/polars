from pathlib import Path

import pytest

import polars as pl


@pytest.fixture
def sample_pcap_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_s": [1000, 2000],
            "time_ns": [100, 200],
            "incl_len": [4, 4],
            "orig_len": [4, 4],
            "data": [b"abcd", b"efgh"],
        },
        schema={
            "time_s": pl.Int64,
            "time_ns": pl.UInt32,
            "incl_len": pl.UInt32,
            "orig_len": pl.UInt32,
            "data": pl.Binary,
        },
    )


def test_pcap_roundtrip(tmp_path: Path, sample_pcap_df: pl.DataFrame) -> None:
    path = tmp_path / "test.pcap"

    # Write
    sample_pcap_df.write_pcap(path)
    assert path.exists()

    # Read
    df_read = pl.read_pcap(path)

    # Compare
    assert df_read.equals(sample_pcap_df)


def test_read_pcap_n_rows(tmp_path: Path, sample_pcap_df: pl.DataFrame) -> None:
    path = tmp_path / "test.pcap"
    sample_pcap_df.write_pcap(path)

    df_read = pl.read_pcap(path, n_rows=1)
    assert df_read.height == 1
    assert df_read.row(0) == sample_pcap_df.row(0)


def test_write_pcap_file_like(tmp_path: Path, sample_pcap_df: pl.DataFrame) -> None:
    path = tmp_path / "test.pcap"
    with path.open("wb") as f:
        sample_pcap_df.write_pcap(f)

    with path.open("rb") as f:
        df_read = pl.read_pcap(f)

    assert df_read.equals(sample_pcap_df)

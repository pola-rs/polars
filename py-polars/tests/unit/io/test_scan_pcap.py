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


def test_scan_pcap(tmp_path: Path, sample_pcap_df: pl.DataFrame) -> None:
    path = tmp_path / "test.pcap"
    sample_pcap_df.write_pcap(path)

    ldf = pl.scan_pcap(path)
    assert isinstance(ldf, pl.LazyFrame)

    df_read = ldf.collect()
    assert df_read.equals(sample_pcap_df)


def test_scan_pcap_projection(tmp_path: Path, sample_pcap_df: pl.DataFrame) -> None:
    path = tmp_path / "test.pcap"
    sample_pcap_df.write_pcap(path)

    ldf = pl.scan_pcap(path).select("time_s", "data")
    df_read = ldf.collect()

    assert df_read.columns == ["time_s", "data"]
    assert df_read.equals(sample_pcap_df.select("time_s", "data"))


def test_scan_pcap_filter(tmp_path: Path, sample_pcap_df: pl.DataFrame) -> None:
    path = tmp_path / "test.pcap"
    sample_pcap_df.write_pcap(path)

    ldf = pl.scan_pcap(path).filter(pl.col("time_s") > 1500)
    df_read = ldf.collect()

    assert df_read.height == 1
    assert df_read.equals(sample_pcap_df.filter(pl.col("time_s") > 1500))


def test_scan_pcap_limit(tmp_path: Path, sample_pcap_df: pl.DataFrame) -> None:
    path = tmp_path / "test.pcap"
    sample_pcap_df.write_pcap(path)

    ldf = pl.scan_pcap(path).limit(1)
    df_read = ldf.collect()

    assert df_read.height == 1
    assert df_read.equals(sample_pcap_df.head(1))

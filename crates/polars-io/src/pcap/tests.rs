use crate::pcap::{PcapReader, PcapWriter};
use crate::shared::SerReader;
use std::fs::File;
use polars_core::prelude::*;
use std::path::Path;
use std::io::Cursor;

#[test]
fn test_pcap_roundtrip() -> PolarsResult<()> {
    let mut df = DataFrame::new(vec![
        Series::new("time_s".into(), [1000i64, 2000i64]).into(),
        Series::new("time_ns".into(), [100u32, 200u32]).into(),
        Series::new("incl_len".into(), [4u32, 4u32]).into(),
        Series::new("orig_len".into(), [4u32, 4u32]).into(),
        Series::new("data".into(), [b"abcd".as_slice(), b"efgh".as_slice()]).into(),
    ])?;

    let mut buf = Vec::new();
    let writer = PcapWriter::new(&mut buf);
    writer.finish(&mut df)?;

    let reader = PcapReader::new(Cursor::new(buf));
    let df_read = reader.finish()?;

    assert_eq!(df_read.height(), 2);
    assert_eq!(df_read.column("time_s")?.i64()?.get(0), Some(1000));
    assert_eq!(df_read.column("data")?.binary()?.get(0), Some(b"abcd".as_slice()));

    Ok(())
}

#[test]
fn test_pcap_read_foods1() -> PolarsResult<()> {
    // Relative path from crates/polars-io
    let path = Path::new("../../examples/datasets/foods1.pcap");
    if !path.exists() {
        // If not found at relative path, try absolute from root
        let root_path = Path::new("examples/datasets/foods1.pcap");
        if !root_path.exists() {
             return Ok(());
        }
        let file = File::open(root_path)?;
        let reader = PcapReader::new(file);
        let df = reader.finish()?;
        assert_eq!(df.height(), 27);
        return Ok(());
    }

    let file = File::open(path)?;
    let reader = PcapReader::new(file);
    let df = reader.finish()?;

    assert_eq!(df.height(), 27);
    assert_eq!(df.column("time_s")?.i64()?.get(0), Some(1678900000));
    assert_eq!(df.column("data")?.binary()?.get(0), Some(b"vegetables".as_slice()));

    Ok(())
}

#[test]
fn test_pcap_read_with_n_rows() -> PolarsResult<()> {
    let path = Path::new("../../examples/datasets/foods1.pcap");
    if !path.exists() {
        let root_path = Path::new("examples/datasets/foods1.pcap");
        if !root_path.exists() {
             return Ok(());
        }
        let file = File::open(root_path)?;
        let df = PcapReader::new(file).with_n_rows(Some(5)).finish()?;
        assert_eq!(df.height(), 5);
        return Ok(());
    }

    let file = File::open(path)?;
    let df = PcapReader::new(file)
        .with_n_rows(Some(5))
        .finish()?;

    assert_eq!(df.height(), 5);
    Ok(())
}

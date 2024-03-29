use std::io::{Seek, SeekFrom, Write};

#[cfg(feature = "flight")]
use arrow::io::ipc::read::FlightFileReader;
use polars::prelude::*;

#[test]
fn test_ipc_compression_variadic_buffers() {
    let mut df = df![
        "foo" => std::iter::repeat("Home delivery vat 24 %").take(3).collect::<Vec<_>>()
    ]
    .unwrap();

    let mut file = std::io::Cursor::new(vec![]);
    IpcWriter::new(&mut file)
        .with_compression(Some(IpcCompression::LZ4))
        .with_pl_flavor(true)
        .finish(&mut df)
        .unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();
    let out = IpcReader::new(file).finish().unwrap();

    assert_eq!(out.shape(), (3, 1));
}

#[test]
#[cfg(feature = "flight")]
fn test_ipc_flight_round_trip() -> PolarsResult<()> {
    // Roundtrip test df -> IPC -> Flight -> IPC Reader
    let mut df = df![
        "foo" => std::iter::repeat("Home delivery vat 24 %").take(3).collect::<Vec<_>>()
    ]
    .unwrap();

    let mut file = std::io::Cursor::new(vec![]);
    IpcWriter::new(&mut file)
        .with_pl_flavor(true)
        .finish(&mut df)
        .unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();

    // Read in raw bytes in streaming format
    let mut flight_reader = FlightFileReader::new(file);
    let mut file_buffer = std::io::Cursor::new(vec![]);
    while let Some(Ok(data)) = flight_reader.next() {
        // Write the metadatasize
        // For format see IPC docs from Arrow
        let metadata_size = data.data_header.len() as i32;
        file_buffer.write(&metadata_size.to_le_bytes())?;
        file_buffer.write(&data.data_header)?;
        file_buffer.write(&data.data_body)?;
    }
    file_buffer.set_position(0);
    // The data comes in IPC streaming format from arrow flight
    let out = IpcStreamReader::new(file_buffer).finish()?;
    assert_eq!(df, out);
    Ok(())
}

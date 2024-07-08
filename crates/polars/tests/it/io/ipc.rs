use std::io::{Cursor, Seek, SeekFrom};

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
        .with_compat_level(CompatLevel::newest())
        .finish(&mut df)
        .unwrap();

    file.seek(SeekFrom::Start(0)).unwrap();
    let out = IpcReader::new(file).finish().unwrap();

    assert_eq!(out.shape(), (3, 1));
}

#[cfg(test)]
pub(crate) fn create_df() -> DataFrame {
    let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
    let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
    DataFrame::new(vec![s0, s1]).unwrap()
}

#[test]
fn write_and_read_ipc() {
    // Vec<T> : Write + Read
    // Cursor<Vec<_>>: Seek
    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut df = create_df();

    IpcWriter::new(&mut buf)
        .finish(&mut df)
        .expect("ipc writer");

    buf.set_position(0);

    let df_read = IpcReader::new(buf).finish().unwrap();
    assert!(df.equals(&df_read));
}

#[test]
fn test_read_ipc_with_projection() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

    IpcWriter::new(&mut buf)
        .finish(&mut df)
        .expect("ipc writer");
    buf.set_position(0);

    let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
    let df_read = IpcReader::new(buf)
        .with_projection(Some(vec![1, 2]))
        .finish()
        .unwrap();
    assert_eq!(df_read.shape(), (3, 2));
    df_read.equals(&expected);
}

#[test]
fn test_read_ipc_with_columns() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

    IpcWriter::new(&mut buf)
        .finish(&mut df)
        .expect("ipc writer");
    buf.set_position(0);

    let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
    let df_read = IpcReader::new(buf)
        .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
        .finish()
        .unwrap();
    df_read.equals(&expected);

    for compat_level in [0, 1].map(|level| CompatLevel::with_level(level).unwrap()) {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df![
            "letters" => ["x", "y", "z"],
            "ints" => [123, 456, 789],
            "floats" => [4.5, 10.0, 10.0],
            "other" => ["misc", "other", "value"],
        ]
        .unwrap();
        IpcWriter::new(&mut buf)
            .with_compat_level(compat_level)
            .finish(&mut df)
            .expect("ipc writer");
        buf.set_position(0);
        let expected = df![
            "letters" => ["x", "y", "z"],
            "floats" => [4.5, 10.0, 10.0],
            "other" => ["misc", "other", "value"],
            "ints" => [123, 456, 789],
        ]
        .unwrap();
        let df_read = IpcReader::new(&mut buf)
            .with_columns(Some(vec![
                "letters".to_string(),
                "floats".to_string(),
                "other".to_string(),
                "ints".to_string(),
            ]))
            .finish()
            .unwrap();
        assert!(df_read.equals(&expected));
    }
}

#[test]
fn test_write_with_compression() {
    let mut df = create_df();

    let compressions = vec![None, Some(IpcCompression::LZ4), Some(IpcCompression::ZSTD)];

    for compression in compressions.into_iter() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        IpcWriter::new(&mut buf)
            .with_compression(compression)
            .finish(&mut df)
            .expect("ipc writer");
        buf.set_position(0);

        let df_read = IpcReader::new(buf)
            .finish()
            .unwrap_or_else(|_| panic!("IPC reader: {:?}", compression));
        assert!(df.equals(&df_read));
    }
}

#[test]
fn write_and_read_ipc_empty_series() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let chunked_array = Float64Chunked::new("empty", &[0_f64; 0]);
    let mut df = DataFrame::new(vec![chunked_array.into_series()]).unwrap();
    IpcWriter::new(&mut buf)
        .finish(&mut df)
        .expect("ipc writer");

    buf.set_position(0);

    let df_read = IpcReader::new(buf).finish().unwrap();
    assert!(df.equals(&df_read));
}

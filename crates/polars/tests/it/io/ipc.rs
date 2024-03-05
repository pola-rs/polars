use std::io::{Seek, SeekFrom};

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

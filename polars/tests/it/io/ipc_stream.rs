use super::*;

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use polars::export::arrow::io::ipc::write;
    use polars_core::df;
    use polars_core::prelude::*;
    use polars_io::ipc::*;
    use polars_io::{SerReader, SerWriter};

    use crate::io::create_df;

    #[test]
    fn write_and_read_ipc_stream() {
        // Vec<T> : Write + Read
        // Cursor<Vec<_>>: Seek
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = create_df();

        IpcStreamWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");

        buf.set_position(0);

        let df_read = IpcStreamReader::new(buf).finish().unwrap();
        assert!(df.frame_equal(&df_read));
    }

    #[test]
    fn test_read_ipc_stream_with_projection() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        IpcStreamWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = IpcStreamReader::new(buf)
            .with_projection(Some(vec![1, 2]))
            .finish()
            .unwrap();
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
    }

    #[test]
    fn test_read_ipc_stream_with_columns() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        IpcStreamWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = IpcStreamReader::new(buf)
            .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
            .finish()
            .unwrap();
        df_read.frame_equal(&expected);

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df![
            "a" => ["x", "y", "z"],
            "b" => [123, 456, 789],
            "c" => [4.5, 10.0, 10.0],
            "d" => ["misc", "other", "value"],
        ]
        .unwrap();
        IpcStreamWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");
        buf.set_position(0);
        let expected = df![
            "a" => ["x", "y", "z"],
            "c" => [4.5, 10.0, 10.0],
            "d" => ["misc", "other", "value"],
            "b" => [123, 456, 789],
        ]
        .unwrap();
        let df_read = IpcStreamReader::new(buf)
            .with_columns(Some(vec![
                "a".to_string(),
                "c".to_string(),
                "d".to_string(),
                "b".to_string(),
            ]))
            .finish()
            .unwrap();
        df_read.frame_equal(&expected);
    }

    #[test]
    fn test_read_invalid_stream() {
        let buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        assert!(IpcStreamReader::new(buf.clone()).arrow_schema().is_err());
        assert!(IpcStreamReader::new(buf).finish().is_err());
    }

    #[test]
    fn test_write_with_compression() {
        let mut df = create_df();

        let compressions = vec![
            None,
            Some(write::Compression::LZ4),
            Some(write::Compression::ZSTD),
        ];

        for compression in compressions.into_iter() {
            let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
            IpcStreamWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&mut df)
                .expect("ipc writer");
            buf.set_position(0);

            let df_read = IpcStreamReader::new(buf)
                .finish()
                .expect(&format!("IPC reader: {:?}", compression));
            assert!(df.frame_equal(&df_read));
        }
    }

    #[test]
    fn write_and_read_ipc_stream_empty_series() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let chunked_array = Float64Chunked::new("empty", &[0_f64; 0]);
        let mut df = DataFrame::new(vec![chunked_array.into_series()]).unwrap();
        IpcStreamWriter::new(&mut buf)
            .finish(&mut df)
            .expect("ipc writer");

        buf.set_position(0);

        let df_read = IpcStreamReader::new(buf).finish().unwrap();
        assert!(df.frame_equal(&df_read));
    }
}

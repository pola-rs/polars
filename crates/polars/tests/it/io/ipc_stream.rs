#[cfg(test)]
mod test {
    use std::io::Cursor;

    use polars_core::prelude::*;
    use polars_core::{assert_df_eq, df};
    use polars_io::ipc::*;
    use polars_io::{SerReader, SerWriter};

    use crate::io::create_df;

    fn create_ipc_stream(mut df: DataFrame) -> Cursor<Vec<u8>> {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

        IpcStreamWriter::new(&mut buf)
            .finish(&mut df)
            .expect("failed to write ICP stream");

        buf.set_position(0);

        buf
    }

    #[test]
    fn write_and_read_ipc_stream() {
        let df = create_df();

        let reader = create_ipc_stream(df);

        let actual = IpcStreamReader::new(reader).finish().unwrap();

        let expected = create_df();
        assert_df_eq!(actual, expected);
    }

    #[test]
    fn test_read_ipc_stream_with_projection() {
        let df = df!(
            "a" => [1],
            "b" => [2],
            "c" => [3],
        )
        .unwrap();

        let reader = create_ipc_stream(df);

        let actual = IpcStreamReader::new(reader)
            .with_projection(Some(vec![1, 2]))
            .finish()
            .unwrap();

        let expected = df!(
            "b" => [2],
            "c" => [3],
        )
        .unwrap();
        assert_df_eq!(actual, expected);
    }

    #[test]
    fn test_read_ipc_stream_with_columns() {
        let df = df!(
            "a" => [1],
            "b" => [2],
            "c" => [3],
        )
        .unwrap();

        let reader = create_ipc_stream(df);

        let actual = IpcStreamReader::new(reader)
            .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
            .finish()
            .unwrap();

        let expected = df!(
            "c" => [3],
            "b" => [2],
        )
        .unwrap();
        assert_df_eq!(actual, expected);
    }

    #[test]
    fn test_read_ipc_stream_with_columns_reorder() {
        let df = df![
            "a" => [1],
            "b" => [2],
            "c" => [3],
        ]
        .unwrap();

        let reader = create_ipc_stream(df);

        let actual = IpcStreamReader::new(reader)
            .with_columns(Some(vec![
                "b".to_string(),
                "c".to_string(),
                "a".to_string(),
            ]))
            .finish()
            .unwrap();

        let expected = df![
            "b" => [2],
            "c" => [3],
            "a" => [1],
        ]
        .unwrap();
        assert_df_eq!(actual, expected);
    }

    #[test]
    fn test_read_invalid_stream() {
        let buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        assert!(IpcStreamReader::new(buf.clone()).arrow_schema().is_err());
        assert!(IpcStreamReader::new(buf).finish().is_err());
    }

    #[test]
    fn test_write_with_lz4_compression() {
        test_write_with_compression(IpcCompression::LZ4);
    }

    #[test]
    fn test_write_with_zstd_compression() {
        test_write_with_compression(IpcCompression::ZSTD);
    }

    fn test_write_with_compression(compression: IpcCompression) {
        let reader = {
            let mut writer: Cursor<Vec<u8>> = Cursor::new(Vec::new());
            IpcStreamWriter::new(&mut writer)
                .with_compression(Some(compression))
                .finish(&mut create_df())
                .unwrap();
            writer.set_position(0);
            writer
        };

        let actual = IpcStreamReader::new(reader).finish().unwrap();
        assert_df_eq!(actual, create_df());
    }

    #[test]
    fn write_and_read_ipc_stream_empty_series() {
        fn df() -> DataFrame {
            DataFrame::new(vec![Float64Chunked::new("empty", &[0_f64; 0]).into_series()]).unwrap()
        }

        let reader = create_ipc_stream(df());

        let actual = IpcStreamReader::new(reader).finish().unwrap();
        assert_df_eq!(df(), actual);
    }
}

mod read;
mod write;

pub use read::*;
pub use write::*;

use super::*;

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use polars_core::df;
    use polars_core::prelude::*;

    use super::{write, AvroReader, AvroWriter};
    use crate::prelude::*;

    #[test]
    fn test_write_and_read_with_compression() -> PolarsResult<()> {
        let mut write_df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "string" => &["a", "b"]
        )?;

        let compressions = vec![
            None,
            Some(write::Compression::Deflate),
            Some(write::Compression::Snappy),
        ];

        for compression in compressions.into_iter() {
            let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

            AvroWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&mut write_df)?;
            buf.set_position(0);

            let read_df = AvroReader::new(buf).finish()?;
            assert!(write_df.equals(&read_df));
        }

        Ok(())
    }

    #[test]
    fn test_with_projection() -> PolarsResult<()> {
        let mut df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "string" => &["a", "b"]
        )?;

        let expected_df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2]
        )?;

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

        AvroWriter::new(&mut buf).finish(&mut df)?;
        buf.set_position(0);

        let read_df = AvroReader::new(buf)
            .with_projection(Some(vec![0, 1]))
            .finish()?;

        assert!(expected_df.equals(&read_df));

        Ok(())
    }

    #[test]
    fn test_with_columns() -> PolarsResult<()> {
        let mut df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "string" => &["a", "b"]
        )?;

        let expected_df = df!(
            "i64" => &[1, 2],
            "string" => &["a", "b"]
        )?;

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

        AvroWriter::new(&mut buf).finish(&mut df)?;
        buf.set_position(0);

        let read_df = AvroReader::new(buf)
            .with_columns(Some(vec!["i64".to_string(), "string".to_string()]))
            .finish()?;

        assert!(expected_df.equals(&read_df));

        Ok(())
    }
}

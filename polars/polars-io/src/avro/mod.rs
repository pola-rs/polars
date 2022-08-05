mod read;
mod write;

use super::*;
use arrow::io::avro::avro_schema::error::Error as AvroError;
pub use read::*;
pub use write::*;

// we cannot implement the From trait because of the orphan rule
fn convert_err(e: AvroError) -> PolarsError {
    PolarsError::ComputeError(format!("{e}").into())
}

#[cfg(test)]
mod test {
    use super::{write, AvroReader, AvroWriter};
    use crate::prelude::*;
    use polars_core::df;
    use polars_core::prelude::*;
    use std::io::Cursor;

    #[test]
    fn test_write_and_read_with_compression() -> Result<()> {
        let mut write_df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
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
            assert!(write_df.frame_equal(&read_df));
        }

        Ok(())
    }

    #[test]
    fn test_with_projection() -> Result<()> {
        let mut df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
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

        assert!(expected_df.frame_equal(&read_df));

        Ok(())
    }

    #[test]
    fn test_with_columns() -> Result<()> {
        let mut df = df!(
            "i64" => &[1, 2],
            "f64" => &[0.1, 0.2],
            "utf8" => &["a", "b"]
        )?;

        let expected_df = df!(
            "i64" => &[1, 2],
            "utf8" => &["a", "b"]
        )?;

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

        AvroWriter::new(&mut buf).finish(&mut df)?;
        buf.set_position(0);

        let read_df = AvroReader::new(buf)
            .with_columns(Some(vec!["i64".to_string(), "utf8".to_string()]))
            .finish()?;

        assert!(expected_df.frame_equal(&read_df));

        Ok(())
    }
}

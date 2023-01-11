//! # Reading Apache parquet files.
//!
//! ## Example
//!
//! ```rust
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     let r = File::open("some_file.parquet").unwrap();
//!     let reader = ParquetReader::new(r);
//!     reader.finish()
//! }
//! ```
//!
#[cfg(feature = "async")]
pub(super) mod async_impl;
pub(super) mod mmap;
pub mod predicates;
mod read;
mod read_impl;
mod write;

pub use read::*;
pub use write::{BrotliLevel, GzipLevel, ZstdLevel, *};

use super::*;

#[cfg(test)]
mod test {
    use std::fs::File;
    use std::io::Cursor;

    use polars_core::df;
    use polars_core::prelude::*;

    use crate::prelude::*;

    #[test]
    fn test_parquet() {
        // In CI: This test will be skipped because the file does not exist.
        if let Ok(r) = File::open("data/simple.parquet") {
            let reader = ParquetReader::new(r);
            let df = reader.finish().unwrap();
            assert_eq!(df.get_column_names(), ["a", "b"]);
            assert_eq!(df.shape(), (3, 2));
        }
    }

    #[test]
    #[cfg(all(feature = "dtype-datetime", feature = "parquet"))]
    fn test_parquet_datetime_round_trip() -> PolarsResult<()> {
        use std::io::{Cursor, Seek, SeekFrom};

        let mut f = Cursor::new(vec![]);

        let mut df = df![
            "datetime" => [Some(191845729i64), Some(89107598), None, Some(3158971092)]
        ]?;

        df.try_apply("datetime", |s| {
            s.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        })?;

        ParquetWriter::new(&mut f).finish(&mut df)?;

        f.seek(SeekFrom::Start(0))?;

        let read = ParquetReader::new(f).finish()?;
        assert!(read.frame_equal_missing(&df));
        Ok(())
    }

    #[test]
    fn test_read_parquet_with_projection() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        ParquetWriter::new(&mut buf)
            .finish(&mut df)
            .expect("parquet writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = ParquetReader::new(buf)
            .with_projection(Some(vec![1, 2]))
            .finish()
            .unwrap();
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
    }

    #[test]
    fn test_read_parquet_with_columns() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df!("a" => [1, 2, 3], "b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();

        ParquetWriter::new(&mut buf)
            .finish(&mut df)
            .expect("parquet writer");
        buf.set_position(0);

        let expected = df!("b" => [2, 3, 4], "c" => [3, 4, 5]).unwrap();
        let df_read = ParquetReader::new(buf)
            .with_columns(Some(vec!["c".to_string(), "b".to_string()]))
            .finish()
            .unwrap();
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
    }
}

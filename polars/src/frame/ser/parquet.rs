//! # Reading Apache parquet files.
//!
//! ## Example
//!
//! ```rust
//! use polars::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> Result<DataFrame> {
//!     let r = File::open("some_file.parquet").unwrap();
//!     let reader = ParquetReader::new(r);
//!     reader.finish()
//! }
//! ```
//!
use super::{finish_reader, ArrowReader, ArrowResult, RecordBatch};
use crate::prelude::*;
use arrow::record_batch::RecordBatchReader;
use parquet::arrow::{
    arrow_reader::ParquetRecordBatchReader, ArrowReader as ParquetArrowReader,
    ParquetFileArrowReader,
};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::io::{Read, Seek};
use std::rc::Rc;
use std::sync::Arc;

/// Read Apache parquet format into a DataFrame.
pub struct ParquetReader<R> {
    reader: R,
    rechunk: bool,
    ignore_parser_error: bool,
}

impl ArrowReader for ParquetRecordBatchReader {
    fn next(&mut self) -> ArrowResult<Option<RecordBatch>> {
        self.next_batch()
    }

    fn schema(&self) -> Arc<Schema> {
        <Self as RecordBatchReader>::schema(self)
    }
}

impl<R> ParquetReader<R> {}

impl<R> SerReader<R> for ParquetReader<R>
where
    R: 'static + Read + Seek + parquet::file::reader::Length + parquet::file::reader::TryClone,
{
    fn new(reader: R) -> Self {
        ParquetReader {
            reader,
            rechunk: false,
            ignore_parser_error: false,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn with_ignore_parser_error(mut self) -> Self {
        self.ignore_parser_error = true;
        self
    }

    fn finish(self) -> Result<DataFrame> {
        let rechunk = self.rechunk;

        let file_reader = Rc::new(SerializedFileReader::new(self.reader)?);
        let n_rows = file_reader.metadata().file_metadata().num_rows() as usize;
        let mut arrow_reader = ParquetFileArrowReader::new(file_reader);
        let record_reader = arrow_reader.get_record_reader(n_rows)?;
        finish_reader(record_reader, rechunk, self.ignore_parser_error, None)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use std::fs::File;

    #[test]
    fn test_parquet() {
        let r = File::open("data/simple.parquet");
        match r {
            // local run test
            Ok(r) => {
                let reader = ParquetReader::new(r);
                let df = reader.finish().unwrap();
                assert_eq!(df.columns(), ["a", "b"]);
                assert_eq!(df.shape(), (3, 2));
            }
            // in ci: pass
            Err(_) => {}
        }
    }
}

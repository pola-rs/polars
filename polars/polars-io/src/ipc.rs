//! # (De)serializing Arrows IPC format.
//!
//! Arrow IPC is a [binary format format](https://arrow.apache.org/docs/python/ipc.html).
//! It is the recommended way to serialize and deserialize Polars DataFrames as this is most true
//! to the data schema.
//!
//! ## Example
//!
//! ```rust
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::io::Cursor;
//!
//!
//! let s0 = Series::new("days", &[0, 1, 2, 3, 4]);
//! let s1 = Series::new("temp", &[22.1, 19.9, 7., 2., 3.]);
//! let mut df = DataFrame::new(vec![s0, s1]).unwrap();
//!
//! // Create an in memory file handler.
//! // Vec<u8>: Read + Write
//! // Cursor<T>: Seek
//!
//! let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
//!
//! // write to the in memory buffer
//! IpcWriter::new(&mut buf).finish(&mut df).expect("ipc writer");
//!
//! // reset the buffers index after writing to the beginning of the buffer
//! buf.set_position(0);
//!
//! // read the buffer into a DataFrame
//! let df_read = IpcReader::new(buf).finish().unwrap();
//! assert!(df.frame_equal(&df_read));
//! ```
use super::{finish_reader, ArrowReader, ArrowResult};
use crate::predicates::PhysicalIoExpr;
use crate::prelude::*;
use arrow::io::ipc::write::WriteOptions;
use arrow::io::ipc::{read, write};
use polars_core::prelude::*;
use std::io::{BufWriter, Read, Seek, Write};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

/// Read Arrows IPC format into a DataFrame
///
/// # Example
/// ```
/// use polars_core::prelude::*;
/// use std::fs::File;
/// use polars_io::ipc::IpcReader;
/// use polars_io::SerReader;
///
/// fn example() -> Result<DataFrame> {
///     let file = File::open("file.ipc").expect("file not found");
///
///     IpcReader::new(file)
///         .finish()
/// }
/// ```
#[must_use]
pub struct IpcReader<R> {
    /// File or Stream object
    reader: R,
    /// Aggregates chunks afterwards to a single chunk.
    rechunk: bool,
    n_rows: Option<usize>,
    projection: Option<Vec<usize>>,
    columns: Option<Vec<String>>,
    row_count: Option<RowCount>,
}

impl<R: Read + Seek> IpcReader<R> {
    /// Get schema of the Ipc File
    pub fn schema(&mut self) -> Result<Schema> {
        let metadata = read::read_file_metadata(&mut self.reader)?;
        Ok((&metadata.schema.fields).into())
    }

    /// Get arrow schema of the Ipc File, this is faster than creating a polars schema.
    pub fn arrow_schema(&mut self) -> Result<ArrowSchema> {
        let metadata = read::read_file_metadata(&mut self.reader)?;
        Ok(metadata.schema)
    }
    /// Stop reading when `n` rows are read.
    pub fn with_n_rows(mut self, num_rows: Option<usize>) -> Self {
        self.n_rows = num_rows;
        self
    }

    /// Columns to select/ project
    pub fn with_columns(mut self, columns: Option<Vec<String>>) -> Self {
        self.columns = columns;
        self
    }

    /// Add a `row_count` column.
    pub fn with_row_count(mut self, row_count: Option<RowCount>) -> Self {
        self.row_count = row_count;
        self
    }

    /// Set the reader's column projection. This counts from 0, meaning that
    /// `vec![0, 4]` would select the 1st and 5th column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Self {
        self.projection = projection;
        self
    }

    #[cfg(feature = "lazy")]
    // todo! hoist to lazy crate
    pub fn finish_with_scan_ops(
        mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        aggregate: Option<&[ScanAggregation]>,
        projection: Option<&[usize]>,
    ) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let metadata = read::read_file_metadata(&mut self.reader)?;
        let projection = projection.map(|x| {
            let mut x = x.to_vec();
            x.sort_unstable();
            x
        });

        let schema = if let Some(projection) = &projection {
            apply_projection(&metadata.schema, projection)
        } else {
            metadata.schema.clone()
        };

        let reader = read::FileReader::new(&mut self.reader, metadata, projection);

        finish_reader(
            reader,
            rechunk,
            self.n_rows,
            predicate,
            aggregate,
            &schema,
            self.row_count,
        )
    }
}

impl<R> ArrowReader for read::FileReader<R>
where
    R: Read + Seek,
{
    fn next_record_batch(&mut self) -> ArrowResult<Option<ArrowChunk>> {
        self.next().map_or(Ok(None), |v| v.map(Some))
    }
}

impl<R> SerReader<R> for IpcReader<R>
where
    R: Read + Seek,
{
    fn new(reader: R) -> Self {
        IpcReader {
            reader,
            rechunk: true,
            n_rows: None,
            columns: None,
            projection: None,
            row_count: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> Result<DataFrame> {
        let rechunk = self.rechunk;
        let metadata = read::read_file_metadata(&mut self.reader)?;
        let schema = &metadata.schema;

        if let Some(columns) = self.columns {
            let mut prj = columns_to_projection(columns, schema)?;

            // Ipc reader panics if the projection is not in increasing order, so sorting is the safer way.
            prj.sort_unstable();
            self.projection = Some(prj);
        }

        let schema = if let Some(projection) = &self.projection {
            apply_projection(&metadata.schema, projection)
        } else {
            metadata.schema.clone()
        };

        let ipc_reader = read::FileReader::new(&mut self.reader, metadata, self.projection);
        finish_reader(
            ipc_reader,
            rechunk,
            self.n_rows,
            None,
            None,
            &schema,
            self.row_count,
        )
    }
}

/// Write a DataFrame to Arrow's IPC format
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::ipc::IpcWriter;
/// use std::fs::File;
/// use polars_io::SerWriter;
///
/// fn example(df: &mut DataFrame) -> Result<()> {
///     let mut file = File::create("file.ipc").expect("could not create file");
///
///     IpcWriter::new(&mut file)
///         .finish(df)
/// }
///
/// ```
#[must_use]
pub struct IpcWriter<W> {
    writer: W,
    compression: Option<write::Compression>,
}

use crate::aggregations::ScanAggregation;
use crate::RowCount;
use polars_core::frame::ArrowChunk;
pub use write::Compression as IpcCompression;

impl<W> IpcWriter<W> {
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<write::Compression>) -> Self {
        self.compression = compression;
        self
    }
}

impl<W> SerWriter<W> for IpcWriter<W>
where
    W: Write,
{
    fn new(writer: W) -> Self {
        IpcWriter {
            writer,
            compression: None,
        }
    }

    fn finish(mut self, df: &mut DataFrame) -> Result<()> {
        let mut ipc_writer = write::FileWriter::try_new(
            &mut self.writer,
            &df.schema().to_arrow(),
            None,
            WriteOptions {
                compression: self.compression,
            },
        )?;
        df.rechunk();
        let iter = df.iter_chunks();

        for batch in iter {
            ipc_writer.write(&batch, None)?
        }
        let _ = ipc_writer.finish()?;
        Ok(())
    }
}

#[must_use]
#[cfg(feature = "partition")]
/// Write a DataFrame to partitioned Arrow's IPC format
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_core::df;
/// use polars_io::ipc::PartitionIpcWriter;
///
/// let df = df!("a" => [1, 1, 2, 3], "b" => [2, 2, 3, 4], "c" => [2, 3, 4, 5]).unwrap();
/// let by = ["a", "b"];
/// fn example(df: &DataFrame) -> Result<()> {
///     let mut file = File::create("file.ipc").expect("could not create file");
///
///     PartitionIpcWriter::new("./tmp_dir", by)
///         .finish(&df)
/// }
///
/// ```
pub struct PartitionIpcWriter<P, I, S> {
    rootdir: P,
    by: I,
    parallel: bool,
    compression: Option<write::Compression>,
    phantom: PhantomData<S>,
}

#[cfg(feature = "partition")]
impl<P, I, S> PartitionIpcWriter<P, I, S>
where
    P: Into<PathBuf>,
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    pub fn new(rootdir: P, by: I) -> Self {
        Self {
            rootdir,
            by,
            parallel: true,
            compression: None,
            phantom: PhantomData,
        }
    }

    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<write::Compression>) -> Self {
        self.compression = compression;
        self
    }

    /// Write the ipc file in parallel (default). The single threaded writer consumes less memory.
    pub fn write_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn finish(self, df: &DataFrame) -> Result<()> {
        use polars_core::POOL;
        use rayon::iter::IntoParallelIterator;
        use rayon::iter::ParallelIterator;
        use uuid::Uuid;

        let rootdir: PathBuf = self.rootdir.into();
        let by: Vec<String> = self.by.into_iter().map(|x| x.as_ref().to_owned()).collect();
        let group = df.groupby(&by)?;

        if self.parallel {
            POOL.install(|| {
                group
                    .get_groups()
                    .idx_ref()
                    .into_par_iter()
                    .map(|t| {
                        let mut partition_df =
                            unsafe { df.take_iter_unchecked(t.1.iter().map(|i| *i as usize)) };
                        let mut path = resolve_partition_dir(&rootdir, &by, &partition_df);
                        std::fs::create_dir_all(&path)?;

                        path.push(format!("part-{}.ipc", Uuid::new_v4().to_string()));

                        let mut file = std::fs::File::create(path)?;
                        let writer = BufWriter::new(&mut file);

                        IpcWriter::new(writer)
                            .with_compression(self.compression)
                            .finish(&mut partition_df)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
        } else {
            for t in group.get_groups().idx_ref().iter() {
                let mut partition_df =
                    unsafe { df.take_iter_unchecked(t.1.iter().map(|i| *i as usize)) };
                let mut path = resolve_partition_dir(&rootdir, &by, &partition_df);
                std::fs::create_dir_all(&path)?;

                path.push(format!("part-{}.ipc", Uuid::new_v4().to_string()));

                let mut file = std::fs::File::create(path)?;
                let writer = BufWriter::new(&mut file);

                IpcWriter::new(writer)
                    .with_compression(self.compression)
                    .finish(&mut partition_df)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use arrow::io::ipc::write;
    use polars_core::df;
    use polars_core::prelude::*;
    use std::io::Cursor;

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
        assert!(df.frame_equal(&df_read));
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
        df_read.frame_equal(&expected);
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
        assert_eq!(df_read.shape(), (3, 2));
        df_read.frame_equal(&expected);
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
            IpcWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&mut df)
                .expect("ipc writer");
            buf.set_position(0);

            let df_read = IpcReader::new(buf)
                .finish()
                .expect(&format!("IPC reader: {:?}", compression));
            assert!(df.frame_equal(&df_read));
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
        assert!(df.frame_equal(&df_read));
    }

    #[test]
    #[cfg(feature = "partition")]
    fn test_partition() -> Result<()> {
        use std::{io::BufReader, path::PathBuf};

        let df = df!("a" => [1, 1, 2, 3], "b" => [2, 2, 3, 4], "c" => [2, 3, 4, 5]).unwrap();
        let by = ["a", "b"];
        let rootdir = format!("tmp-{}", uuid::Uuid::new_v4().to_string());
        PartitionIpcWriter::new(&rootdir, by).finish(&df)?;

        let expected_dfs = [
            df!("a" => [1, 1], "b" => [2, 2], "c" => [2, 3])?,
            df!("a" => [2], "b" => [3], "c" => [4])?,
            df!("a" => [3], "b" => [4], "c" => [5])?,
        ];

        let expected: Vec<(PathBuf, DataFrame)> = ["a=1/b=2", "a=2/b=3", "a=3/b=4"]
            .into_iter()
            .zip(expected_dfs.into_iter())
            .map(|(p, df)| (PathBuf::from(format!("{}/{}", &rootdir, p)), df))
            .collect();

        for (expected_dir, expected_df) in expected.iter() {
            assert!(expected_dir.exists());

            let ipc_paths = std::fs::read_dir(&expected_dir)?
                .map(|e| {
                    let entry = e?;
                    Ok(entry.path())
                })
                .collect::<Result<Vec<_>>>()?;

            assert_eq!(ipc_paths.len(), 1);
            let reader = BufReader::new(std::fs::File::open(&ipc_paths[0])?);
            let df = IpcReader::new(reader).finish()?;
            assert!(expected_df.frame_equal(&df));
        }

        std::fs::remove_dir_all(&rootdir)?;

        Ok(())
    }
}

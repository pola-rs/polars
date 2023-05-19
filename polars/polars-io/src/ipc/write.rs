use std::io::Write;
use std::path::PathBuf;

use arrow::io::ipc::write;
use arrow::io::ipc::write::WriteOptions;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;
use crate::WriterFactory;

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
/// fn example(df: &mut DataFrame) -> PolarsResult<()> {
///     let mut file = File::create("file.ipc").expect("could not create file");
///
///     IpcWriter::new(&mut file)
///         .finish(df)
/// }
///
/// ```
#[must_use]
pub struct IpcWriter<W> {
    pub(super) writer: W,
    pub(super) compression: Option<IpcCompression>,
}

impl<W: Write> IpcWriter<W> {
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<IpcCompression>) -> Self {
        self.compression = compression;
        self
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let mut writer = write::FileWriter::new(
            self.writer,
            schema.to_arrow(),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        );
        writer.start()?;

        Ok(BatchedWriter { writer })
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

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        let mut ipc_writer = write::FileWriter::try_new(
            &mut self.writer,
            df.schema().to_arrow(),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        )?;
        df.align_chunks();
        let iter = df.iter_chunks();

        for batch in iter {
            ipc_writer.write(&batch, None)?
        }
        ipc_writer.finish()?;
        Ok(())
    }
}

pub struct BatchedWriter<W: Write> {
    writer: write::FileWriter<W>,
}

impl<W: Write> BatchedWriter<W> {
    /// Write a batch to the parquet writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let iter = df.iter_chunks();
        for batch in iter {
            self.writer.write(&batch, None)?
        }
        Ok(())
    }

    /// Writes the footer of the IPC file.
    pub fn finish(&mut self) -> PolarsResult<()> {
        self.writer.finish()?;
        Ok(())
    }
}

/// Compression codec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IpcCompression {
    /// LZ4 (framed)
    LZ4,
    /// ZSTD
    #[default]
    ZSTD,
}

impl From<IpcCompression> for write::Compression {
    fn from(value: IpcCompression) -> Self {
        match value {
            IpcCompression::LZ4 => write::Compression::LZ4,
            IpcCompression::ZSTD => write::Compression::ZSTD,
        }
    }
}

pub struct IpcWriterOption {
    compression: Option<IpcCompression>,
    extension: PathBuf,
}

impl IpcWriterOption {
    pub fn new() -> Self {
        Self {
            compression: None,
            extension: PathBuf::from(".ipc"),
        }
    }

    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<IpcCompression>) -> Self {
        self.compression = compression;
        self
    }

    /// Set the extension. Defaults to ".ipc".
    pub fn with_extension(mut self, extension: PathBuf) -> Self {
        self.extension = extension;
        self
    }
}

impl Default for IpcWriterOption {
    fn default() -> Self {
        Self::new()
    }
}

impl WriterFactory for IpcWriterOption {
    fn create_writer<W: Write + 'static>(&self, writer: W) -> Box<dyn SerWriter<W>> {
        Box::new(IpcWriter::new(writer).with_compression(self.compression))
    }

    fn extension(&self) -> PathBuf {
        self.extension.to_owned()
    }
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use polars_core::df;
    use polars_core::prelude::*;

    use crate::prelude::*;

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
        df_read.frame_equal(&expected);

        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
        let mut df = df![
            "letters" => ["x", "y", "z"],
            "ints" => [123, 456, 789],
            "floats" => [4.5, 10.0, 10.0],
            "other" => ["misc", "other", "value"],
        ]
        .unwrap();
        IpcWriter::new(&mut buf)
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
        assert!(df_read.frame_equal(&expected));
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
}

use std::io::Write;
use std::path::PathBuf;

use arrow::io::ipc::write;
use arrow::io::ipc::write::WriteOptions;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;
use crate::shared::WriterFactory;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpcWriterOptions {
    /// Data page compression
    pub compression: Option<IpcCompression>,
    /// maintain the order the data was processed
    pub maintain_order: bool,
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
    /// Polars' flavor of arrow. This might be temporary.
    pub(super) pl_flavor: bool,
}

impl<W: Write> IpcWriter<W> {
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<IpcCompression>) -> Self {
        self.compression = compression;
        self
    }

    pub fn with_pl_flavor(mut self, pl_flavor: bool) -> Self {
        self.pl_flavor = pl_flavor;
        self
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let mut writer = write::FileWriter::new(
            self.writer,
            Arc::new(schema.to_arrow(self.pl_flavor)),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        );
        writer.start()?;

        Ok(BatchedWriter {
            writer,
            pl_flavor: self.pl_flavor,
        })
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
            pl_flavor: false,
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        let mut ipc_writer = write::FileWriter::try_new(
            &mut self.writer,
            Arc::new(df.schema().to_arrow(self.pl_flavor)),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        )?;
        df.align_chunks();
        let iter = df.iter_chunks(self.pl_flavor);

        for batch in iter {
            ipc_writer.write(&batch, None)?
        }
        ipc_writer.finish()?;
        Ok(())
    }
}

pub struct BatchedWriter<W: Write> {
    writer: write::FileWriter<W>,
    pl_flavor: bool,
}

impl<W: Write> BatchedWriter<W> {
    /// Write a batch to the parquet writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let iter = df.iter_chunks(self.pl_flavor);
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

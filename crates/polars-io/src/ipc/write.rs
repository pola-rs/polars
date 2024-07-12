use std::io::Write;

use arrow::io::ipc::write;
use arrow::io::ipc::write::WriteOptions;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;
use crate::shared::schema_to_arrow_checked;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpcWriterOptions {
    /// Data page compression
    pub compression: Option<IpcCompression>,
    /// maintain the order the data was processed
    pub maintain_order: bool,
}

impl IpcWriterOptions {
    pub fn to_writer<W: Write>(&self, writer: W) -> IpcWriter<W> {
        IpcWriter::new(writer).with_compression(self.compression)
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
    pub(super) compat_level: CompatLevel,
}

impl<W: Write> IpcWriter<W> {
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<IpcCompression>) -> Self {
        self.compression = compression;
        self
    }

    pub fn with_compat_level(mut self, compat_level: CompatLevel) -> Self {
        self.compat_level = compat_level;
        self
    }

    pub fn batched(self, schema: &Schema) -> PolarsResult<BatchedWriter<W>> {
        let schema = schema_to_arrow_checked(schema, self.compat_level, "ipc")?;
        let mut writer = write::FileWriter::new(
            self.writer,
            Arc::new(schema),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        );
        writer.start()?;

        Ok(BatchedWriter {
            writer,
            compat_level: self.compat_level,
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
            compat_level: CompatLevel::newest(),
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        let schema = schema_to_arrow_checked(&df.schema(), self.compat_level, "ipc")?;
        let mut ipc_writer = write::FileWriter::try_new(
            &mut self.writer,
            Arc::new(schema),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        )?;
        df.align_chunks();
        let iter = df.iter_chunks(self.compat_level, true);

        for batch in iter {
            ipc_writer.write(&batch, None)?
        }
        ipc_writer.finish()?;
        Ok(())
    }
}

pub struct BatchedWriter<W: Write> {
    writer: write::FileWriter<W>,
    compat_level: CompatLevel,
}

impl<W: Write> BatchedWriter<W> {
    /// Write a batch to the parquet writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let iter = df.iter_chunks(self.compat_level, true);
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

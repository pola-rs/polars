use std::io::Write;

use arrow::datatypes::Metadata;
use arrow::io::ipc::write::{self, EncodedData, WriteOptions};
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;
use crate::shared::schema_to_arrow_checked;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IpcWriterOptions {
    /// Data page compression
    pub compression: Option<IpcCompression>,
    /// Compatibility level
    pub compat_level: CompatLevel,
    /// Size of each written chunk.
    pub chunk_size: IdxSize,
}

impl Default for IpcWriterOptions {
    fn default() -> Self {
        Self {
            compression: None,
            compat_level: CompatLevel::newest(),
            chunk_size: 1 << 18,
        }
    }
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
///     let mut writer = IpcWriter::new(&mut file);
///
///     let custom_metadata = [
///         ("first_name".into(), "John".into()),
///         ("last_name".into(), "Doe".into()),
///     ]
///     .into_iter()
///     .collect();
///     writer.set_custom_schema_metadata(Arc::new(custom_metadata));
///     writer.finish(df)
/// }
///
/// ```
#[must_use]
pub struct IpcWriter<W> {
    pub(super) writer: W,
    pub(super) compression: Option<IpcCompression>,
    /// Polars' flavor of arrow. This might be temporary.
    pub(super) compat_level: CompatLevel,
    pub(super) parallel: bool,
    pub(super) custom_schema_metadata: Option<Arc<Metadata>>,
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

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
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

    /// Sets custom schema metadata. Must be called before `start` is called
    pub fn set_custom_schema_metadata(&mut self, custom_metadata: Arc<Metadata>) {
        self.custom_schema_metadata = Some(custom_metadata);
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
            parallel: true,
            custom_schema_metadata: None,
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        let schema = schema_to_arrow_checked(df.schema(), self.compat_level, "ipc")?;
        let mut ipc_writer = write::FileWriter::try_new(
            &mut self.writer,
            Arc::new(schema),
            None,
            WriteOptions {
                compression: self.compression.map(|c| c.into()),
            },
        )?;
        if let Some(custom_metadata) = &self.custom_schema_metadata {
            ipc_writer.set_custom_schema_metadata(Arc::clone(custom_metadata));
        }

        if self.parallel {
            df.align_chunks_par();
        } else {
            df.align_chunks();
        }
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
    /// Write a batch to the ipc writer.
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

    /// Write a encoded data to the ipc writer.
    ///
    /// # Panics
    /// The caller must ensure the chunks in the given [`DataFrame`] are aligned.
    pub fn write_encoded(
        &mut self,
        dictionaries: &[EncodedData],
        message: &EncodedData,
    ) -> PolarsResult<()> {
        self.writer.write_encoded(dictionaries, message)?;
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

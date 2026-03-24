use std::io::Write;
use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::prelude::{CsvWriterOptions, ExternalCompression, UTF8_BOM, csv_header};
use polars_io::utils::compression::CompressedWriter;
#[cfg(feature = "cloud")]
use polars_io::utils::file::{AsyncDynWriteable, AsyncWriteable};
#[cfg(feature = "cloud")]
use tokio::io::AsyncWriteExt as _;

use crate::async_executor;
use crate::nodes::io_sinks::components::sink_morsel::SinkMorselPermit;
use crate::nodes::io_sinks::writers::csv::morsel_serializer::MorselSerializer;
use crate::nodes::io_sinks::writers::interface::FileOpenTaskHandle;

pub struct IOWriter {
    pub file: FileOpenTaskHandle,
    pub filled_serializer_rx: tokio::sync::mpsc::Receiver<(
        async_executor::AbortOnDropHandle<PolarsResult<MorselSerializer>>,
        SinkMorselPermit,
    )>,
    pub reuse_serializer_tx: tokio::sync::mpsc::Sender<MorselSerializer>,
    pub schema: SchemaRef,
    pub options: Arc<CsvWriterOptions>,
}

impl IOWriter {
    pub async fn run(self) -> PolarsResult<()> {
        #[cfg(feature = "cloud")]
        {
            self.run_cloud().await
        }
        #[cfg(not(feature = "cloud"))]
        {
            self.run_without_cloud().await
        }
    }

    #[cfg(feature = "cloud")]
    async fn run_cloud(self) -> PolarsResult<()> {
        let Self {
            file,
            mut filled_serializer_rx,
            reuse_serializer_tx,
            schema,
            options,
        } = self;

        let (writable, sync_on_close) = file.await?;

        let mut writer = match options.compression {
            // Natively convert into `AsyncWriteable` to allow native async optimizations.
            ExternalCompression::Uncompressed => writable.try_into_async_writeable()?,
            // Our compression encoders only offer sync `io::Write` capabilities, so we wrap them in
            // `task::block_in_place` provided by `AsyncDynWriteable`. In theory this could
            // bottleneck the pipeline if there are a large number of files being written into in
            // parallel, since the tokio thread-pool is smaller than the computation thread-pool.
            ExternalCompression::Gzip { level } => AsyncWriteable::Dyn(AsyncDynWriteable(
                Box::new(CompressedWriter::gzip(writable, level)),
            )),
            ExternalCompression::Zstd { level } => AsyncWriteable::Dyn(AsyncDynWriteable(
                Box::new(CompressedWriter::zstd(writable, level)?),
            )),
        };

        if options.include_bom {
            writer.write_all(&UTF8_BOM).await?;
        }

        if options.include_header {
            let names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
            writer
                .write_all(&csv_header(names.as_slice(), &options.serialize_options)?)
                .await?;
        }

        while let Some((handle, permit)) = filled_serializer_rx.recv().await {
            let mut serializer = handle.await?;

            writer
                .write_all_owned(&mut serializer.serialized_data)
                .await?;

            drop(permit);

            let _ = reuse_serializer_tx.send(serializer).await;
        }

        writer.close(sync_on_close).await?;

        Ok(())
    }

    #[cfg(not(feature = "cloud"))]
    async fn run_without_cloud(self) -> PolarsResult<()> {
        let Self {
            file,
            mut filled_serializer_rx,
            reuse_serializer_tx,
            schema,
            options,
        } = self;

        let (mut writable, sync_on_close) = file.await?;

        match options.compression {
            ExternalCompression::Uncompressed => {
                let mut w = writable.as_buffered();
                if options.include_bom {
                    tokio::task::block_in_place(|| w.write_all(&UTF8_BOM))?;
                }
                if options.include_header {
                    let names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
                    let header = csv_header(names.as_slice(), &options.serialize_options)?;
                    tokio::task::block_in_place(|| w.write_all(&header))?;
                }
                while let Some((handle, permit)) = filled_serializer_rx.recv().await {
                    let mut serializer = handle.await?;
                    tokio::task::block_in_place(|| w.write_all(&serializer.serialized_data))?;
                    drop(permit);
                    let _ = reuse_serializer_tx.send(serializer).await;
                }
                drop(w);
                writable.close(sync_on_close)?;
            },
            ExternalCompression::Gzip { level } => {
                let cw = CompressedWriter::gzip(writable, level);
                let mut file = polars_io::utils::file::Writeable::Dyn(Box::new(cw));
                let mut w = file.as_buffered();
                if options.include_bom {
                    tokio::task::block_in_place(|| w.write_all(&UTF8_BOM))?;
                }
                if options.include_header {
                    let names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
                    let header = csv_header(names.as_slice(), &options.serialize_options)?;
                    tokio::task::block_in_place(|| w.write_all(&header))?;
                }
                while let Some((handle, permit)) = filled_serializer_rx.recv().await {
                    let mut serializer = handle.await?;
                    tokio::task::block_in_place(|| w.write_all(&serializer.serialized_data))?;
                    drop(permit);
                    let _ = reuse_serializer_tx.send(serializer).await;
                }
                drop(w);
                file.close(sync_on_close)?;
            },
            ExternalCompression::Zstd { level } => {
                let cw = CompressedWriter::zstd(writable, level)?;
                let mut file = polars_io::utils::file::Writeable::Dyn(Box::new(cw));
                let mut w = file.as_buffered();
                if options.include_bom {
                    tokio::task::block_in_place(|| w.write_all(&UTF8_BOM))?;
                }
                if options.include_header {
                    let names: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();
                    let header = csv_header(names.as_slice(), &options.serialize_options)?;
                    tokio::task::block_in_place(|| w.write_all(&header))?;
                }
                while let Some((handle, permit)) = filled_serializer_rx.recv().await {
                    let mut serializer = handle.await?;
                    tokio::task::block_in_place(|| w.write_all(&serializer.serialized_data))?;
                    drop(permit);
                    let _ = reuse_serializer_tx.send(serializer).await;
                }
                drop(w);
                file.close(sync_on_close)?;
            },
        }

        Ok(())
    }
}

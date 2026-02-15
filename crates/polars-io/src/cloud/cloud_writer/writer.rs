use std::num::NonZeroUsize;
use std::sync::Arc;

use bytes::Bytes;
use polars_error::PolarsResult;

use crate::cloud::PolarsObjectStore;
use crate::cloud::cloud_writer::bufferer::BytesBufferer;
use crate::cloud::cloud_writer::internal_writer::{InternalCloudWriter, InternalCloudWriterState};
use crate::metrics::{IOMetrics, OptIOMetrics};

pub struct CloudWriter {
    writer: InternalCloudWriter,
    bufferer: BytesBufferer,
}

impl CloudWriter {
    pub fn new(
        store: PolarsObjectStore,
        path: object_store::path::Path,
        upload_chunk_size: usize,
        max_concurrency: NonZeroUsize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> Self {
        let bufferer = BytesBufferer::new(upload_chunk_size);

        Self {
            writer: InternalCloudWriter {
                store,
                path,
                max_concurrency,
                io_metrics: OptIOMetrics(io_metrics),
                state: InternalCloudWriterState::NotStarted,
            },
            bufferer,
        }
    }

    pub async fn start(&mut self) -> PolarsResult<()> {
        self.writer.start().await
    }

    pub async fn write_all_owned(&mut self, mut bytes: Bytes) -> PolarsResult<()> {
        while !bytes.is_empty() {
            self.bufferer.push_owned(&mut bytes);

            if let Some(payload) = self.bufferer.flush_full_chunk() {
                self.writer.put(payload).await?;
            }
        }

        Ok(())
    }

    pub(super) fn fill_buffer_from_slice(&mut self, bytes: &mut &[u8]) -> bool {
        self.bufferer.push_slice(bytes);
        self.bufferer.is_full()
    }

    pub(super) async fn flush_full_chunk(&mut self) -> PolarsResult<()> {
        if let Some(payload) = self.bufferer.flush_full_chunk() {
            self.writer.put(payload).await?;
        }

        Ok(())
    }

    pub(super) async fn flush(&mut self) -> PolarsResult<()> {
        if let Some(payload) = self.bufferer.flush() {
            self.writer.put(payload).await?;
        }

        assert!(self.bufferer.is_empty());

        Ok(())
    }

    pub(super) fn has_buffered_bytes(&self) -> bool {
        !self.bufferer.is_empty()
    }

    pub async fn finish(&mut self) -> PolarsResult<()> {
        if let Some(payload) = self.bufferer.flush() {
            self.writer.put(payload).await?;
        }

        assert!(self.bufferer.is_empty());

        self.writer.finish().await
    }
}

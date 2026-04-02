use std::num::NonZeroUsize;

use futures::StreamExt as _;
use futures::stream::FuturesUnordered;
use object_store::PutPayload;
use polars_error::{PolarsError, PolarsResult};
use polars_utils::async_utils::error_capture::{ErrorCapture, ErrorHandle};
use polars_utils::async_utils::tokio_handle_ext;

use crate::cloud::PolarsObjectStore;
use crate::cloud::cloud_writer::multipart_upload::PlMultipartUpload;
use crate::metrics::OptIOMetrics;

/// Cloud writer that provides the `put()` function, does not perform any buffering.
pub(super) struct InternalCloudWriter {
    pub(super) store: PolarsObjectStore,
    pub(super) path: object_store::path::Path,
    pub(super) max_concurrency: NonZeroUsize,
    pub(super) io_metrics: OptIOMetrics,
    pub(super) state: InternalCloudWriterState,
}

pub(super) enum InternalCloudWriterState {
    NotStarted,
    Started(StartedState),
    Finished,
}

type WriterState = InternalCloudWriterState;

pub(super) struct StartedState {
    multipart: PlMultipartUpload,
    tasks: FuturesUnordered<tokio_handle_ext::AbortOnDropHandle<()>>,
    error_handle: ErrorHandle<PolarsError>,
    error_capture: ErrorCapture<PolarsError>,
}

impl InternalCloudWriter {
    pub(super) async fn start(&mut self) -> PolarsResult<()> {
        if let WriterState::NotStarted = &self.state {
            let path_ref = &self.path;
            let multipart = PlMultipartUpload::new(
                self.store
                    .exec_with_rebuild_retry_on_err(|s| async move {
                        s.put_multipart_opts(path_ref, object_store::PutMultipartOptions::default())
                            .await
                    })
                    .await?,
                self.store.error_context(),
            );

            let (error_capture, error_handle) = ErrorCapture::new();

            self.state = WriterState::Started(StartedState {
                multipart,
                tasks: FuturesUnordered::new(),
                error_handle,
                error_capture,
            });
        }

        Ok(())
    }

    async fn get_or_init_started_state(&mut self) -> PolarsResult<&mut StartedState> {
        loop {
            match &self.state {
                WriterState::Started(_) => {
                    let WriterState::Started(state) = &mut self.state else {
                        unreachable!()
                    };
                    return Ok(state);
                },
                WriterState::NotStarted => self.start().await?,
                WriterState::Finished => panic!(),
            }
        }
    }

    /// Takes `self.state`, replacing with it `Finished`. Returns `None` if `self.state` is not
    /// `Started`.
    fn take_started_state(&mut self) -> Option<StartedState> {
        if !matches!(&self.state, WriterState::Started(_)) {
            return None;
        }

        let WriterState::Started(state) = std::mem::replace(&mut self.state, WriterState::Finished)
        else {
            unreachable!()
        };

        Some(state)
    }

    pub(super) async fn put(&mut self, payload: PutPayload) -> PolarsResult<()> {
        let io_metrics = self.io_metrics.clone();
        let max_concurrency = self.max_concurrency.get();

        let state = self.get_or_init_started_state().await?;

        if state.error_handle.has_errored() {
            let state = self.take_started_state().unwrap();
            return Err(state.error_handle.join().await.unwrap_err());
        }

        while state.tasks.len() >= max_concurrency {
            state.tasks.next().await;
        }

        let num_bytes = payload.content_length() as u64;
        let upload_fut = state.multipart.put(payload);

        let fut = async move { io_metrics.record_bytes_tx(num_bytes, upload_fut).await };

        let handle = tokio_handle_ext::AbortOnDropHandle(tokio::spawn(
            state.error_capture.clone().wrap_future(fut),
        ));

        state.tasks.push(handle);

        Ok(())
    }

    pub(super) async fn finish(&mut self) -> PolarsResult<()> {
        let Some(StartedState {
            mut multipart,
            tasks,
            error_handle,
            error_capture,
        }) = self.take_started_state()
        else {
            return Ok(());
        };

        drop(error_capture);
        error_handle.join().await?;

        for handle in tasks {
            handle.await.unwrap();
        }

        multipart.finish().await?;

        Ok(())
    }
}

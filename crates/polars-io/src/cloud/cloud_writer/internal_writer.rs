use std::num::NonZeroUsize;

use futures::StreamExt as _;
use futures::stream::FuturesUnordered;
use object_store::PutPayload;
use polars_async::primitives::wait_group::WaitToken;
use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult, polars_bail};
use polars_utils::async_utils::error_capture::{ErrorCapture, ErrorHandle};
use polars_utils::async_utils::tokio_handle_ext;

use crate::cloud::PolarsObjectStore;
use crate::cloud::cloud_writer::multipart_upload::PlMultipartUpload;
use crate::cloud::cloud_writer::{CLEANUP_TIMEOUT, hold, run_to_completion};
use crate::metrics::OptIOMetrics;

/// Cloud writer that provides the `put()` function, does not perform any buffering.
pub(super) struct InternalCloudWriter {
    pub(super) store: PolarsObjectStore,
    pub(super) path: object_store::path::Path,
    pub(super) max_concurrency: NonZeroUsize,
    pub(super) io_metrics: OptIOMetrics,
    pub(super) state: InternalCloudWriterState,
    /// Held by the upload's tasks, so that a waiter sees them stopped.
    pub(super) wait_token: Option<WaitToken>,
}

pub(super) enum InternalCloudWriterState {
    NotStarted,
    Started(StartedState),
    Finished,
}

type WriterState = InternalCloudWriterState;

pub(super) struct StartedState {
    // Dropped before `multipart`, so that no part starts after the abort.
    tasks: FuturesUnordered<tokio_handle_ext::AbortOnDropHandle<()>>,
    multipart: PlMultipartUpload,
    error_handle: ErrorHandle<PolarsError>,
    error_capture: ErrorCapture<PolarsError>,
}

impl StartedState {
    /// Lets the part uploads in flight finish, then aborts the upload. Returns the first part
    /// upload error.
    async fn abort(self) -> PolarsResult<()> {
        let StartedState {
            tasks,
            multipart,
            error_handle,
            error_capture,
        } = self;

        drop(error_capture);
        // `error_handle` stays alive meanwhile, as dropping it stops the part uploads.
        let drained = tokio::time::timeout(CLEANUP_TIMEOUT, tasks.collect::<Vec<_>>()).await;
        if drained.is_err() && polars_core::config::verbose() {
            eprintln!(
                "[CloudWriter]: aborting with part uploads in flight, which may still be stored"
            );
        }
        let result = error_handle.join().await;
        multipart.abort().await;
        result
    }
}

impl Drop for InternalCloudWriter {
    fn drop(&mut self) {
        if let Some(state) = self.take_started_state() {
            ASYNC.spawn(hold(self.wait_token.clone(), state.abort()));
        }
    }
}

impl InternalCloudWriter {
    async fn start(&mut self) -> PolarsResult<()> {
        if let WriterState::NotStarted = &self.state {
            let store = self.store.clone();
            let path = self.path.clone();
            let wait_token = self.wait_token.clone();
            // An upload created after this writer was dropped is aborted with the task's output.
            let multipart = run_to_completion(self.wait_token.clone(), async move {
                let path_ref = &path;
                let inner = store
                    .exec_with_rebuild_retry_on_err(|s| async move {
                        s.put_multipart_opts(path_ref, object_store::PutMultipartOptions::default())
                            .await
                    })
                    .await?;
                Ok(PlMultipartUpload::new(
                    inner,
                    store.error_context(),
                    wait_token,
                ))
            })
            .await?;

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
                WriterState::Finished => {
                    polars_bail!(ComputeError: "cannot write to cloud writer after it finished or failed")
                },
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
        let wait_token = self.wait_token.clone();

        let state = self.get_or_init_started_state().await?;

        if state.error_handle.has_errored() {
            let state = self.take_started_state().unwrap();
            return run_to_completion(self.wait_token.clone(), state.abort()).await;
        }

        while state.tasks.len() >= max_concurrency {
            state.tasks.next().await;
        }

        let num_bytes = payload.content_length() as u64;
        let upload_fut = state.multipart.put(payload);

        let fut = async move { io_metrics.record_bytes_tx(num_bytes, upload_fut).await };

        let handle = tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn(hold(
            wait_token,
            state.error_capture.clone().wrap_future(fut),
        )));

        state.tasks.push(handle);

        Ok(())
    }

    /// Uploads `payload` as the entire object using a single `PUT` request.
    async fn put_single(&mut self, payload: PutPayload) -> PolarsResult<()> {
        self.state = WriterState::Finished;

        let store = self.store.clone();
        let path = self.path.clone();
        let io_metrics = self.io_metrics.clone();
        let num_bytes = payload.content_length() as u64;

        run_to_completion(self.wait_token.clone(), async move {
            let path_ref = &path;
            let upload_fut = store.exec_with_rebuild_retry_on_err(|s| {
                let payload = payload.clone();
                async move {
                    s.put_opts(path_ref, payload, object_store::PutOptions::default())
                        .await
                }
            });

            io_metrics.record_bytes_tx(num_bytes, upload_fut).await?;

            Ok(())
        })
        .await
    }

    /// Uploads `tail` and completes the object, via a single `PUT` if multipart never started.
    pub(super) async fn finish(&mut self, tail: Option<PutPayload>) -> PolarsResult<()> {
        if let WriterState::NotStarted = &self.state {
            return self.put_single(tail.unwrap_or_default()).await;
        }

        if let Some(payload) = tail {
            self.put(payload).await?;
        }

        // In place, so that a drop meanwhile still aborts the upload.
        if let WriterState::Started(state) = &mut self.state {
            while !state.error_handle.has_errored()
                && let Some(result) = state.tasks.next().await
            {
                result.unwrap();
            }
        }

        let Some(state) = self.take_started_state() else {
            return Ok(());
        };

        if state.error_handle.has_errored() {
            return run_to_completion(self.wait_token.clone(), state.abort()).await;
        }

        let StartedState { mut multipart, .. } = state;

        run_to_completion(self.wait_token.clone(), async move {
            multipart.finish().await?;
            Ok(())
        })
        .await
    }
}

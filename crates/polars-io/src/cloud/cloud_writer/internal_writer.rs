use std::num::NonZeroUsize;

use futures::StreamExt as _;
use futures::stream::FuturesUnordered;
use object_store::PutPayload;
use polars_core::runtime::ASYNC;
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
    /// Holds the initial payload. If finish() is called before a second payload arrives,
    /// a single direct `PUT` is issued instead of starting a multipart upload.
    FirstPut(PutPayload),
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
        if matches!(
            &self.state,
            WriterState::NotStarted | WriterState::FirstPut(_)
        ) {
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

            let old_state = std::mem::replace(
                &mut self.state,
                WriterState::Started(StartedState {
                    multipart,
                    tasks: FuturesUnordered::new(),
                    error_handle,
                    error_capture,
                }),
            );

            // If there was a buffered first payload, upload it as the first multipart chunk
            if let WriterState::FirstPut(first_payload) = old_state {
                self.put_into_started(first_payload).await?;
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    async fn get_or_init_started_state(&mut self) -> PolarsResult<&mut StartedState> {
        loop {
            match &self.state {
                WriterState::Started(_) => {
                    let WriterState::Started(state) = &mut self.state else {
                        unreachable!()
                    };
                    return Ok(state);
                },
                WriterState::NotStarted | WriterState::FirstPut(_) => self.start().await?,
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

    /// Dispatches a payload directly when `self.state` is guaranteed to be `Started`.
    async fn put_into_started(&mut self, payload: PutPayload) -> PolarsResult<()> {
        let WriterState::Started(state) = &mut self.state else {
            panic!("Expected Started state");
        };

        let io_metrics = self.io_metrics.clone();
        let max_concurrency = self.max_concurrency.get();

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

        let handle = tokio_handle_ext::AbortOnDropHandle(
            ASYNC.spawn(state.error_capture.clone().wrap_future(fut)),
        );

        state.tasks.push(handle);

        Ok(())
    }

    pub(super) async fn put(&mut self, payload: PutPayload) -> PolarsResult<()> {
        match &self.state {
            WriterState::NotStarted => {
                self.state = WriterState::FirstPut(payload);
                Ok(())
            },
            WriterState::FirstPut(_) => {
                self.start().await?;
                self.put_into_started(payload).await
            },
            WriterState::Started(_) => self.put_into_started(payload).await,
            WriterState::Finished => panic!("Cannot put on finished InternalCloudWriter"),
        }
    }

    pub(super) async fn finish(&mut self) -> PolarsResult<()> {
        let state = std::mem::replace(&mut self.state, WriterState::Finished);

        match state {
            WriterState::NotStarted => Ok(()),
            WriterState::FirstPut(payload) => {
                let path_ref = &self.path;
                let io_metrics = self.io_metrics.clone();
                let num_bytes = payload.content_length() as u64;

                let put_fut = self.store.exec_with_rebuild_retry_on_err(|s| {
                    let payload = payload.clone();
                    async move {
                        s.put_opts(path_ref, payload, object_store::PutOptions::default())
                            .await
                    }
                });

                let upload_fut = async move { put_fut.await.map_err(PolarsError::from) };

                io_metrics.record_bytes_tx(num_bytes, upload_fut).await?;
                Ok(())
            },
            WriterState::Started(StartedState {
                mut multipart,
                tasks,
                error_handle,
                error_capture,
            }) => {
                drop(error_capture);
                error_handle.join().await?;

                for handle in tasks {
                    handle.await.unwrap();
                }

                multipart.finish().await?;
                Ok(())
            },
            WriterState::Finished => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use object_store::PutPayload;
    use object_store::path::Path;

    use super::*;
    use crate::cloud::object_store_setup::build_object_store;

    #[tokio::test]
    async fn test_single_put_buffering() -> PolarsResult<()> {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_single_put.txt");
        let url_str = format!("file://{}", file_path.to_str().unwrap());

        let (_, polars_store) = build_object_store(url_str.as_str().into(), None, false).await?;

        let path = Path::from(file_path.to_str().unwrap());

        let mut writer = InternalCloudWriter {
            store: polars_store,
            path: path.clone(),
            max_concurrency: NonZeroUsize::new(2).unwrap(),
            io_metrics: OptIOMetrics(None),
            state: InternalCloudWriterState::NotStarted,
        };

        let payload = PutPayload::from("hello single put");
        writer.put(payload).await?;

        // Before finish(), state should be FirstPut
        assert!(matches!(
            writer.state,
            InternalCloudWriterState::FirstPut(_)
        ));

        writer.finish().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_transition_to_multipart() -> PolarsResult<()> {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_multipart.txt");
        let url_str = format!("file://{}", file_path.to_str().unwrap());

        let (_, polars_store) = build_object_store(url_str.as_str().into(), None, false).await?;

        let path = Path::from(file_path.to_str().unwrap());

        let mut writer = InternalCloudWriter {
            store: polars_store,
            path: path.clone(),
            max_concurrency: NonZeroUsize::new(2).unwrap(),
            io_metrics: OptIOMetrics(None),
            state: InternalCloudWriterState::NotStarted,
        };

        // First put -> FirstPut state
        writer.put(PutPayload::from("chunk 1")).await?;
        assert!(matches!(
            writer.state,
            InternalCloudWriterState::FirstPut(_)
        ));

        // Second put -> Should escalate to Started state
        writer.put(PutPayload::from("chunk 2")).await?;
        assert!(matches!(writer.state, InternalCloudWriterState::Started(_)));

        writer.finish().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_empty_finish() -> PolarsResult<()> {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_empty.txt");
        let url_str = format!("file://{}", file_path.to_str().unwrap());

        let (_, polars_store) = build_object_store(url_str.as_str().into(), None, false).await?;

        let path = Path::from(file_path.to_str().unwrap());

        let mut writer = InternalCloudWriter {
            store: polars_store,
            path: path.clone(),
            max_concurrency: NonZeroUsize::new(2).unwrap(),
            io_metrics: OptIOMetrics(None),
            state: InternalCloudWriterState::NotStarted,
        };

        // Calling finish immediately without putting data should return Ok(())
        writer.finish().await?;
        assert!(matches!(writer.state, InternalCloudWriterState::Finished));

        Ok(())
    }
}

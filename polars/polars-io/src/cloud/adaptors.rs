//! Interface with the object_store crate and define AsyncSeek, AsyncRead.
//! This is used, for example, by the parquet2 crate.
use std::io::{self};
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::lock::Mutex;
use futures::{AsyncRead, AsyncSeek, Future, TryFutureExt};
use object_store::path::Path;
use object_store::ObjectStore;

type OptionalFuture = Arc<Mutex<Option<BoxFuture<'static, std::io::Result<Vec<u8>>>>>>;

/// Adaptor to translate from AsyncSeek and AsyncRead to the object_store get_range API.
pub struct CloudReader {
    // The current position in the stream, it is set by seeking and updated by reading bytes.
    pos: u64,
    // The total size of the object is required when seeking from the end of the file.
    length: Option<u64>,
    // Hold an reference to the store in a thread safe way.
    object_store: Arc<Mutex<Box<dyn ObjectStore>>>,
    // The path in the object_store of the current object being read.
    path: Path,
    // If a read is pending then `active` will point to its future.
    active: OptionalFuture,
}

impl CloudReader {
    pub fn new(
        length: Option<u64>,
        object_store: Arc<Mutex<Box<dyn ObjectStore>>>,
        path: Path,
    ) -> Self {
        Self {
            pos: 0,
            length,
            object_store,
            path,
            active: Arc::new(Mutex::new(None)),
        }
    }

    /// For each read request we create a new future.
    async fn read_operation(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        length: usize,
    ) -> std::task::Poll<std::io::Result<Vec<u8>>> {
        let start = self.pos as usize;

        // If we already have a future just poll it.
        if let Some(fut) = self.active.lock().await.as_mut() {
            return Future::poll(fut.as_mut(), cx);
        }

        // Create the future.
        let future = {
            let path = self.path.clone();
            let arc = self.object_store.clone();
            // Use an async move block to get our owned objects.
            async move {
                let object_store = arc.lock().await;
                object_store
                    .get_range(&path, start..start + length)
                    .map_ok(|r| r.to_vec())
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("object store error {e:?}"),
                        )
                    })
                    .await
            }
        };
        // Prepare for next read.
        self.pos += length as u64;

        let mut future = Box::pin(future);

        // Need to poll it once to get the pump going.
        let polled = Future::poll(future.as_mut(), cx);

        // Save for next time.
        let mut state = self.active.lock().await;
        *state = Some(future);
        polled
    }
}

impl AsyncRead for CloudReader {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        // Use block_on in order to get the future result in this thread and copy the data in the output buffer.
        // With this approach we keep ownership of the buffer and we don't have to pass it to the future runtime.
        match block_on(self.read_operation(cx, buf.len())) {
            Poll::Ready(Ok(bytes)) => {
                buf.copy_from_slice(&bytes);
                Poll::Ready(Ok(bytes.len()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl AsyncSeek for CloudReader {
    fn poll_seek(
        mut self: Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
        pos: io::SeekFrom,
    ) -> std::task::Poll<std::io::Result<u64>> {
        match pos {
            io::SeekFrom::Start(pos) => self.pos = pos,
            io::SeekFrom::End(pos) => {
                let length = self.length.ok_or::<io::Error>(io::Error::new(
                    std::io::ErrorKind::Other,
                    "Cannot seek from end of stream when length is unknown.",
                ))?;
                self.pos = (length as i64 + pos) as u64
            }
            io::SeekFrom::Current(pos) => self.pos = (self.pos as i64 + pos) as u64,
        };
        std::task::Poll::Ready(Ok(self.pos))
    }
}

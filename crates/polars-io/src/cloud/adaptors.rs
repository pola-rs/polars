//! Interface with the object_store crate and define AsyncSeek, AsyncRead.
//! This is used, for example, by the [parquet2] crate.
//!
//! [parquet2]: https://crates.io/crates/parquet2
use std::io::{self};
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::lock::Mutex;
use futures::{AsyncRead, AsyncSeek, Future, TryFutureExt};
use object_store::path::Path;
use object_store::{MultipartId, ObjectStore};
use polars_core::cloud::CloudOptions;
use polars_error::{PolarsError, PolarsResult};
use tokio::io::{AsyncWrite, AsyncWriteExt};

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
            },
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
            },
            io::SeekFrom::Current(pos) => self.pos = (self.pos as i64 + pos) as u64,
        };
        std::task::Poll::Ready(Ok(self.pos))
    }
}

/// Adaptor which wraps the asynchronous interface of [ObjectStore::put_multipart](https://docs.rs/object_store/latest/object_store/trait.ObjectStore.html#tymethod.put_multipart)
/// exposing a synchronous interface which implements `std::io::Write`.
///
/// This allows it to be used in sync code which would otherwise write to a simple File or byte stream,
/// such as with `polars::prelude::CsvWriter`.
pub struct CloudWriter {
    // Hold a reference to the store
    object_store: Arc<dyn ObjectStore>,
    // The path in the object_store which we want to write to
    path: Path,
    // ID of a partially-done upload, used to abort the upload on error
    multipart_id: MultipartId,
    // The Tokio runtime which the writer uses internally.
    runtime: tokio::runtime::Runtime,
    // Internal writer, constructed at creation
    writer: Box<dyn AsyncWrite + Send + Unpin>,
}

impl CloudWriter {
    /// Construct a new CloudWriter, re-using the given `object_store`
    ///
    /// Creates a new (current-thread) Tokio runtime
    /// which bridges the sync writing process with the async ObjectStore multipart uploading.
    /// TODO: Naming?
    pub fn new_with_object_store(
        object_store: Arc<dyn ObjectStore>,
        path: Path,
    ) -> PolarsResult<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();
        let build_result =
            runtime.block_on(async { Self::build_writer(&object_store, &path).await });
        match build_result {
            Err(error) => Err(PolarsError::from(error)),
            Ok((multipart_id, writer)) => Ok(CloudWriter {
                object_store,
                path,
                multipart_id,
                runtime,
                writer,
            }),
        }
    }

    /// Constructs a new CloudWriter from a path and an optional set of CloudOptions.
    ///
    /// Wrapper around `CloudWriter::new_with_object_store` that is useful if you only have a single write task.
    /// TODO: Naming?
    pub fn new(uri: &str, cloud_options: Option<&CloudOptions>) -> PolarsResult<Self> {
        let (cloud_location, object_store) = crate::cloud::build(uri, cloud_options)?;
        let object_store = Arc::from(object_store);
        Self::new_with_object_store(object_store, cloud_location.prefix.into())
    }

    async fn build_writer(
        object_store: &Arc<dyn ObjectStore>,
        path: &Path,
    ) -> object_store::Result<(MultipartId, Box<dyn AsyncWrite + Send + Unpin>)> {
        let (multipart_id, s3_writer) = object_store.put_multipart(path).await?;
        Ok((multipart_id, s3_writer))
    }

    fn abort(&self) {
        let _ = self.runtime.block_on(async {
            self.object_store
                .abort_multipart(&self.path, &self.multipart_id)
                .await
        });
    }
}

impl std::io::Write for CloudWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let res = self.runtime.block_on(self.writer.write(buf));
        if res.is_err() {
            self.abort();
        }
        res
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let res = self.runtime.block_on(self.writer.flush());
        if res.is_err() {
            self.abort();
        }
        res
    }
}

impl Drop for CloudWriter {
    fn drop(&mut self) {
        let _ = self.runtime.block_on(self.writer.shutdown());
    }
}

#[cfg(feature = "csv")]
#[cfg(test)]
mod tests {
    use object_store::ObjectStore;
    use polars_core::df;
    use polars_core::prelude::{DataFrame, NamedFrom};

    use super::*;

    fn example_dataframe() -> DataFrame {
        df!(
            "foo" => &[1, 2, 3],
            "bar" => &[None, Some("bak"), Some("baz")],
        )
        .unwrap()
    }

    #[test]
    fn csv_to_local_objectstore_cloudwriter() {
        use crate::csv::CsvWriter;
        use crate::prelude::SerWriter;

        let mut df = example_dataframe();

        let object_store: Box<dyn ObjectStore> = Box::new(
            object_store::local::LocalFileSystem::new_with_prefix(std::env::temp_dir())
                .expect("Could not initialize connection"),
        );
        let object_store: Arc<dyn ObjectStore> = Arc::from(object_store);

        let path: object_store::path::Path = "cloud_writer_example.csv".into();

        let mut cloud_writer = CloudWriter::new_with_object_store(object_store, path).unwrap();
        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write dataframe as CSV to remote location");
    }

    // Skip this tests on Windows since it does not have a convenient /tmp/ location.
    #[cfg_attr(target_os = "windows", ignore)]
    #[test]
    fn cloudwriter_from_cloudlocation_test() {
        use crate::csv::CsvWriter;
        use crate::prelude::SerWriter;

        let mut df = example_dataframe();

        let mut cloud_writer =
            CloudWriter::new("file:///tmp/cloud_writer_example2.csv", None).unwrap();

        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write dataframe as CSV to remote location");
    }
}

//! Interface with the object_store crate and define AsyncSeek, AsyncRead.

use std::sync::Arc;

use object_store::path::Path;
use object_store::{MultipartId, ObjectStore};
use polars_error::{to_compute_err, PolarsResult};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::CloudOptions;
use crate::pl_async::get_runtime;

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
    // Internal writer, constructed at creation
    writer: Box<dyn AsyncWrite + Send + Unpin>,
}

impl CloudWriter {
    /// Construct a new CloudWriter, re-using the given `object_store`
    ///
    /// Creates a new (current-thread) Tokio runtime
    /// which bridges the sync writing process with the async ObjectStore multipart uploading.
    /// TODO: Naming?
    pub async fn new_with_object_store(
        object_store: Arc<dyn ObjectStore>,
        path: Path,
    ) -> PolarsResult<Self> {
        let (multipart_id, writer) = Self::build_writer(&object_store, &path).await?;
        Ok(CloudWriter {
            object_store,
            path,
            multipart_id,
            writer,
        })
    }

    /// Constructs a new CloudWriter from a path and an optional set of CloudOptions.
    ///
    /// Wrapper around `CloudWriter::new_with_object_store` that is useful if you only have a single write task.
    /// TODO: Naming?
    pub async fn new(uri: &str, cloud_options: Option<&CloudOptions>) -> PolarsResult<Self> {
        let (cloud_location, object_store) =
            crate::cloud::build_object_store(uri, cloud_options).await?;
        Self::new_with_object_store(object_store, cloud_location.prefix.into()).await
    }

    async fn build_writer(
        object_store: &Arc<dyn ObjectStore>,
        path: &Path,
    ) -> object_store::Result<(MultipartId, Box<dyn AsyncWrite + Send + Unpin>)> {
        let (multipart_id, s3_writer) = object_store.put_multipart(path).await?;
        Ok((multipart_id, s3_writer))
    }

    async fn abort(&self) -> PolarsResult<()> {
        self.object_store
            .abort_multipart(&self.path, &self.multipart_id)
            .await
            .map_err(to_compute_err)
    }
}

impl std::io::Write for CloudWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        get_runtime().block_on(async {
            let res = self.writer.write(buf).await;
            if res.is_err() {
                let _ = self.abort().await;
            }
            res
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        get_runtime().block_on(async {
            let res = self.writer.flush().await;
            if res.is_err() {
                let _ = self.abort().await;
            }
            res
        })
    }
}

impl Drop for CloudWriter {
    fn drop(&mut self) {
        let _ = get_runtime().block_on(self.writer.shutdown());
    }
}

#[cfg(feature = "csv")]
#[cfg(test)]
mod tests {
    use polars_core::df;
    use polars_core::prelude::DataFrame;

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
        use crate::csv::write::CsvWriter;
        use crate::prelude::SerWriter;

        let mut df = example_dataframe();

        let object_store: Arc<dyn ObjectStore> = Arc::new(
            object_store::local::LocalFileSystem::new_with_prefix(std::env::temp_dir())
                .expect("Could not initialize connection"),
        );

        let path: object_store::path::Path = "cloud_writer_example.csv".into();

        let mut cloud_writer = get_runtime()
            .block_on(CloudWriter::new_with_object_store(object_store, path))
            .unwrap();
        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write DataFrame as CSV to remote location");
    }

    // Skip this tests on Windows since it does not have a convenient /tmp/ location.
    #[cfg_attr(target_os = "windows", ignore)]
    #[test]
    fn cloudwriter_from_cloudlocation_test() {
        use crate::csv::write::CsvWriter;
        use crate::prelude::SerWriter;

        let mut df = example_dataframe();

        let mut cloud_writer = get_runtime()
            .block_on(CloudWriter::new(
                "file:///tmp/cloud_writer_example2.csv",
                None,
            ))
            .unwrap();

        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write DataFrame as CSV to remote location");
    }
}

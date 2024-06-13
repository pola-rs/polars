//! Interface with the object_store crate and define AsyncSeek, AsyncRead.

use std::sync::Arc;

use object_store::path::Path;
use object_store::{MultipartUpload, ObjectStore, PutPayload};
use polars_error::{to_compute_err, PolarsResult};

use super::CloudOptions;
use crate::pl_async::get_runtime;

/// Adaptor which wraps the asynchronous interface of [ObjectStore::put_multipart](https://docs.rs/object_store/latest/object_store/trait.ObjectStore.html#tymethod.put_multipart)
/// exposing a synchronous interface which implements `std::io::Write`.
///
/// This allows it to be used in sync code which would otherwise write to a simple File or byte stream,
/// such as with `polars::prelude::CsvWriter`.
pub struct CloudWriter {
    // Internal writer, constructed at creation
    writer: Box<dyn MultipartUpload>,
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
        let writer = object_store.put_multipart(&path).await?;
        Ok(CloudWriter { writer })
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

    async fn abort(&mut self) -> PolarsResult<()> {
        self.writer.abort().await.map_err(to_compute_err)
    }
}

impl std::io::Write for CloudWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // SAFETY:
        // We extend the lifetime for the duration of this function. This is safe as well block the
        // async runtime here
        let buf = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(buf) };
        get_runtime().block_on(async {
            let res = self.writer.put_part(PutPayload::from_static(buf)).await;
            if res.is_err() {
                let _ = self.abort().await;
            }
            Ok(buf.len())
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        get_runtime().block_on(async {
            let res = self.writer.complete().await;
            if res.is_err() {
                let _ = self.abort().await;
            }
            Ok(())
        })
    }
}

impl Drop for CloudWriter {
    fn drop(&mut self) {
        let _ = get_runtime().block_on(self.writer.complete());
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

//! Interface with the object_store crate and define AsyncSeek, AsyncRead.

use std::future::Future;
use std::sync::Arc;

use object_store::buffered::BufWriter;
use object_store::path::Path;
use object_store::ObjectStore;
use polars_error::{PolarsError, PolarsResult};
use tokio::io::AsyncWriteExt;

use super::{object_path_from_str, CloudOptions};
use crate::pl_async::{get_runtime, get_upload_chunk_size};

/// CloudWriter's synchronous functions should be callable from async contexts,
/// so we ensure we are `block_in_place` using this util function.
fn block_in_place_on<T>(func: impl Future<Output = T>) -> T {
    let rt = get_runtime();
    tokio::task::block_in_place(|| rt.block_on(func))
}

/// Adaptor which wraps the interface of [ObjectStore::BufWriter] exposing a synchronous interface
/// which implements `std::io::Write`.
///
/// This allows it to be used in sync code which would otherwise write to a simple File or byte stream,
/// such as with `polars::prelude::CsvWriter`.
///
/// [ObjectStore::BufWriter]: https://docs.rs/object_store/latest/object_store/buffered/struct.BufWriter.html
pub struct CloudWriter {
    writer: BufWriter,
    last_err: Option<std::io::Error>,
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
        let writer = BufWriter::with_capacity(object_store, path, get_upload_chunk_size());
        Ok(CloudWriter {
            writer,
            last_err: None,
        })
    }

    /// Constructs a new CloudWriter from a path and an optional set of CloudOptions.
    ///
    /// Wrapper around `CloudWriter::new_with_object_store` that is useful if you only have a single write task.
    /// TODO: Naming?
    pub async fn new(uri: &str, cloud_options: Option<&CloudOptions>) -> PolarsResult<Self> {
        if let Some(local_path) = uri.strip_prefix("file://") {
            // Local paths must be created first, otherwise object store will not write anything.
            if !matches!(std::fs::exists(local_path), Ok(true)) {
                panic!(
                    "[CloudWriter] Expected local file to be created: {}",
                    local_path
                );
            }
        }

        let (cloud_location, object_store) =
            crate::cloud::build_object_store(uri, cloud_options, false).await?;
        Self::new_with_object_store(
            object_store.to_dyn_object_store().await,
            object_path_from_str(&cloud_location.prefix)?,
        )
    }

    pub fn get_buf_writer(&self) -> &BufWriter {
        &self.writer
    }

    pub fn get_buf_writer_mut(&mut self) -> &mut BufWriter {
        &mut self.writer
    }

    fn try_with_writer_sync<F, O>(&mut self, func: F) -> std::io::Result<O>
    where
        F: Fn(&mut BufWriter) -> std::io::Result<O>,
    {
        if let Some(e) = &self.last_err {
            return Err(clone_io_err(e));
        }

        match func(&mut self.writer) {
            Ok(v) => Ok(v),
            Err(e) => {
                self.last_err.replace(clone_io_err(&e));
                Err(e)
            },
        }
    }

    async fn try_with_writer<'a, F, Fut, O>(&'a mut self, func: F) -> std::io::Result<O>
    where
        F: Fn(&'a mut BufWriter) -> Fut,
        Fut: Future<Output = std::io::Result<O>> + 'a,
    {
        if let Some(e) = &self.last_err {
            return Err(clone_io_err(e));
        }

        match func(&mut self.writer).await {
            Ok(v) => Ok(v),
            Err(e) => {
                self.last_err = Some(clone_io_err(&e));
                Err(e)
            },
        }
    }

    pub async fn close(&mut self) -> PolarsResult<()> {
        self.try_with_writer(|writer| writer.shutdown())
            .await
            .map_err(PolarsError::from)?;
        self.last_err = Some(std::io::Error::new(std::io::ErrorKind::Other, "closed"));
        Ok(())
    }

    pub fn close_sync(&mut self) -> PolarsResult<()> {
        block_in_place_on(self.close())
    }
}

impl std::io::Write for CloudWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // SAFETY:
        // We extend the lifetime for the duration of this function. This is safe as we block the
        // async runtime here
        let buf = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(buf) };

        self.try_with_writer_sync(|writer| {
            block_in_place_on(async { writer.write_all(buf).await.map(|_t| buf.len()) })
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.try_with_writer_sync(|writer| block_in_place_on(async { writer.flush().await }))
    }
}

impl Drop for CloudWriter {
    fn drop(&mut self) {
        // TODO: Once we are properly calling `close()` from all contexts this can instead be a
        // debug_assert that we are in an `Err(_)` state when dropping.
        if !std::thread::panicking() && self.last_err.is_none() {
            self.close_sync().unwrap();
        }
    }
}

fn clone_io_err(e: &std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), e.to_string())
}

#[cfg(test)]
mod tests {

    use polars_core::df;
    use polars_core::prelude::DataFrame;

    fn example_dataframe() -> DataFrame {
        df!(
            "foo" => &[1, 2, 3],
            "bar" => &[None, Some("bak"), Some("baz")],
        )
        .unwrap()
    }

    #[test]
    #[cfg(feature = "csv")]
    fn csv_to_local_objectstore_cloudwriter() {
        use super::*;
        use crate::csv::write::CsvWriter;
        use crate::prelude::SerWriter;

        let mut df = example_dataframe();

        let object_store: Arc<dyn ObjectStore> = Arc::new(
            object_store::local::LocalFileSystem::new_with_prefix(std::env::temp_dir())
                .expect("Could not initialize connection"),
        );

        let path: object_store::path::Path = "cloud_writer_example.csv".into();

        let mut cloud_writer = CloudWriter::new_with_object_store(object_store, path).unwrap();
        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write DataFrame as CSV to remote location");
    }

    // Skip this tests on Windows since it does not have a convenient /tmp/ location.
    #[cfg_attr(target_os = "windows", ignore)]
    #[cfg(feature = "csv")]
    #[test]
    fn cloudwriter_from_cloudlocation_test() {
        use super::*;
        use crate::csv::write::CsvWriter;
        use crate::prelude::{CsvReadOptions, SerWriter};
        use crate::SerReader;

        let mut df = example_dataframe();

        let path = "/tmp/cloud_writer_example2.csv";

        std::fs::File::create(path).unwrap();

        let mut cloud_writer = get_runtime()
            .block_on(CloudWriter::new(format!("file://{}", path).as_str(), None))
            .unwrap();

        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write DataFrame as CSV to remote location");

        cloud_writer.close_sync().unwrap();

        assert_eq!(
            CsvReadOptions::default()
                .try_into_reader_with_file_path(Some(path.into()))
                .unwrap()
                .finish()
                .unwrap(),
            df
        );
    }
}

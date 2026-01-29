//! Interface with the object_store crate and define AsyncSeek, AsyncRead.

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::buffered::BufWriter;
use object_store::path::Path;
use polars_error::PolarsResult;
use polars_utils::pl_path::PlRefPath;
use tokio::io::AsyncWriteExt;

use super::{CloudOptions, object_path_from_str};
use crate::pl_async::get_runtime;
use crate::utils::file::WriteableTrait;

fn clone_io_err(e: &std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), e.to_string())
}

/// Adaptor which wraps the interface of [ObjectStore::BufWriter] exposing a synchronous interface
/// which implements `std::io::Write`.
///
/// This allows it to be used in sync code which would otherwise write to a simple File or byte stream,
/// such as with `polars::prelude::CsvWriter`.
///
/// [ObjectStore::BufWriter]: https://docs.rs/object_store/latest/object_store/buffered/struct.BufWriter.html
pub struct BlockingCloudWriter {
    state: std::io::Result<BufWriter>,
}

impl BlockingCloudWriter {
    /// Construct a new BlockingCloudWriter, re-using the given `object_store`
    ///
    /// Creates a new (current-thread) Tokio runtime
    /// which bridges the sync writing process with the async ObjectStore multipart uploading.
    /// TODO: Naming?
    pub fn new_with_object_store(
        object_store: Arc<dyn ObjectStore>,
        path: Path,
        cloud_upload_chunk_size: usize,
        cloud_upload_max_concurrency: usize,
    ) -> PolarsResult<Self> {
        let writer = BufWriter::with_capacity(object_store, path, cloud_upload_chunk_size)
            .with_max_concurrency(cloud_upload_max_concurrency);
        Ok(BlockingCloudWriter { state: Ok(writer) })
    }

    /// Constructs a new BlockingCloudWriter from a path and an optional set of CloudOptions.
    ///
    /// Wrapper around `BlockingCloudWriter::new_with_object_store` that is useful if you only have a single write task.
    /// TODO: Naming?
    pub async fn new(
        uri: PlRefPath,
        cloud_options: Option<&CloudOptions>,
        cloud_upload_chunk_size: usize,
        cloud_upload_max_concurrency: usize,
    ) -> PolarsResult<Self> {
        let (cloud_location, object_store) =
            crate::cloud::build_object_store(uri, cloud_options, false).await?;
        Self::new_with_object_store(
            object_store.to_dyn_object_store().await,
            object_path_from_str(&cloud_location.prefix)?,
            cloud_upload_chunk_size,
            cloud_upload_max_concurrency,
        )
    }

    /// Returns the underlying [`object_store::buffered::BufWriter`]
    pub fn try_into_inner(mut self) -> std::io::Result<BufWriter> {
        // We can't just return self.state:
        // * cannot move out of type `adaptors::BlockingCloudWriter`, which implements the `Drop` trait
        std::mem::replace(&mut self.state, Err(std::io::Error::other("")))
    }

    /// Closes the writer, or returns the existing error if it exists. After this function is called
    /// the writer is guaranteed to be in an error state.
    pub fn close(&mut self) -> std::io::Result<()> {
        match self.try_with_writer(|writer| get_runtime().block_in_place_on(writer.shutdown())) {
            Ok(_) => {
                self.state = Err(std::io::Error::other("closed"));
                Ok(())
            },
            Err(e) => Err(e),
        }
    }

    fn try_with_writer<F, O>(&mut self, func: F) -> std::io::Result<O>
    where
        F: Fn(&mut BufWriter) -> std::io::Result<O>,
    {
        let writer: &mut BufWriter = self.state.as_mut().map_err(|e| clone_io_err(e))?;
        match func(writer) {
            Ok(v) => Ok(v),
            Err(e) => {
                self.state = Err(clone_io_err(&e));
                Err(e)
            },
        }
    }
}

impl std::io::Write for BlockingCloudWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // SAFETY:
        // We extend the lifetime for the duration of this function. This is safe as we block the
        // async runtime here
        let buf = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(buf) };

        self.try_with_writer(|writer| {
            get_runtime()
                .block_in_place_on(async { writer.write_all(buf).await.map(|_t| buf.len()) })
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.try_with_writer(|writer| get_runtime().block_in_place_on(writer.flush()))
    }
}

impl WriteableTrait for BlockingCloudWriter {
    fn close(&mut self) -> std::io::Result<()> {
        BlockingCloudWriter::close(self)
    }

    fn sync_all(&self) -> std::io::Result<()> {
        Ok(())
    }

    fn sync_data(&self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for BlockingCloudWriter {
    fn drop(&mut self) {
        if self.state.is_err() {
            return;
        }

        // Note: We should not hit here - the writer should instead be explicitly closed.
        // But we still have this here as a safety measure to prevent silently dropping errors.
        match self.close() {
            Ok(()) => {},
            e @ Err(_) => {
                if std::thread::panicking() {
                    eprintln!("ERROR: CloudWriter errored on close: {e:?}")
                } else {
                    e.unwrap()
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {

    use polars_core::df;
    use polars_core::prelude::DataFrame;

    use crate::{get_upload_chunk_size, get_upload_concurrency};

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

        let mut cloud_writer = BlockingCloudWriter::new_with_object_store(
            object_store,
            path,
            get_upload_chunk_size(),
            get_upload_concurrency(),
        )
        .unwrap();
        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write DataFrame as CSV to remote location");
    }

    // Skip this tests on Windows since it does not have a convenient /tmp/ location.
    #[cfg_attr(target_os = "windows", ignore)]
    #[cfg(feature = "csv")]
    #[test]
    fn cloudwriter_from_cloudlocation_test() {
        use polars_utils::pl_path::format_file_uri;

        use super::*;
        use crate::csv::write::CsvWriter;
        use crate::prelude::{CsvReadOptions, SerWriter};
        use crate::{SerReader, get_upload_concurrency};

        let mut df = example_dataframe();

        let path = "/tmp/cloud_writer_example2.csv";

        std::fs::File::create(path).unwrap();

        let mut cloud_writer = get_runtime()
            .block_on(BlockingCloudWriter::new(
                format_file_uri(path),
                None,
                get_upload_chunk_size(),
                get_upload_concurrency(),
            ))
            .unwrap();

        CsvWriter::new(&mut cloud_writer)
            .finish(&mut df)
            .expect("Could not write DataFrame as CSV to remote location");

        cloud_writer.close().unwrap();

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

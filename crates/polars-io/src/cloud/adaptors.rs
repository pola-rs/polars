//! Interface with the object_store crate and define AsyncSeek, AsyncRead.

use std::sync::Arc;

use object_store::buffered::BufWriter;
use object_store::path::Path;
use object_store::ObjectStore;
use polars_error::{to_compute_err, PolarsResult};
use tokio::io::AsyncWriteExt;

use super::{object_path_from_str, CloudOptions};
use crate::pl_async::{get_runtime, get_upload_chunk_size};

enum WriterState {
    Open(BufWriter),
    /// Note: `Err` state is also used as the close state on success.
    Err(std::io::Error),
}

impl WriterState {
    fn try_with_writer<F, O>(&mut self, func: F) -> std::io::Result<O>
    where
        F: Fn(&mut BufWriter) -> std::io::Result<O>,
    {
        match self {
            Self::Open(writer) => match func(writer) {
                Ok(v) => Ok(v),
                Err(e) => {
                    let _ = get_runtime().block_on_potential_spawn(writer.abort());
                    *self = Self::Err(e);
                    self.try_with_writer(func)
                },
            },
            Self::Err(e) => Err(std::io::Error::new(e.kind(), e.to_string())),
        }
    }
}

/// Adaptor which wraps the interface of [ObjectStore::BufWriter] exposing a synchronous interface
/// which implements `std::io::Write`.
///
/// This allows it to be used in sync code which would otherwise write to a simple File or byte stream,
/// such as with `polars::prelude::CsvWriter`.
///
/// [ObjectStore::BufWriter]: https://docs.rs/object_store/latest/object_store/buffered/struct.BufWriter.html
pub struct CloudWriter {
    // Internal writer, constructed at creation
    inner: WriterState,
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
            inner: WriterState::Open(writer),
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

    pub fn close(&mut self) -> PolarsResult<()> {
        let WriterState::Open(writer) = &mut self.inner else {
            panic!();
        };

        get_runtime()
            .block_on_potential_spawn(async { writer.shutdown().await })
            .map_err(to_compute_err)?;

        self.inner = WriterState::Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "impl error: file was closed",
        ));

        Ok(())
    }
}

impl std::io::Write for CloudWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // SAFETY:
        // We extend the lifetime for the duration of this function. This is safe as well block the
        // async runtime here
        let buf = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(buf) };

        self.inner.try_with_writer(|writer| {
            get_runtime()
                .block_on_potential_spawn(async { writer.write_all(buf).await.map(|_t| buf.len()) })
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.try_with_writer(|writer| {
            get_runtime().block_on_potential_spawn(async { writer.flush().await })
        })
    }
}

impl Drop for CloudWriter {
    fn drop(&mut self) {
        // TODO: Properly raise this error instead of panicking.
        match self.inner {
            WriterState::Open(_) => self.close().unwrap(),
            WriterState::Err(_) => {},
        }
    }
}

#[cfg(feature = "csv")]
#[cfg(test)]
mod tests {
    use polars_core::df;
    use polars_core::prelude::DataFrame;

    use super::*;
    use crate::prelude::CsvReadOptions;
    use crate::SerReader;

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

        let mut cloud_writer = CloudWriter::new_with_object_store(object_store, path).unwrap();
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

        let path = "/tmp/cloud_writer_example2.csv";

        std::fs::File::create(path).unwrap();

        let mut cloud_writer = get_runtime()
            .block_on(CloudWriter::new(format!("file://{}", path).as_str(), None))
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

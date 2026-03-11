use std::sync::{Arc, Mutex};

use polars_core::prelude::{InitHashMaps, PlHashSet};
use polars_error::{PolarsResult, polars_bail};
use polars_io::cloud::CloudOptions;
use polars_io::metrics::IOMetrics;
use polars_io::pl_async;
use polars_io::utils::file::Writeable;
use polars_plan::dsl::file_provider::{FileProviderReturn, FileProviderType};
use polars_plan::prelude::file_provider::FileProviderArgs;
use polars_utils::pl_path::PlRefPath;

pub struct FileProvider {
    pub base_path: PlRefPath,
    pub cloud_options: Option<Arc<CloudOptions>>,
    pub provider_type: FileProviderType,
    pub upload_chunk_size: usize,
    pub upload_max_concurrency: usize,
    pub io_metrics: Option<Arc<IOMetrics>>,
    opened_paths: Mutex<PlHashSet<String>>,
}

impl FileProvider {
    pub fn new(
        base_path: PlRefPath,
        cloud_options: Option<Arc<CloudOptions>>,
        provider_type: FileProviderType,
        upload_chunk_size: usize,
        upload_max_concurrency: usize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> Self {
        Self {
            base_path,
            cloud_options,
            provider_type,
            upload_chunk_size,
            upload_max_concurrency,
            io_metrics,
            opened_paths: Mutex::new(PlHashSet::new()),
        }
    }

    pub async fn open_file(&self, args: FileProviderArgs) -> PolarsResult<Writeable> {
        let provided_path: String = match &self.provider_type {
            FileProviderType::Hive(p) => p.get_path(args)?,
            FileProviderType::Iceberg(p) => p.get_path(args)?,
            FileProviderType::Function(f) => {
                let f = f.clone();

                let out = pl_async::get_runtime()
                    .spawn_blocking(move || f.get_path_or_file(args))
                    .await
                    .unwrap()?;

                match out {
                    FileProviderReturn::Path(p) => p,
                    FileProviderReturn::Writeable(v) => return Ok(v),
                }
            },
        };

        let path = self.base_path.join(&provided_path);

        {
            let mut opened = self.opened_paths.lock().unwrap();
            if !opened.insert(provided_path.clone()) {
                polars_bail!(
                    ComputeError:
                    "tried to write to '{}', which has already been written in this sink operation. \
                     This will corrupt data. Ensure your file_path_provider returns unique output \
                     paths (consider using the `index_in_partition` argument)",
                    path
                );
            }
        }

        if !path.has_scheme()
            && let Some(path) = path.parent()
        {
            // Ignore errors from directory creation - the `Writeable::try_new()` below will raise
            // appropriate errors.
            let _ = tokio::fs::DirBuilder::new()
                .recursive(true)
                .create(path)
                .await;
        }

        Writeable::try_new(
            path,
            self.cloud_options.as_deref(),
            self.upload_chunk_size,
            self.upload_max_concurrency,
            self.io_metrics.clone(),
        )
    }
}

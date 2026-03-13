use std::sync::Arc;

use polars_error::{PolarsResult, polars_ensure};
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
}

impl FileProvider {
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

        polars_ensure!(
            path.as_str().starts_with(self.base_path.as_str()),
            ComputeError:
            "provided path '{provided_path}' is absolute but does not start with base path '{}'",
            self.base_path,
        );

        let has_parent_dir_component = provided_path
            .as_bytes()
            .split(|c| *c == b'/' || *c == b'\\')
            .any(|bytes| bytes == b"..");

        polars_ensure!(
            !has_parent_dir_component,
            ComputeError:
            "provided path '{provided_path}' contained parent dir component '..'"
        );

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

use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use once_cell::sync::Lazy;
use polars_error::{PolarsError, PolarsResult};

use super::cache::{get_env_file_cache_ttl, FILE_CACHE};
use super::entry::FileCacheEntry;
use super::file_fetcher::{CloudFileFetcher, LocalFileFetcher};
use crate::cloud::{
    build_object_store, object_path_from_str, CloudLocation, CloudOptions, PolarsObjectStore,
};
use crate::path_utils::{ensure_directory_init, is_cloud_url, POLARS_TEMP_DIR_BASE_PATH};
use crate::pl_async;

pub static FILE_CACHE_PREFIX: Lazy<Box<Path>> = Lazy::new(|| {
    let path = POLARS_TEMP_DIR_BASE_PATH
        .join("file-cache/")
        .into_boxed_path();

    if let Err(err) = ensure_directory_init(path.as_ref()) {
        panic!(
            "failed to create file cache directory: path = {}, err = {}",
            path.to_str().unwrap(),
            err
        );
    }

    path
});

pub(super) fn last_modified_u64(metadata: &std::fs::Metadata) -> u64 {
    metadata
        .modified()
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub(super) fn update_last_accessed(file: &std::fs::File) {
    let file_metadata = file.metadata().unwrap();

    if let Err(e) = file.set_times(
        std::fs::FileTimes::new()
            .set_modified(file_metadata.modified().unwrap())
            .set_accessed(std::time::SystemTime::now()),
    ) {
        panic!("failed to update file last accessed time: {}", e);
    }
}

pub fn init_entries_from_uri_list(
    uri_list: &[Arc<str>],
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Vec<Arc<FileCacheEntry>>> {
    if uri_list.is_empty() {
        return Ok(Default::default());
    }

    let first_uri = uri_list.first().unwrap().as_ref();

    let file_cache_ttl = cloud_options
        .map(|x| x.file_cache_ttl)
        .unwrap_or_else(get_env_file_cache_ttl);

    if is_cloud_url(first_uri) {
        let object_stores = pl_async::get_runtime().block_on_potential_spawn(async {
            futures::future::try_join_all(
                (0..if first_uri.starts_with("http") {
                    // Object stores for http are tied to the path.
                    uri_list.len()
                } else {
                    1
                })
                    .map(|i| async move {
                        let (_, object_store) =
                            build_object_store(&uri_list[i], cloud_options, false).await?;
                        PolarsResult::Ok(PolarsObjectStore::new(object_store))
                    }),
            )
            .await
        })?;

        uri_list
            .iter()
            .enumerate()
            .map(|(i, uri)| {
                FILE_CACHE.init_entry(
                    uri.clone(),
                    || {
                        let CloudLocation { prefix, .. } =
                            CloudLocation::new(uri.as_ref(), false).unwrap();
                        let cloud_path = object_path_from_str(&prefix)?;

                        let object_store =
                            object_stores[std::cmp::min(i, object_stores.len() - 1)].clone();
                        let uri = uri.clone();

                        Ok(Arc::new(CloudFileFetcher {
                            uri,
                            object_store,
                            cloud_path,
                        }))
                    },
                    file_cache_ttl,
                )
            })
            .collect::<PolarsResult<Vec<_>>>()
    } else {
        uri_list
            .iter()
            .map(|uri| {
                let uri = std::fs::canonicalize(uri.as_ref()).map_err(|err| {
                    let msg = Some(format!("{}: {}", err, uri.as_ref()).into());
                    PolarsError::IO {
                        error: err.into(),
                        msg,
                    }
                })?;
                let uri = Arc::<str>::from(uri.to_str().unwrap());

                FILE_CACHE.init_entry(
                    uri.clone(),
                    || Ok(Arc::new(LocalFileFetcher::from_uri(uri.clone()))),
                    file_cache_ttl,
                )
            })
            .collect::<PolarsResult<Vec<_>>>()
    }
}

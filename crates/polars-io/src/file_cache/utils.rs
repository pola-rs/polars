use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use once_cell::sync::Lazy;
use polars_error::{to_compute_err, PolarsError, PolarsResult};

use super::cache::FILE_CACHE;
use super::entry::FileCacheEntry;
use super::file_fetcher::{CloudFileFetcher, LocalFileFetcher};
use crate::cloud::{build_object_store, CloudLocation, CloudOptions, PolarsObjectStore};
use crate::pl_async;
use crate::prelude::{is_cloud_url, POLARS_TEMP_DIR_BASE_PATH};

pub(super) static FILE_CACHE_PREFIX: Lazy<Box<Path>> = Lazy::new(|| {
    let path = POLARS_TEMP_DIR_BASE_PATH
        .join("file-cache/")
        .into_boxed_path();

    if let Err(err) = std::fs::create_dir_all(path.as_ref()) {
        if !path.is_dir() {
            panic!("failed to create file cache directory: {}", err);
        }
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

pub fn init_entries_from_uri_list<A: AsRef<[Arc<str>]>>(
    uri_list: A,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Vec<Arc<FileCacheEntry>>> {
    let uri_list = uri_list.as_ref();

    if uri_list.is_empty() {
        return Ok(Default::default());
    }

    let first_uri = uri_list.first().unwrap().as_ref();

    if is_cloud_url(first_uri) {
        let (_, object_store) = pl_async::get_runtime()
            .block_on_potential_spawn(build_object_store(first_uri, cloud_options))?;
        let object_store = PolarsObjectStore::new(object_store);

        uri_list
            .iter()
            .map(|uri| {
                FILE_CACHE.init_entry(uri.clone(), || {
                    let CloudLocation {
                        prefix, expansion, ..
                    } = CloudLocation::new(uri.as_ref()).unwrap();

                    let cloud_path = {
                        assert!(expansion.is_none(), "path should not contain wildcards");
                        object_store::path::Path::from_url_path(prefix).map_err(to_compute_err)?
                    };

                    let object_store = object_store.clone();
                    let uri = uri.clone();

                    Ok(Arc::new(CloudFileFetcher {
                        uri,
                        object_store,
                        cloud_path,
                    }))
                })
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

                FILE_CACHE.init_entry(uri.clone(), || {
                    Ok(Arc::new(LocalFileFetcher::from_uri(uri.clone())))
                })
            })
            .collect::<PolarsResult<Vec<_>>>()
    }
}

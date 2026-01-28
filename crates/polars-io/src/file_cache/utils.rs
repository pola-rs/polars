use std::sync::{Arc, LazyLock};
use std::time::UNIX_EPOCH;

use polars_error::{PolarsError, PolarsResult};
use polars_utils::pl_path::{CloudScheme, PlRefPath};

use super::cache::{FILE_CACHE, get_env_file_cache_ttl};
use super::entry::FileCacheEntry;
use super::file_fetcher::{CloudFileFetcher, LocalFileFetcher};
use crate::cloud::{CloudLocation, CloudOptions, build_object_store, object_path_from_str};
use crate::path_utils::{POLARS_TEMP_DIR_BASE_PATH, ensure_directory_init};

pub static FILE_CACHE_PREFIX: LazyLock<PlRefPath> = LazyLock::new(|| {
    let path = PlRefPath::try_from_path(&POLARS_TEMP_DIR_BASE_PATH.join("file-cache/")).unwrap();

    if let Err(err) = ensure_directory_init(path.as_ref()) {
        panic!(
            "failed to create file cache directory: path = {}, err = {}",
            path, err
        );
    }

    path
});

pub(super) fn last_modified_u64(metadata: &std::fs::Metadata) -> u64 {
    u64::try_from(
        metadata
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
}

pub(super) fn update_last_accessed(file: &std::fs::File) {
    let file_metadata = file.metadata().unwrap();

    if let Err(e) = file.set_times(
        std::fs::FileTimes::new()
            .set_modified(file_metadata.modified().unwrap())
            .set_accessed(std::time::SystemTime::now()),
    ) {
        panic!("failed to update file last accessed time: {e}");
    }
}

pub async fn init_entries_from_uri_list(
    mut uri_list: impl ExactSizeIterator<Item = PlRefPath>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Vec<Arc<FileCacheEntry>>> {
    init_entries_from_uri_list_impl(&mut uri_list, cloud_options).await
}

async fn init_entries_from_uri_list_impl(
    uri_list: &mut dyn ExactSizeIterator<Item = PlRefPath>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Vec<Arc<FileCacheEntry>>> {
    #[allow(clippy::len_zero)]
    if uri_list.len() == 0 {
        return Ok(Default::default());
    }

    let mut uri_list = uri_list.peekable();

    let first_uri = uri_list.peek().unwrap().clone();

    let file_cache_ttl = cloud_options
        .map(|x| x.file_cache_ttl)
        .unwrap_or_else(get_env_file_cache_ttl);

    if first_uri.has_scheme() {
        let shared_object_store = if !matches!(
            first_uri.scheme(),
            Some(CloudScheme::Http | CloudScheme::Https) // Object stores for http are tied to the path.
        ) {
            let (_, object_store) = build_object_store(first_uri, cloud_options, false).await?;
            Some(object_store)
        } else {
            None
        };

        futures::future::try_join_all(uri_list.map(|uri| {
            let shared_object_store = shared_object_store.clone();

            async move {
                let object_store = if let Some(shared_object_store) = shared_object_store.clone() {
                    shared_object_store
                } else {
                    let (_, object_store) =
                        build_object_store(uri.clone(), cloud_options, false).await?;
                    object_store
                };

                FILE_CACHE.init_entry(
                    uri.clone(),
                    &|| {
                        let CloudLocation { prefix, .. } =
                            CloudLocation::new(uri.clone(), false).unwrap();
                        let cloud_path = object_path_from_str(&prefix)?;
                        let object_store = object_store.clone();

                        Ok(Arc::new(CloudFileFetcher {
                            uri: uri.clone(),
                            object_store,
                            cloud_path,
                        }))
                    },
                    file_cache_ttl,
                )
            }
        }))
        .await
    } else {
        let mut out = Vec::with_capacity(uri_list.len());
        for uri in uri_list {
            let uri = tokio::fs::canonicalize(uri.as_str()).await.map_err(|err| {
                let msg = Some(format!("{}: {}", err, uri).into());
                PolarsError::IO {
                    error: err.into(),
                    msg,
                }
            })?;
            let uri = PlRefPath::try_from_pathbuf(uri)?;

            out.push(FILE_CACHE.init_entry(
                uri.clone(),
                &|| Ok(Arc::new(LocalFileFetcher::from_uri(uri.clone()))),
                file_cache_ttl,
            )?)
        }
        Ok(out)
    }
}

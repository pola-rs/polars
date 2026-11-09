use std::sync::{Arc, LazyLock};
use std::time::UNIX_EPOCH;

use polars_error::{PolarsError, PolarsResult};
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_path::{CloudScheme, PlRefPath};

use super::cache::{FILE_CACHE, get_env_file_cache_ttl};
use super::entry::FileCacheEntry;
use super::file_fetcher::{CloudFileFetcher, LocalFileFetcher};
use crate::cloud::{
    CloudLocation, CloudOptions, PolarsObjectStore, build_object_store, object_path_from_str,
};
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
    uri_list: impl ExactSizeIterator<Item = PlRefPath> + Send + 'static,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Vec<Arc<FileCacheEntry>>> {
    init_entries_from_uri_list_impl(Box::new(uri_list), cloud_options).await
}

async fn init_entries_from_uri_list_impl(
    uri_list: Box<dyn ExactSizeIterator<Item = PlRefPath> + Send + 'static>,
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
        let uri_list: Vec<PlRefPath> = uri_list.collect();

        // Http URIs can differ in origin.
        let is_http = matches!(
            first_uri.scheme(),
            Some(CloudScheme::Http | CloudScheme::Https)
        );

        let authorities: Vec<&str> = if is_http {
            Vec::new()
        } else {
            uri_list
                .iter()
                .map(|uri| &uri.as_str()[..uri.authority_end_position()])
                .collect()
        };

        // One object store per bucket, held here so that global cache evictions cannot
        // affect this call.
        let mut representatives: PlHashMap<&str, &PlRefPath> = PlHashMap::default();
        for (uri, authority) in uri_list.iter().zip(&authorities) {
            representatives.entry(authority).or_insert(uri);
        }

        let shared_object_stores: PlHashMap<&str, PolarsObjectStore> =
            futures::future::try_join_all(representatives.into_iter().map(
                |(authority, uri)| async move {
                    let (_, object_store) =
                        build_object_store(uri.clone(), cloud_options, false).await?;
                    PolarsResult::Ok((authority, object_store))
                },
            ))
            .await?
            .into_iter()
            .collect();

        futures::future::try_join_all(uri_list.iter().enumerate().map(|(i, uri)| {
            let uri = uri.clone();
            let shared_object_store = authorities
                .get(i)
                .and_then(|authority| shared_object_stores.get(authority))
                .cloned();

            async move {
                let object_store = if let Some(shared_object_store) = shared_object_store {
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

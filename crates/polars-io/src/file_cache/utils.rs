use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::UNIX_EPOCH;

use polars_error::{PolarsError, PolarsResult};
use polars_utils::plpath::{CloudScheme, PlPathRef};

use super::cache::{FILE_CACHE, get_env_file_cache_ttl};
use super::entry::FileCacheEntry;
use super::file_fetcher::{CloudFileFetcher, LocalFileFetcher};
use crate::cloud::{CloudLocation, CloudOptions, build_object_store, object_path_from_str};
use crate::path_utils::{POLARS_TEMP_DIR_BASE_PATH, ensure_directory_init};
use crate::pl_async;

pub static FILE_CACHE_PREFIX: LazyLock<Box<Path>> = LazyLock::new(|| {
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

pub fn init_entries_from_uri_list(
    mut uri_list: impl ExactSizeIterator<Item = Arc<str>>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Vec<Arc<FileCacheEntry>>> {
    init_entries_from_uri_list_impl(&mut uri_list, cloud_options)
}

fn init_entries_from_uri_list_impl(
    uri_list: &mut dyn ExactSizeIterator<Item = Arc<str>>,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Vec<Arc<FileCacheEntry>>> {
    #[expect(clippy::len_zero)]
    if uri_list.len() == 0 {
        return Ok(Default::default());
    }

    let mut uri_list = uri_list.peekable();

    let first_uri = PlPathRef::new(uri_list.peek().unwrap().as_ref());

    let file_cache_ttl = cloud_options
        .map(|x| x.file_cache_ttl)
        .unwrap_or_else(get_env_file_cache_ttl);

    if first_uri.is_cloud_url() {
        let shared_object_store = (!matches!(
            first_uri.scheme(),
            Some(CloudScheme::Http | CloudScheme::Https) // Object stores for http are tied to the path.
        ))
        .then(|| {
            pl_async::get_runtime().block_in_place_on(async {
                let (_, object_store) =
                    build_object_store(first_uri.to_str(), cloud_options, false).await?;

                PolarsResult::Ok(object_store)
            })
        })
        .transpose()?;

        pl_async::get_runtime().block_in_place_on(async {
            futures::future::try_join_all(uri_list.map(|uri| {
                let shared_object_store = shared_object_store.clone();

                async move {
                    let object_store =
                        if let Some(shared_object_store) = shared_object_store.clone() {
                            shared_object_store
                        } else {
                            let (_, object_store) =
                                build_object_store(&uri, cloud_options, false).await?;
                            object_store
                        };

                    FILE_CACHE.init_entry(
                        uri.clone(),
                        &|| {
                            let CloudLocation { prefix, .. } =
                                CloudLocation::new(uri.as_ref(), false).unwrap();
                            let cloud_path = object_path_from_str(&prefix)?;

                            let uri = uri.clone();
                            let object_store = object_store.clone();

                            Ok(Arc::new(CloudFileFetcher {
                                uri,
                                object_store,
                                cloud_path,
                            }))
                        },
                        file_cache_ttl,
                    )
                }
            }))
            .await
        })
    } else {
        uri_list
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
                    &|| Ok(Arc::new(LocalFileFetcher::from_uri(uri.clone()))),
                    file_cache_ttl,
                )
            })
            .collect::<PolarsResult<Vec<_>>>()
    }
}

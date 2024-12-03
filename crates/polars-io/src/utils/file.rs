use std::io::Write;

use polars_core::config;
use polars_error::{feature_gated, to_compute_err, PolarsResult};
use polars_utils::mmap::ensure_not_mapped;

use crate::cloud::{CloudOptions, CloudWriter};
use crate::{is_cloud_url, pl_async, resolve_homedir};

/// Open a path for writing. Supports cloud paths.
pub fn try_get_writeable(
    path: &str,
    #[cfg_attr(not(feature = "cloud"), allow(unused))] cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Box<dyn Write + Send>> {
    let is_cloud = is_cloud_url(path);

    if is_cloud {
        feature_gated!("cloud", {
            if path.starts_with("file://") {
                std::fs::File::create(
                    &path[const {
                        if cfg!(target_family = "windows") {
                            "file://".len()
                        } else {
                            "file:/".len()
                        }
                    }..],
                )
                .map_err(to_compute_err)?;
            }

            let writer = pl_async::get_runtime()
                .block_on_potential_spawn(CloudWriter::new(path, cloud_options))?;
            Ok(Box::new(writer))
        })
    } else if config::force_async() {
        feature_gated!("cloud", {
            let path = resolve_homedir(&path);
            std::fs::File::create(&path).map_err(to_compute_err)?;
            let path = std::fs::canonicalize(&path)?;

            ensure_not_mapped(&path.metadata()?)?;

            let path = format!(
                "{}{}",
                if path.starts_with("/") {
                    "file:/"
                } else {
                    "file://"
                },
                path.to_str().unwrap()
            );

            let writer = pl_async::get_runtime()
                .block_on_potential_spawn(CloudWriter::new(&path, cloud_options))?;
            Ok(Box::new(writer))
        })
    } else {
        let path = resolve_homedir(&path);
        std::fs::File::create(&path).map_err(to_compute_err)?;
        let path = std::fs::canonicalize(&path)?;

        Ok(Box::new(polars_utils::open_file_write(&path)?))
    }
}

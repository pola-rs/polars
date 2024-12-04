use std::io::Write;

use polars_core::config;
use polars_error::{feature_gated, PolarsError, PolarsResult};
use polars_utils::mmap::ensure_not_mapped;

use crate::cloud::CloudOptions;
use crate::{is_cloud_url, resolve_homedir};

/// Open a path for writing. Supports cloud paths.
pub fn try_get_writeable(
    path: &str,
    #[cfg_attr(not(feature = "cloud"), allow(unused))] cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Box<dyn Write + Send>> {
    let is_cloud = is_cloud_url(path);
    let verbose = config::verbose();

    if is_cloud {
        feature_gated!("cloud", {
            use crate::cloud::CloudWriter;

            if verbose {
                eprintln!("try_get_writeable: cloud: {}", path)
            }

            if path.starts_with("file://") {
                std::fs::File::create(&path[const { "file://".len() }..])
                    .map_err(PolarsError::from)?;
            }

            let writer = crate::pl_async::get_runtime()
                .block_on_potential_spawn(CloudWriter::new(path, cloud_options))?;
            Ok(Box::new(writer))
        })
    } else if config::force_async() {
        feature_gated!("cloud", {
            use crate::cloud::CloudWriter;

            let path = resolve_homedir(&path);

            if verbose {
                eprintln!(
                    "try_get_writeable: forced async: {}",
                    path.to_str().unwrap()
                )
            }

            std::fs::File::create(&path).map_err(PolarsError::from)?;
            let path = std::fs::canonicalize(&path)?;

            ensure_not_mapped(&path.metadata()?)?;

            let path = format!(
                "file://{}",
                if cfg!(target_family = "windows") {
                    path.to_str().unwrap().strip_prefix(r#"\\?\"#).unwrap()
                } else {
                    path.to_str().unwrap()
                }
            );

            if verbose {
                eprintln!("try_get_writeable: forced async converted path: {}", path)
            }

            let writer = crate::pl_async::get_runtime()
                .block_on_potential_spawn(CloudWriter::new(&path, cloud_options))?;
            Ok(Box::new(writer))
        })
    } else {
        let path = resolve_homedir(&path);
        std::fs::File::create(&path).map_err(PolarsError::from)?;
        let path = std::fs::canonicalize(&path)?;

        if verbose {
            eprintln!("try_get_writeable: local: {}", path.to_str().unwrap())
        }

        Ok(Box::new(polars_utils::open_file_write(&path)?))
    }
}

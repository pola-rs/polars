use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::path::Path;

#[cfg(feature = "cloud")]
pub use async_writeable::AsyncWriteable;
use polars_core::config;
use polars_error::{feature_gated, PolarsResult};
use polars_utils::create_file;
use polars_utils::mmap::ensure_not_mapped;

use crate::cloud::CloudOptions;
use crate::{is_cloud_url, resolve_homedir};

/// Non-async writeable file, abstracted over local files or cloud files.
///
/// Also see: `Writeable::into_async_writeable` and `AsyncWriteable`.
pub enum Writeable {
    Local(std::fs::File),
    #[cfg(feature = "cloud")]
    Cloud(crate::cloud::CloudWriter),
}

impl Writeable {
    pub fn try_new(
        path: &str,
        #[cfg_attr(not(feature = "cloud"), allow(unused))] cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        let is_cloud = is_cloud_url(path);
        let verbose = config::verbose();

        if is_cloud {
            feature_gated!("cloud", {
                use crate::cloud::CloudWriter;

                if verbose {
                    eprintln!("try_get_writeable: cloud: {}", path)
                }

                if path.starts_with("file://") {
                    create_file(Path::new(&path[const { "file://".len() }..]))?;
                }

                let writer = crate::pl_async::get_runtime()
                    .block_on_potential_spawn(CloudWriter::new(path, cloud_options))?;
                Ok(Self::Cloud(writer))
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

                create_file(&path)?;
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
                Ok(Self::Cloud(writer))
            })
        } else {
            let path = resolve_homedir(&path);
            create_file(&path)?;

            // Note: `canonicalize` does not work on some systems.

            if verbose {
                eprintln!(
                    "try_get_writeable: local: {} (canonicalize: {:?})",
                    path.to_str().unwrap(),
                    std::fs::canonicalize(&path)
                )
            }

            Ok(Self::Local(polars_utils::open_file_write(&path)?))
        }
    }

    pub fn into_box_dyn_write(self) -> Box<dyn Write + Send> {
        match self {
            Self::Local(v) => Box::new(v),
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => Box::new(v),
        }
    }

    #[cfg(feature = "cloud")]
    pub fn into_async_writeable(self) -> AsyncWriteable {
        match self {
            Self::Local(v) => AsyncWriteable::Local(tokio::fs::File::from_std(v)),
            Self::Cloud(v) => AsyncWriteable::Cloud(v),
        }
    }

    pub fn close(self) -> PolarsResult<()> {
        match self {
            // @RAISE_FILE_CLOSE_ERR
            Self::Local(_) => Ok(()),
            #[cfg(feature = "cloud")]
            Self::Cloud(mut v) => v.close_sync(),
        }
    }
}

impl Deref for Writeable {
    type Target = dyn Write + Send;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Local(v) => v,
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v,
        }
    }
}

impl DerefMut for Writeable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Local(v) => v,
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v,
        }
    }
}

/// Note: Prefer using [`Writeable`] / [`Writeable::try_new`] where possible.
///
/// Open a path for writing. Supports cloud paths.
pub fn try_get_writeable(
    path: &str,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<Box<dyn Write + Send>> {
    Writeable::try_new(path, cloud_options).map(|x| x.into_box_dyn_write())
}

#[cfg(feature = "cloud")]
mod async_writeable {
    use std::ops::{Deref, DerefMut};

    use polars_error::PolarsResult;

    // Imported for docstring
    #[allow(unused)]
    use super::Writeable;

    /// Construct this using [`Writeable::try_new()`] and then [`Writeable::into_async_writeable`]
    pub enum AsyncWriteable {
        Local(tokio::fs::File),
        Cloud(crate::cloud::CloudWriter),
    }

    impl AsyncWriteable {
        pub async fn close(self) -> PolarsResult<()> {
            match self {
                // TODO: Raise error from closing local file (this only syncs)
                // @RAISE_FILE_CLOSE_ERR
                Self::Local(v) => v.sync_all().await.map_err(|e| e.into()),
                Self::Cloud(mut v) => v.close().await,
            }
        }
    }

    impl Deref for AsyncWriteable {
        type Target = dyn tokio::io::AsyncWrite + Send + Unpin;

        fn deref(&self) -> &Self::Target {
            match self {
                Self::Local(v) => v,
                #[cfg(feature = "cloud")]
                Self::Cloud(v) => v,
            }
        }
    }

    impl DerefMut for AsyncWriteable {
        fn deref_mut(&mut self) -> &mut Self::Target {
            match self {
                Self::Local(v) => v,
                #[cfg(feature = "cloud")]
                Self::Cloud(v) => v,
            }
        }
    }
}

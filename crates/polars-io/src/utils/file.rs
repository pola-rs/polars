use std::ops::{Deref, DerefMut};
use std::path::Path;

#[cfg(feature = "cloud")]
pub use async_writeable::AsyncWriteable;
use polars_core::config;
use polars_error::{PolarsError, PolarsResult, feature_gated};
use polars_utils::create_file;
use polars_utils::file::{ClosableFile, WriteClose};
use polars_utils::mmap::ensure_not_mapped;

use crate::cloud::CloudOptions;
use crate::{is_cloud_url, resolve_homedir};

/// Holds a non-async writeable file, abstracted over local files or cloud files.
///
/// This implements `DerefMut` to a trait object implementing [`std::io::Write`].
///
/// Also see: `Writeable::try_into_async_writeable` and `AsyncWriteable`.
#[allow(clippy::large_enum_variant)] // It will be boxed
pub enum Writeable {
    Local(std::fs::File),
    #[cfg(feature = "cloud")]
    Cloud(crate::cloud::BlockingCloudWriter),
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
                use crate::cloud::BlockingCloudWriter;

                if verbose {
                    eprintln!("Writeable: try_new: cloud: {}", path)
                }

                if path.starts_with("file://") {
                    create_file(Path::new(&path[const { "file://".len() }..]))?;
                }

                let writer = crate::pl_async::get_runtime()
                    .block_in_place_on(BlockingCloudWriter::new(path, cloud_options))?;
                Ok(Self::Cloud(writer))
            })
        } else if config::force_async() {
            feature_gated!("cloud", {
                use crate::cloud::BlockingCloudWriter;

                let path = resolve_homedir(&path);

                if verbose {
                    eprintln!(
                        "Writeable: try_new: forced async: {}",
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
                    eprintln!("Writeable: try_new: forced async converted path: {}", path)
                }

                let writer = crate::pl_async::get_runtime()
                    .block_in_place_on(BlockingCloudWriter::new(&path, cloud_options))?;
                Ok(Self::Cloud(writer))
            })
        } else {
            let path = resolve_homedir(&path);
            create_file(&path)?;

            // Note: `canonicalize` does not work on some systems.

            if verbose {
                eprintln!(
                    "Writeable: try_new: local: {} (canonicalize: {:?})",
                    path.to_str().unwrap(),
                    std::fs::canonicalize(&path)
                )
            }

            Ok(Self::Local(polars_utils::open_file_write(&path)?))
        }
    }

    /// This returns `Result<>` - if a write was performed before calling this,
    /// `CloudWriter` can be in an Err(_) state.
    #[cfg(feature = "cloud")]
    pub fn try_into_async_writeable(self) -> PolarsResult<AsyncWriteable> {
        match self {
            Self::Local(v) => Ok(AsyncWriteable::Local(tokio::fs::File::from_std(v))),
            // Moves the `BufWriter` out of the `BlockingCloudWriter` wrapper, as
            // `BlockingCloudWriter` has a `Drop` impl that we don't want.
            Self::Cloud(v) => v
                .try_into_inner()
                .map(AsyncWriteable::Cloud)
                .map_err(PolarsError::from),
        }
    }

    pub fn close(self) -> std::io::Result<()> {
        match self {
            Self::Local(v) => ClosableFile::from(v).close(),
            #[cfg(feature = "cloud")]
            Self::Cloud(mut v) => v.close(),
        }
    }
}

impl Deref for Writeable {
    type Target = dyn std::io::Write + Send;

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
) -> PolarsResult<Box<dyn WriteClose + Send>> {
    Writeable::try_new(path, cloud_options).map(|x| match x {
        Writeable::Local(v) => Box::new(ClosableFile::from(v)) as Box<dyn WriteClose + Send>,
        #[cfg(feature = "cloud")]
        Writeable::Cloud(v) => Box::new(v) as Box<dyn WriteClose + Send>,
    })
}

#[cfg(feature = "cloud")]
mod async_writeable {
    use std::ops::{Deref, DerefMut};

    use polars_error::{PolarsError, PolarsResult};
    use polars_utils::file::ClosableFile;
    use tokio::io::AsyncWriteExt;

    use super::Writeable;
    use crate::cloud::CloudOptions;

    /// Holds an async writeable file, abstracted over local files or cloud files.
    ///
    /// This implements `DerefMut` to a trait object implementing [`tokio::io::AsyncWrite`].
    ///
    /// Note: It is important that you do not call `shutdown()` on the deref'ed `AsyncWrite` object.
    /// You should instead call the [`AsyncWriteable::close`] at the end.
    pub enum AsyncWriteable {
        Local(tokio::fs::File),
        Cloud(object_store::buffered::BufWriter),
    }

    impl AsyncWriteable {
        pub async fn try_new(
            path: &str,
            cloud_options: Option<&CloudOptions>,
        ) -> PolarsResult<Self> {
            // TODO: Native async impl
            Writeable::try_new(path, cloud_options).and_then(|x| x.try_into_async_writeable())
        }

        pub async fn close(self) -> PolarsResult<()> {
            match self {
                Self::Local(v) => async {
                    let f = v.into_std().await;
                    ClosableFile::from(f).close()
                }
                .await
                .map_err(PolarsError::from),
                Self::Cloud(mut v) => v.shutdown().await.map_err(PolarsError::from),
            }
        }
    }

    impl Deref for AsyncWriteable {
        type Target = dyn tokio::io::AsyncWrite + Send + Unpin;

        fn deref(&self) -> &Self::Target {
            match self {
                Self::Local(v) => v,
                Self::Cloud(v) => v,
            }
        }
    }

    impl DerefMut for AsyncWriteable {
        fn deref_mut(&mut self) -> &mut Self::Target {
            match self {
                Self::Local(v) => v,
                Self::Cloud(v) => v,
            }
        }
    }
}

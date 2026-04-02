use std::io;
#[cfg(feature = "cloud")]
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[cfg(feature = "cloud")]
pub use async_writeable::{AsyncDynWriteable, AsyncWriteable};
use polars_error::{PolarsResult, feature_gated, polars_err};
use polars_utils::create_file;
use polars_utils::file::close_file;
use polars_utils::mmap::ensure_not_mapped;
use polars_utils::pl_path::{PlRefPath, format_file_uri};

use super::sync_on_close::SyncOnCloseType;
use crate::cloud::CloudOptions;
use crate::metrics::IOMetrics;
use crate::resolve_homedir;

// TODO document precise contract.
pub trait WriteableTrait: std::io::Write {
    fn close(&mut self) -> std::io::Result<()>;
    fn sync_all(&self) -> std::io::Result<()>;
    fn sync_data(&self) -> std::io::Result<()>;
}

/// Holds a non-async writeable file, abstracted over local files or cloud files.
///
/// This implements `DerefMut` to a trait object implementing [`std::io::Write`].
///
/// Also see: `Writeable::try_into_async_writeable` and `AsyncWriteable`.
#[allow(clippy::large_enum_variant)] // It will be boxed
pub enum Writeable {
    /// An abstract implementation for writable.
    ///
    /// This is used to implement writing to in-memory and arbitrary file descriptors.
    Dyn(Box<dyn WriteableTrait + Send>),
    Local(std::fs::File),
    #[cfg(feature = "cloud")]
    Cloud(crate::cloud::cloud_writer::CloudWriterIoTraitWrap),
}

impl Writeable {
    pub fn try_new(
        path: PlRefPath,
        #[cfg_attr(not(feature = "cloud"), expect(unused))] cloud_options: Option<&CloudOptions>,
        #[cfg_attr(not(feature = "cloud"), expect(unused))] cloud_upload_chunk_size: usize,
        #[cfg_attr(not(feature = "cloud"), expect(unused))] cloud_upload_concurrency: usize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Self> {
        Ok(if path.has_scheme() {
            feature_gated!("cloud", {
                use crate::cloud::cloud_writer::CloudWriterIoTraitWrap;
                use crate::pl_async::get_runtime;

                let writer = get_runtime().block_in_place_on(new_cloud_writer(
                    path,
                    cloud_options,
                    cloud_upload_chunk_size,
                    cloud_upload_concurrency.try_into().unwrap(),
                    io_metrics,
                ))?;

                Self::Cloud(CloudWriterIoTraitWrap::from(writer))
            })
        } else if polars_config::config().force_async() {
            feature_gated!("cloud", {
                let path = resolve_homedir(path.as_std_path());
                create_file(&path)?;
                let path = std::fs::canonicalize(&path)?;

                ensure_not_mapped(&path.metadata()?)?;

                let path = path.to_str().ok_or_else(|| polars_err!(non_utf8_path))?;
                let path = format_file_uri(path);

                use crate::cloud::cloud_writer::CloudWriterIoTraitWrap;
                use crate::pl_async::get_runtime;

                let writer = get_runtime().block_in_place_on(new_cloud_writer(
                    path,
                    cloud_options,
                    cloud_upload_chunk_size,
                    cloud_upload_concurrency.try_into().unwrap(),
                    io_metrics,
                ))?;

                Self::Cloud(CloudWriterIoTraitWrap::from(writer))
            })
        } else {
            let path = resolve_homedir(path.as_std_path());
            create_file(&path)?;

            Self::Local(polars_utils::open_file_write(&path)?)
        })
    }

    /// This returns `Result<>` - if a write was performed before calling this,
    /// `CloudWriter` can be in an Err(_) state.
    #[cfg(feature = "cloud")]
    pub fn try_into_async_writeable(self) -> PolarsResult<AsyncWriteable> {
        use self::async_writeable::AsyncDynWriteable;

        match self {
            Self::Dyn(v) => Ok(AsyncWriteable::Dyn(AsyncDynWriteable(v))),
            Self::Local(v) => Ok(AsyncWriteable::Local(tokio::fs::File::from_std(v))),
            Self::Cloud(v) => Ok(AsyncWriteable::Cloud(v)),
        }
    }

    pub fn as_buffered(&mut self) -> BufferedWriteable<'_> {
        match self {
            Writeable::Dyn(v) => BufferedWriteable::BufWriter(std::io::BufWriter::new(v.as_mut())),
            Writeable::Local(v) => BufferedWriteable::BufWriter(std::io::BufWriter::new(v)),
            #[cfg(feature = "cloud")]
            Writeable::Cloud(v) => BufferedWriteable::Direct(v as _),
        }
    }

    pub fn sync_all(&self) -> io::Result<()> {
        match self {
            Self::Dyn(v) => v.sync_all(),
            Self::Local(v) => v.sync_all(),
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.sync_all(),
        }
    }

    pub fn sync_data(&self) -> io::Result<()> {
        match self {
            Self::Dyn(v) => v.sync_data(),
            Self::Local(v) => v.sync_data(),
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.sync_data(),
        }
    }

    pub fn close(self, sync: SyncOnCloseType) -> std::io::Result<()> {
        match sync {
            SyncOnCloseType::All => self.sync_all()?,
            SyncOnCloseType::Data => self.sync_data()?,
            SyncOnCloseType::None => {},
        }

        match self {
            Self::Dyn(mut v) => v.close(),
            Self::Local(v) => close_file(v),
            #[cfg(feature = "cloud")]
            Self::Cloud(mut v) => v.close(),
        }
    }
}

impl io::Write for Writeable {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Dyn(v) => v.write(buf),
            Self::Local(v) => v.write(buf),
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sync_all()
    }
}

impl Deref for Writeable {
    type Target = dyn io::Write + Send;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Dyn(v) => v,
            Self::Local(v) => v,
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v,
        }
    }
}

impl DerefMut for Writeable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Dyn(v) => v,
            Self::Local(v) => v,
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v,
        }
    }
}

/// Avoid BufWriter wrapping on writers that already have internal buffering.
pub enum BufferedWriteable<'a> {
    BufWriter(std::io::BufWriter<&'a mut (dyn std::io::Write + Send)>),
    Direct(&'a mut (dyn std::io::Write + Send)),
}

impl<'a> Deref for BufferedWriteable<'a> {
    type Target = dyn io::Write + Send + 'a;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::BufWriter(v) => v as _,
            Self::Direct(v) => v,
        }
    }
}

impl DerefMut for BufferedWriteable<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::BufWriter(v) => v as _,
            Self::Direct(v) => v,
        }
    }
}

#[cfg(feature = "cloud")]
async fn new_cloud_writer(
    path: PlRefPath,
    cloud_options: Option<&CloudOptions>,
    cloud_upload_chunk_size: usize,
    cloud_upload_concurrency: NonZeroUsize,
    io_metrics: Option<Arc<IOMetrics>>,
) -> PolarsResult<crate::cloud::cloud_writer::CloudWriter> {
    use crate::cloud::cloud_writer::CloudWriter;
    use crate::cloud::object_path_from_str;

    let (cloud_location, object_store) =
        crate::cloud::build_object_store(path, cloud_options, false).await?;

    let mut writer = CloudWriter::new(
        object_store,
        object_path_from_str(&cloud_location.prefix)?,
        cloud_upload_chunk_size,
        cloud_upload_concurrency,
        io_metrics,
    );

    writer.start().await?;

    Ok(writer)
}

#[cfg(feature = "cloud")]
mod async_writeable {
    use std::io;
    use std::ops::{Deref, DerefMut};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use bytes::Bytes;
    use polars_error::{PolarsError, PolarsResult};
    use polars_utils::file::close_file;
    use polars_utils::pl_path::PlRefPath;
    use tokio::io::AsyncWriteExt;
    use tokio::task;

    use super::{Writeable, WriteableTrait};
    use crate::cloud::CloudOptions;
    use crate::metrics::IOMetrics;
    use crate::utils::sync_on_close::SyncOnCloseType;

    /// Turn an abstract io::Write into an abstract tokio::io::AsyncWrite.
    pub struct AsyncDynWriteable(pub Box<dyn WriteableTrait + Send>);

    impl tokio::io::AsyncWrite for AsyncDynWriteable {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            let result = task::block_in_place(|| self.get_mut().0.write(buf));
            Poll::Ready(result)
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            let result = task::block_in_place(|| self.get_mut().0.flush());
            Poll::Ready(result)
        }

        fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            self.poll_flush(cx)
        }
    }

    /// Holds an async writeable file, abstracted over local files or cloud files.
    ///
    /// This implements `DerefMut` to a trait object implementing [`tokio::io::AsyncWrite`].
    ///
    /// Note: It is important that you do not call `shutdown()` on the deref'ed `AsyncWrite` object.
    /// You should instead call the [`AsyncWriteable::close`] at the end.
    pub enum AsyncWriteable {
        Dyn(AsyncDynWriteable),
        Local(tokio::fs::File),
        Cloud(crate::cloud::cloud_writer::CloudWriterIoTraitWrap),
    }

    impl AsyncWriteable {
        pub async fn try_new(
            path: PlRefPath,
            cloud_options: Option<&CloudOptions>,
            cloud_upload_chunk_size: usize,
            cloud_upload_concurrency: usize,
            io_metrics: Option<Arc<IOMetrics>>,
        ) -> PolarsResult<Self> {
            // TODO: Native async impl
            Writeable::try_new(
                path,
                cloud_options,
                cloud_upload_chunk_size,
                cloud_upload_concurrency,
                io_metrics,
            )
            .and_then(|x| x.try_into_async_writeable())
        }

        /// If this writer holds a cloud writer, it will `mem::take(T)`. `T` is unmodified for other
        /// writer types.
        pub async fn write_all_owned<T>(&mut self, src: &mut T) -> io::Result<()>
        where
            T: AsRef<[u8]> + Default + Drop, // `Drop` is to exclude `&[u8]` slices.
            Bytes: From<T>,
        {
            match self {
                Self::Cloud(v) => v.write_all_owned(Bytes::from(std::mem::take(src))).await,
                Self::Dyn(_) | Self::Local(_) => self.write_all(src.as_ref()).await,
            }
        }

        pub async fn sync_all(&mut self) -> io::Result<()> {
            match self {
                Self::Dyn(v) => task::block_in_place(|| v.0.as_ref().sync_all()),
                Self::Local(v) => v.sync_all().await,
                Self::Cloud(_) => Ok(()),
            }
        }

        pub async fn sync_data(&mut self) -> io::Result<()> {
            match self {
                Self::Dyn(v) => task::block_in_place(|| v.0.as_ref().sync_data()),
                Self::Local(v) => v.sync_data().await,
                Self::Cloud(_) => Ok(()),
            }
        }

        pub async fn close(mut self, sync: SyncOnCloseType) -> PolarsResult<()> {
            match sync {
                SyncOnCloseType::All => self.sync_all().await?,
                SyncOnCloseType::Data => self.sync_data().await?,
                SyncOnCloseType::None => {},
            }

            match self {
                Self::Dyn(mut v) => {
                    v.shutdown().await.map_err(PolarsError::from)?;
                    Ok(task::block_in_place(|| v.0.close())?)
                },
                Self::Local(v) => async {
                    let f = v.into_std().await;
                    close_file(f)
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
                Self::Dyn(v) => v,
                Self::Local(v) => v,
                Self::Cloud(v) => v,
            }
        }
    }

    impl DerefMut for AsyncWriteable {
        fn deref_mut(&mut self) -> &mut Self::Target {
            match self {
                Self::Dyn(v) => v,
                Self::Local(v) => v,
                Self::Cloud(v) => v,
            }
        }
    }
}

use std::io;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[cfg(feature = "cloud")]
pub use async_writable::{AsyncDynWritable, AsyncWritable};
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
pub trait WritableTrait: std::io::Write {
    fn close(&mut self) -> std::io::Result<()>;
    fn sync_all(&self) -> std::io::Result<()>;
    fn sync_data(&self) -> std::io::Result<()>;
}

/// Holds a non-async writable file, abstracted over local files or cloud files.
///
/// This implements `DerefMut` to a trait object implementing [`std::io::Write`].
///
/// Also see: `Writable::try_into_async_writable` and `AsyncWritable`.
#[allow(clippy::large_enum_variant)] // It will be boxed
pub enum Writable {
    /// An abstract implementation for writable.
    ///
    /// This is used to implement writing to in-memory and arbitrary file descriptors.
    Dyn(Box<dyn WritableTrait + Send>),
    Local(std::fs::File),
    #[cfg(feature = "cloud")]
    Cloud(crate::cloud::cloud_writer::CloudWriterIoTraitWrap),
}

impl Writable {
    pub fn try_new(
        path: PlRefPath,
        #[cfg_attr(not(feature = "cloud"), expect(unused))] cloud_options: Option<&CloudOptions>,
        #[cfg_attr(not(feature = "cloud"), expect(unused))] cloud_upload_chunk_size: Option<
            NonZeroUsize,
        >,
        #[cfg_attr(not(feature = "cloud"), expect(unused))] cloud_upload_concurrency: usize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Self> {
        Ok(if path.has_scheme() {
            feature_gated!("cloud", {
                use polars_core::runtime::ASYNC;

                use crate::cloud::cloud_writer::CloudWriterIoTraitWrap;

                let writer = ASYNC.block_in_place_on(new_cloud_writer(
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

                use polars_core::runtime::ASYNC;

                use crate::cloud::cloud_writer::CloudWriterIoTraitWrap;

                let writer = ASYNC.block_in_place_on(new_cloud_writer(
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

    /// If this writer holds a cloud writer, it will `mem::take(T)`. `T` is unmodified for other
    /// writer types.
    #[cfg(feature = "cloud")]
    pub async fn write_all_owned<T>(&mut self, src: &mut T) -> io::Result<()>
    where
        T: AsRef<[u8]> + Default + Drop, // `Drop` is to exclude `&[u8]` slices.
        bytes::Bytes: From<T>,
    {
        match self {
            Self::Cloud(v) => {
                v.write_all_owned(bytes::Bytes::from(std::mem::take(src)))
                    .await
            },
            Self::Dyn(_) | Self::Local(_) => self.write_all(src.as_ref()),
        }
    }

    /// This returns `Result<>` - if a write was performed before calling this,
    /// `CloudWriter` can be in an Err(_) state.
    #[cfg(feature = "cloud")]
    pub fn try_into_async_writable(self) -> PolarsResult<AsyncWritable> {
        use self::async_writable::AsyncDynWritable;

        match self {
            Self::Dyn(v) => Ok(AsyncWritable::Dyn(AsyncDynWritable(v))),
            Self::Local(v) => Ok(AsyncWritable::Local(tokio::fs::File::from_std(v))),
            Self::Cloud(v) => Ok(AsyncWritable::Cloud(v)),
        }
    }

    pub fn as_buffered(&mut self) -> BufferedWritable<'_> {
        match self {
            Writable::Dyn(v) => BufferedWritable::BufWriter(std::io::BufWriter::new(v.as_mut())),
            Writable::Local(v) => BufferedWritable::BufWriter(std::io::BufWriter::new(v)),
            #[cfg(feature = "cloud")]
            Writable::Cloud(v) => BufferedWritable::Direct(v as _),
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

impl io::Write for Writable {
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

impl Deref for Writable {
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

impl DerefMut for Writable {
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
pub enum BufferedWritable<'a> {
    BufWriter(std::io::BufWriter<&'a mut (dyn std::io::Write + Send)>),
    Direct(&'a mut (dyn std::io::Write + Send)),
}

impl<'a> Deref for BufferedWritable<'a> {
    type Target = dyn io::Write + Send + 'a;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::BufWriter(v) => v as _,
            Self::Direct(v) => v,
        }
    }
}

impl DerefMut for BufferedWritable<'_> {
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
    cloud_upload_chunk_size: Option<NonZeroUsize>,
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
mod async_writable {
    use std::io;
    use std::num::NonZeroUsize;
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

    use super::{Writable, WritableTrait};
    use crate::cloud::CloudOptions;
    use crate::metrics::IOMetrics;
    use crate::utils::sync_on_close::SyncOnCloseType;

    /// Turn an abstract io::Write into an abstract tokio::io::AsyncWrite.
    pub struct AsyncDynWritable(pub Box<dyn WritableTrait + Send>);

    impl tokio::io::AsyncWrite for AsyncDynWritable {
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

    /// Holds an async writable file, abstracted over local files or cloud files.
    ///
    /// This implements `DerefMut` to a trait object implementing [`tokio::io::AsyncWrite`].
    ///
    /// Note: It is important that you do not call `shutdown()` on the deref'ed `AsyncWrite` object.
    /// You should instead call the [`AsyncWritable::close`] at the end.
    pub enum AsyncWritable {
        Dyn(AsyncDynWritable),
        Local(tokio::fs::File),
        Cloud(crate::cloud::cloud_writer::CloudWriterIoTraitWrap),
    }

    impl AsyncWritable {
        pub async fn try_new(
            path: PlRefPath,
            cloud_options: Option<&CloudOptions>,
            cloud_upload_chunk_size: Option<NonZeroUsize>,
            cloud_upload_concurrency: usize,
            io_metrics: Option<Arc<IOMetrics>>,
        ) -> PolarsResult<Self> {
            // TODO: Native async impl
            Writable::try_new(
                path,
                cloud_options,
                cloud_upload_chunk_size,
                cloud_upload_concurrency,
                io_metrics,
            )
            .and_then(|x| x.try_into_async_writable())
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

    impl Deref for AsyncWritable {
        type Target = dyn tokio::io::AsyncWrite + Send + Unpin;

        fn deref(&self) -> &Self::Target {
            match self {
                Self::Dyn(v) => v,
                Self::Local(v) => v,
                Self::Cloud(v) => v,
            }
        }
    }

    impl DerefMut for AsyncWritable {
        fn deref_mut(&mut self) -> &mut Self::Target {
            match self {
                Self::Dyn(v) => v,
                Self::Local(v) => v,
                Self::Cloud(v) => v,
            }
        }
    }
}

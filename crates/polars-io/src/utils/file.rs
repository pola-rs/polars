use std::io;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use polars_error::{PolarsResult, feature_gated, polars_err};
use polars_utils::file::close_file;
use polars_utils::io::create_file;
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

            Self::Local(polars_utils::io::open_file_write(&path)?)
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
        match self {
            Self::Dyn(v) => v.flush(),
            Self::Local(v) => v.flush(),
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.flush(),
        }
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

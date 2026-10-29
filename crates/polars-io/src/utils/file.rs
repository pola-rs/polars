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

/// Holds a local file with transactional staging and cleanup on drop.
pub struct LocalFileWriter {
    file: Option<std::fs::File>,
    staging: Option<FileStaging>,
}

struct FileStaging {
    target_path: std::path::PathBuf,
    temp_path: std::path::PathBuf,
    completed: bool,
}

impl LocalFileWriter {
    pub fn from_file(file: std::fs::File) -> Self {
        Self {
            file: Some(file),
            staging: None,
        }
    }

    pub fn new_staged(path: &std::path::Path) -> PolarsResult<Self> {
        let target_path = path.to_path_buf();

        // Check if destination is a special device (e.g. /dev/null, /dev/stdout, etc.)
        let is_special = {
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileTypeExt;
                if target_path == std::path::Path::new("/dev/null")
                    || target_path == std::path::Path::new("/dev/stdout")
                    || target_path == std::path::Path::new("/dev/stderr")
                {
                    true
                } else if let Ok(meta) = std::fs::metadata(&target_path) {
                    let ft = meta.file_type();
                    ft.is_char_device() || ft.is_fifo() || ft.is_socket() || ft.is_block_device()
                } else {
                    false
                }
            }
            #[cfg(not(unix))]
            {
                false
            }
        };

        if is_special {
            create_file(&target_path)?;
            let file = polars_utils::io::open_file_write(&target_path)?;
            return Ok(Self::from_file(file));
        }

        let dir = target_path.parent().unwrap_or(std::path::Path::new("."));
        let filename = target_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("polars_sink");

        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let temp_path = dir.join(format!(
            ".{filename}.tmp_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));

        create_file(&temp_path)?;
        let file = polars_utils::io::open_file_write(&temp_path)?;

        Ok(Self {
            file: Some(file),
            staging: Some(FileStaging {
                target_path,
                temp_path,
                completed: false,
            }),
        })
    }

    pub fn sync_all(&self) -> io::Result<()> {
        if let Some(file) = &self.file {
            file.sync_all()
        } else {
            Ok(())
        }
    }

    pub fn sync_data(&self) -> io::Result<()> {
        if let Some(file) = &self.file {
            file.sync_data()
        } else {
            Ok(())
        }
    }

    pub fn close(mut self) -> io::Result<()> {
        if let Some(file) = self.file.take() {
            close_file(file)?;
        }
        if let Some(ref mut staging) = self.staging {
            std::fs::rename(&staging.temp_path, &staging.target_path)?;
            staging.completed = true;
        }
        Ok(())
    }
}

impl io::Write for LocalFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.as_mut().unwrap().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.as_mut().unwrap().flush()
    }
}

impl Deref for LocalFileWriter {
    type Target = std::fs::File;

    fn deref(&self) -> &Self::Target {
        self.file.as_ref().unwrap()
    }
}

impl DerefMut for LocalFileWriter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.file.as_mut().unwrap()
    }
}

impl Drop for LocalFileWriter {
    fn drop(&mut self) {
        if let Some(ref mut staging) = self.staging {
            if !staging.completed {
                if let Some(file) = self.file.take() {
                    let _ = close_file(file);
                }
                let _ = std::fs::remove_file(&staging.temp_path);
            }
        }
    }
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
    Local(LocalFileWriter),
    #[cfg(feature = "cloud")]
    Cloud(crate::cloud::cloud_writer::CloudWriterIoTraitWrap),
}

impl Writable {
    pub fn local_from_file(file: std::fs::File) -> Self {
        Self::Local(LocalFileWriter::from_file(file))
    }

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
            Self::Local(LocalFileWriter::new_staged(&path)?)
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
            Self::Local(v) => v.close(),
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

    Ok(CloudWriter::new(
        object_store,
        object_path_from_str(&cloud_location.prefix)?,
        cloud_upload_chunk_size,
        cloud_upload_concurrency,
        io_metrics,
    ))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn test_local_file_writer_staging_success() {
        let temp_dir = std::env::temp_dir().join(format!("polars_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let target_path = temp_dir.join("test_success.txt");
        let _ = std::fs::remove_file(&target_path);

        let mut writer = LocalFileWriter::new_staged(&target_path).unwrap();
        assert!(!target_path.exists());

        let temp_path = writer.staging.as_ref().unwrap().temp_path.clone();
        assert!(temp_path.exists());

        writer.write_all(b"hello world").unwrap();
        writer.close().unwrap();

        assert!(target_path.exists());
        assert!(!temp_path.exists());
        assert_eq!(std::fs::read(&target_path).unwrap(), b"hello world");

        let _ = std::fs::remove_file(&target_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_local_file_writer_drop_cleanup() {
        let temp_dir =
            std::env::temp_dir().join(format!("polars_test_drop_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let target_path = temp_dir.join("test_drop.txt");
        let _ = std::fs::remove_file(&target_path);

        let temp_path = {
            let mut writer = LocalFileWriter::new_staged(&target_path).unwrap();
            let tp = writer.staging.as_ref().unwrap().temp_path.clone();
            assert!(tp.exists());
            writer.write_all(b"corrupted partial data").unwrap();
            // Drop writer without calling close()
            tp
        };

        // File must not exist at target, and temp file must have been deleted
        assert!(!target_path.exists());
        assert!(!temp_path.exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_local_file_writer_preserve_existing_on_failure() {
        let temp_dir =
            std::env::temp_dir().join(format!("polars_test_preserve_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let target_path = temp_dir.join("test_preserve.txt");
        std::fs::write(&target_path, b"original content").unwrap();

        let temp_path = {
            let mut writer = LocalFileWriter::new_staged(&target_path).unwrap();
            let tp = writer.staging.as_ref().unwrap().temp_path.clone();
            assert!(tp.exists());
            writer.write_all(b"partial failure").unwrap();
            tp
        };

        // Target path should still have the original content untouched
        assert!(target_path.exists());
        assert_eq!(std::fs::read(&target_path).unwrap(), b"original content");
        assert!(!temp_path.exists());

        let _ = std::fs::remove_file(&target_path);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

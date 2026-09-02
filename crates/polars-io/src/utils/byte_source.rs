use std::cmp::Ordering;
use std::ops::Range;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use dio_align::DioAlign;
use futures::{StreamExt, TryStreamExt};
use polars_buffer::Buffer;
use polars_core::prelude::PlHashMap;
use polars_error::{PolarsResult, feature_gated, polars_err};
use polars_utils::aliases::InitHashMaps;
use polars_utils::io::_limit_path_len_io_err;
use polars_utils::mmap::MMapSemaphore;
use polars_utils::pl_path::PlRefPath;
use tokio::sync::Semaphore;

use crate::cloud::concurrency_config::{ConcurrencyStrategy, FetchConfig};
use crate::cloud::options::CloudOptions;
#[cfg(feature = "cloud")]
use crate::cloud::{
    CloudLocation, ObjectStorePath, PolarsObjectStore, build_object_store, object_path_from_str,
};
use crate::metrics::{IOMetrics, OptIOMetrics};

#[cfg(target_os = "linux")]
pub mod dio_align;
#[cfg(target_os = "linux")]
mod direct_io;

//kdn DEV CONTROL STATICS
#[cfg(target_os = "linux")]
static PLDEV_OPEN_FADV: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("PLDEV_OPEN_FADV").map_or(0, |x| {
        let fadv = x.parse::<usize>().unwrap();

        if polars_config::config().verbose() {
            //defaults to POSIX_FADV_NORMAL
            match fadv {
                0 => {
                    eprintln!("PLDEV_OPEN_FADV posix_adv_normal");
                },
                1 => {
                    eprintln!("PLDEV_OPEN_FADV posix_adv_sequential");
                },
                2 => {
                    eprintln!("PLDEV_OPEN_FADV posix_adv_random");
                },
                3 => {
                    eprintln!("PLDEV_OPEN_FADV posix_adv_willneed");
                },
                _ => panic!("illegal value for PLDEV_OPEN_FADV: {}", fadv),
            }
        };

        fadv
    })
});

#[allow(async_fn_in_trait)]
pub trait ByteSource: Send + Sync {
    async fn get_size(&self) -> PolarsResult<usize>;
    /// # Panics
    /// Panics if `range` is not in bounds.
    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Buffer<u8>>;
    /// Note: This will mutably sort ranges for coalescing.
    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, Buffer<u8>>>;
}

/// Byte source backed by a `Buffer`, which can potentially be memory-mapped.
pub struct BufferByteSource(pub Buffer<u8>);

impl BufferByteSource {
    async fn try_new_mmap_from_path(
        path: &Path,
        _cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        let file = Arc::new(
            tokio::fs::File::open(path)
                .await
                .map_err(|err| _limit_path_len_io_err(path, err))?
                .into_std()
                .await,
        );

        Ok(Self(Buffer::from_owner(MMapSemaphore::new_from_file(
            &file,
        )?)))
    }
}

impl ByteSource for BufferByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.0.as_ref().len())
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Buffer<u8>> {
        let out = self.0.clone().sliced(range);
        Ok(out)
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, Buffer<u8>>> {
        Ok(ranges
            .iter()
            .map(|x| (x.start, self.0.clone().sliced(x.clone())))
            .collect())
    }
}

/// Byte source backed by a `File`.
pub struct FileByteSource {
    file: Arc<std::fs::File>,
    // Alignment for O_DIRECT, if supported.
    o_direct_align: Option<DioAlign>,
    // Manage concurrency.
    concurrency: usize,
    permits: Arc<Semaphore>,
    // File size.
    size: u64,
    io_metrics: OptIOMetrics,
}

#[derive(Clone)]
pub struct FileReadContext {
    pub enable_o_direct: bool,
    pub concurrency: usize,
    pub permits: Arc<Semaphore>,
}

impl std::fmt::Debug for FileReadContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileReadContext")
            .field("available_permits", &self.permits.available_permits())
            .field("enable_odirect", &self.enable_o_direct)
            .field("concurrency", &self.concurrency)
            .finish()
    }
}

#[cfg(target_os = "linux")]
fn _fadvise(file: &std::fs::File) {
    match *PLDEV_OPEN_FADV {
        0 => {}, //defaults to POSIX_FADV_NORMAL
        1 => unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        },
        2 => unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
        },
        3 => {
            unsafe {
                // Makes little sense in this context.
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED);
            }
        },
        _ => unreachable!("unexpected PLDEV_OPEN_FADV"),
    };
}

#[cfg(unix)]
pub(crate) fn pread_exact(
    file: &std::fs::File,
    buf: &mut [u8],
    offset: u64,
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn pread_exact(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut filled = 0;
    while filled < buf.len() {
        match file.seek_read(&mut buf[filled..], offset + filled as u64)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => filled += n,
        }
    }
    Ok(())
}

impl FileByteSource {
    async fn try_new_from_path(
        path: PlRefPath,
        read_context: FileReadContext,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Self> {
        let path = path.as_std_path();
        let enable_o_direct = read_context.enable_o_direct;

        let file = {
            #[cfg(target_os = "linux")]
            let f = if enable_o_direct {
                direct_io::open_o_direct(path)
            } else {
                std::fs::File::open(path)
            };
            #[cfg(not(target_os = "linux"))]
            let f = std::fs::File::open(path);

            f.map_err(|e| _limit_path_len_io_err(path, e))?
        };

        let file = Arc::new(file);

        #[cfg(target_os = "linux")]
        let o_direct_align = if enable_o_direct {
            direct_io::probe_fd(&file)
        } else {
            None
        };

        #[cfg(not(target_os = "linux"))]
        let o_direct_align = None;

        #[cfg(target_os = "linux")]
        if o_direct_align.is_none() {
            // Inert under O_DIRECT: there is no page cache to advise.
            _fadvise(&file);
        }

        let concurrency = read_context.concurrency;
        let permits = read_context.permits;

        let size = file
            .metadata()
            .map_err(|e| _limit_path_len_io_err(path, e))?
            .len();

        if polars_config::config().verbose() {
            let name = if polars_config::config().verbose_sensitive() {
                path.display().to_string()
            } else {
                "<file>".to_string()
            };
            match (enable_o_direct, o_direct_align) {
                (true, Some(a)) => eprintln!(
                    "[FileByteSource]: {name}: direct IO active, \
                        alignment offset: {}, memory: {}",
                    a.offset, a.memory
                ),
                (true, None) => eprintln!(
                    "[FileByteSource]: {name}: direct IO requested but not active, \
                        using buffered reads"
                ),
                (false, _) => {},
            }
        }

        Ok(FileByteSource {
            file,
            o_direct_align,
            concurrency,
            permits,
            size,
            io_metrics: OptIOMetrics(io_metrics),
        })
    }

    pub fn try_new_from_std(
        file: std::fs::File,
        read_context: FileReadContext,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Self> {
        let size = file.metadata()?.len();

        #[cfg(target_os = "linux")]
        let o_direct_align = {
            let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
            // Explicitly check it is enabled.
            (flags >= 0 && (flags & libc::O_DIRECT) != 0)
                .then(|| DioAlign::probe(&file))
                .flatten()
        };

        #[cfg(not(target_os = "linux"))]
        let o_direct_align = None;

        let concurrency = read_context.concurrency;
        let permits = read_context.permits;

        // For verbose logging only. We cannot enable since the file_handle was handed to us.
        let enable_o_direct = read_context.enable_o_direct;

        if polars_config::config().verbose() {
            match (enable_o_direct, o_direct_align) {
                (true, Some(a)) => eprintln!(
                    "[FileByteSource]: direct IO active, alignment offset: {}, memory: {}",
                    a.offset, a.memory
                ),
                (true, None) => eprintln!(
                    "[FileByteSource]: direct IO requested but not active, using buffered reads",
                ),
                (false, _) => {},
            }
        }

        Ok(Self {
            file: Arc::new(file),
            o_direct_align,
            concurrency,
            permits,
            size,
            io_metrics: OptIOMetrics(io_metrics),
        })
    }

    pub fn set_io_metrics(&mut self, io_metrics: Option<Arc<IOMetrics>>) -> &mut Self {
        self.io_metrics = OptIOMetrics(io_metrics);
        self
    }

    pub fn io_metrics(&self) -> &OptIOMetrics {
        &self.io_metrics
    }
}

impl ByteSource for FileByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        usize::try_from(self.size)
            .map_err(|_| polars_err!(ComputeError: "file size {} does not fit in usize", self.size))
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Buffer<u8>> {
        assert!(range.end as u64 <= self.size);

        let file = self.file.clone();
        let offset = range.start as u64;
        let len = range.len();
        let size = self.size;
        let o_direct = self.o_direct_align;

        let _permit = self.permits.acquire().await.unwrap();

        self.io_metrics()
            .record_io_read(len as u64, async move {
                tokio::task::spawn_blocking(move || -> PolarsResult<Buffer<u8>> {
                    #[cfg(target_os = "linux")]
                    if let Some(align) = o_direct {
                        return direct_io::read_aligned(&file, align, offset, len, size);
                    }
                    let mut buf = vec![0u8; len];
                    pread_exact(&file, &mut buf, offset)?;
                    Ok(Buffer::from(buf))
                })
                .await
            })
            .await
            .expect("blocking task panicked")
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, Buffer<u8>>> {
        if let [range] = ranges {
            let mut out = PlHashMap::with_capacity(1);
            out.insert(range.start, self.get_range(range.clone()).await?);
            return Ok(out);
        }

        // Sort by original start and merge.
        ranges.sort_unstable_by_key(|r| r.start);

        let mut spans: Vec<Range<usize>> = Vec::with_capacity(ranges.len());

        // Note: threshold for coalescing; individual ranges may exceed MAX_SPAN.
        const MAX_SPAN: usize = 8 << 20;
        let max_gap = self.o_direct_align.map_or(4096, |a| a.offset);

        for r in ranges.iter() {
            match spans.last_mut() {
                Some(last)
                    if r.start.saturating_sub(last.end) <= max_gap
                        && r.end.saturating_sub(last.start) <= MAX_SPAN =>
                {
                    last.end = last.end.max(r.end)
                },
                _ => spans.push(r.clone()),
            }
        }

        let mut fetched: Vec<(Range<usize>, Buffer<u8>)> = futures::stream::iter(spans)
            .map(|span| async move {
                let buf = self.get_range(span.clone()).await?;
                PolarsResult::Ok((span, buf))
            })
            .buffer_unordered(self.concurrency)
            .try_collect()
            .await?;

        // Slice out of the containing span into the original ranges.
        // Sort enables binary search.
        fetched.sort_unstable_by_key(|(s, _)| s.start);

        let mut out = PlHashMap::with_capacity(ranges.len());
        for r in ranges.iter() {
            let idx = fetched
                .binary_search_by(|(s, _)| {
                    if s.end <= r.start {
                        Ordering::Less
                    } else if s.start > r.start {
                        Ordering::Greater
                    } else {
                        Ordering::Equal
                    }
                })
                .expect("every range lies within a span");

            let (span, buf) = &fetched[idx];

            let off = r.start - span.start;
            out.insert(r.start, buf.clone().sliced(off..off + r.len()));
        }

        debug_assert_eq!(out.len(), ranges.len());
        Ok(out)
    }
}

#[cfg(feature = "cloud")]
pub struct ObjectStoreByteSource {
    store: PolarsObjectStore,
    path: ObjectStorePath,
    config: FetchConfig,
}

#[cfg(feature = "cloud")]
impl ObjectStoreByteSource {
    async fn try_new_from_path(
        path: PlRefPath,
        cloud_options: Option<&CloudOptions>,
        io_metrics: Option<Arc<IOMetrics>>,
        config: FetchConfig,
    ) -> PolarsResult<Self> {
        let (CloudLocation { prefix, .. }, mut store) =
            build_object_store(path, cloud_options, false).await?;
        let path = object_path_from_str(&prefix)?;

        store.set_io_metrics(io_metrics);

        Ok(Self {
            store,
            path,
            config,
        })
    }

    #[allow(unused)]
    fn chunk_size(&self) -> usize {
        self.config.chunk_size
    }

    fn concurrency_strategy(&self) -> ConcurrencyStrategy {
        self.config.strategy
    }
}

#[cfg(feature = "cloud")]
impl ByteSource for ObjectStoreByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self
            .store
            .head(&self.path, self.concurrency_strategy())
            .await?
            .size as usize)
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Buffer<u8>> {
        self.store.get_range(&self.path, range, self.config).await
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, Buffer<u8>>> {
        self.store
            .get_ranges_sort(&self.path, ranges, self.config)
            .await
    }
}

/// Dynamic dispatch to async functions.
pub enum DynByteSource {
    Buffer(BufferByteSource),
    File(FileByteSource),
    #[cfg(feature = "cloud")]
    Cloud(ObjectStoreByteSource),
}

impl DynByteSource {
    pub fn variant_name(&self) -> &str {
        match self {
            Self::Buffer(_) => "Buffer",
            Self::File(_) => "File",
            #[cfg(feature = "cloud")]
            Self::Cloud(_) => "Cloud",
        }
    }

    pub fn is_cloud(&self) -> bool {
        match self {
            Self::Buffer(_) => false,
            Self::File(_) => false,
            #[cfg(feature = "cloud")]
            Self::Cloud(_) => true,
        }
    }

    pub fn chunk_size(&self) -> Option<usize> {
        match self {
            Self::Buffer(_) => None,
            Self::File(_) => None,
            #[cfg(feature = "cloud")]
            Self::Cloud(source) => Some(source.config.chunk_size),
        }
    }

    pub fn concurrency_strategy(&self) -> Option<ConcurrencyStrategy> {
        match self {
            Self::Buffer(_) => None,
            // Concurrency is handled directly inside FileByteSource.
            Self::File(_) => None,
            #[cfg(feature = "cloud")]
            Self::Cloud(source) => Some(source.concurrency_strategy()),
        }
    }
}

impl Default for DynByteSource {
    fn default() -> Self {
        Self::Buffer(BufferByteSource(Buffer::new()))
    }
}

impl ByteSource for DynByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        match self {
            Self::Buffer(v) => v.get_size().await,
            Self::File(v) => v.get_size().await,
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.get_size().await,
        }
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Buffer<u8>> {
        match self {
            Self::Buffer(v) => v.get_range(range).await,
            Self::File(v) => v.get_range(range).await,
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.get_range(range).await,
        }
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, Buffer<u8>>> {
        match self {
            Self::Buffer(v) => v.get_ranges(ranges).await,
            Self::File(v) => v.get_ranges(ranges).await,
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.get_ranges(ranges).await,
        }
    }
}

impl From<BufferByteSource> for DynByteSource {
    fn from(value: BufferByteSource) -> Self {
        Self::Buffer(value)
    }
}

impl From<FileByteSource> for DynByteSource {
    fn from(value: FileByteSource) -> Self {
        Self::File(value)
    }
}

#[cfg(feature = "cloud")]
impl From<ObjectStoreByteSource> for DynByteSource {
    fn from(value: ObjectStoreByteSource) -> Self {
        Self::Cloud(value)
    }
}

impl From<Buffer<u8>> for DynByteSource {
    fn from(value: Buffer<u8>) -> Self {
        Self::Buffer(BufferByteSource(value))
    }
}

#[derive(Clone, Debug)]
pub enum DynByteSourceBuilder {
    Mmap,
    /// Use std::fs::File positional read (pread).
    FilePread(FileReadContext),
    /// Supports both cloud and local files, requires cloud feature.
    ObjectStore(FetchConfig),
}

impl DynByteSourceBuilder {
    pub async fn try_build_from_path(
        &self,
        path: PlRefPath,
        cloud_options: Option<&CloudOptions>,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<DynByteSource> {
        Ok(match self {
            Self::Mmap => {
                BufferByteSource::try_new_mmap_from_path(path.as_std_path(), cloud_options)
                    .await?
                    .into()
            },
            Self::FilePread(read_context) => {
                FileByteSource::try_new_from_path(path, read_context.clone(), io_metrics)
                    .await?
                    .into()
            },
            Self::ObjectStore(fetch_config) => feature_gated!("cloud", {
                ObjectStoreByteSource::try_new_from_path(
                    path,
                    cloud_options,
                    io_metrics,
                    *fetch_config,
                )
                .await?
                .into()
            }),
        })
    }

    pub fn chunk_size(&self) -> Option<usize> {
        match self {
            Self::Mmap => None,
            Self::FilePread(_) => None,
            Self::ObjectStore(fetch_config) => Some(fetch_config.chunk_size),
        }
    }

    pub fn concurrency_strategy(&self) -> Option<&ConcurrencyStrategy> {
        match self {
            Self::Mmap => None,
            Self::FilePread(_) => None,
            Self::ObjectStore(fetch_config) => Some(&fetch_config.strategy),
        }
    }
}

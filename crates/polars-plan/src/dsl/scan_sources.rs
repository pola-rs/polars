use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::error::{PolarsResult, feature_gated};
use polars_error::polars_err;
use polars_io::cloud::CloudOptions;
#[cfg(feature = "cloud")]
use polars_io::file_cache::FileCacheEntry;
use polars_io::metrics::IOMetrics;
use polars_io::utils::byte_source::{DynByteSource, DynByteSourceBuilder};
use polars_io::{expand_paths, expand_paths_hive, expanded_from_single_directory};
use polars_utils::mmap::MMapSemaphore;
use polars_utils::pl_path::PlRefPath;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::UnifiedScanArgs;

#[cfg(feature = "serde")]
fn serialize_paths<S: Serializer>(paths: &Buffer<PlRefPath>, s: S) -> Result<S::Ok, S::Error> {
    paths.as_slice().serialize(s)
}

#[cfg(feature = "serde")]
fn deserialize_paths<'de, D: Deserializer<'de>>(d: D) -> Result<Buffer<PlRefPath>, D::Error> {
    let v: Vec<PlRefPath> = Deserialize::deserialize(d)?;
    Ok(Buffer::from(v))
}

/// Set of sources to scan from
///
/// This can either be a list of paths to files, opened files or in-memory buffers. Mixing of
/// buffers is not currently possible.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone)]
pub enum ScanSources {
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serialize_paths",
            deserialize_with = "deserialize_paths"
        )
    )]
    #[cfg_attr(feature = "dsl-schema", schemars(with = "Vec<PlRefPath>"))]
    Paths(Buffer<PlRefPath>),
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    Files(Arc<[File]>),
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    Buffers(Arc<[Buffer<u8>]>),
}

impl Debug for ScanSources {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Paths(p) => write!(f, "paths: {:?}", p.as_ref()),
            Self::Files(p) => write!(f, "files: {} files", p.len()),
            Self::Buffers(b) => write!(f, "buffers: {} in-memory-buffers", b.len()),
        }
    }
}

/// A reference to a single item in [`ScanSources`]
#[derive(Debug, Clone, Copy)]
pub enum ScanSourceRef<'a> {
    Path(&'a PlRefPath),
    File(&'a File),
    Buffer(&'a Buffer<u8>),
}

/// A single source to scan from
#[derive(Debug, Clone)]
pub enum ScanSource {
    Path(PlRefPath),
    File(Arc<File>),
    Buffer(Buffer<u8>),
}

impl ScanSource {
    pub fn from_sources(sources: ScanSources) -> Result<Self, ScanSources> {
        if sources.len() == 1 {
            match sources {
                ScanSources::Paths(ps) => Ok(Self::Path(ps.as_ref()[0].clone())),
                ScanSources::Files(fs) => {
                    assert_eq!(fs.len(), 1);
                    let ptr: *const File = Arc::into_raw(fs) as *const File;
                    // SAFETY: A [T] with length 1 can be interpreted as T
                    let f: Arc<File> = unsafe { Arc::from_raw(ptr) };

                    Ok(Self::File(f))
                },
                ScanSources::Buffers(bs) => Ok(Self::Buffer(bs.as_ref()[0].clone())),
            }
        } else {
            Err(sources)
        }
    }

    pub fn into_sources(self) -> ScanSources {
        match self {
            ScanSource::Path(p) => ScanSources::Paths(Buffer::from_iter([p])),
            ScanSource::File(f) => {
                let ptr: *const [File] = std::ptr::slice_from_raw_parts(Arc::into_raw(f), 1);
                // SAFETY: A T can be interpreted as [T] with length 1.
                let fs: Arc<[File]> = unsafe { Arc::from_raw(ptr) };
                ScanSources::Files(fs)
            },
            ScanSource::Buffer(m) => ScanSources::Buffers([m].into()),
        }
    }

    pub fn as_scan_source_ref(&self) -> ScanSourceRef<'_> {
        match self {
            ScanSource::Path(path) => ScanSourceRef::Path(path),
            ScanSource::File(file) => ScanSourceRef::File(file.as_ref()),
            ScanSource::Buffer(mem_slice) => ScanSourceRef::Buffer(mem_slice),
        }
    }

    pub fn run_async(&self) -> bool {
        self.as_scan_source_ref().run_async()
    }

    pub fn is_cloud_url(&self) -> bool {
        if let ScanSource::Path(path) = self {
            path.has_scheme()
        } else {
            false
        }
    }
}

/// An iterator for [`ScanSources`]
pub struct ScanSourceIter<'a> {
    sources: &'a ScanSources,
    offset: usize,
}

impl Default for ScanSources {
    fn default() -> Self {
        // We need to use `Paths` here to avoid erroring when doing hive-partitioned scans of empty
        // file lists.
        Self::Paths(Buffer::new())
    }
}

impl std::hash::Hash for ScanSources {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        // @NOTE: This is a bit crazy
        //
        // We don't really want to hash the file descriptors or the whole buffers so for now we
        // just settle with the fact that the memory behind Arc's does not really move. Therefore,
        // we can just hash the pointer.
        match self {
            Self::Paths(paths) => paths.hash(state),
            Self::Files(files) => files.as_ptr().hash(state),
            Self::Buffers(buffers) => buffers.as_ptr().hash(state),
        }
    }
}

impl PartialEq for ScanSources {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScanSources::Paths(l), ScanSources::Paths(r)) => l == r,
            (ScanSources::Files(l), ScanSources::Files(r)) => std::ptr::eq(l.as_ptr(), r.as_ptr()),
            (ScanSources::Buffers(l), ScanSources::Buffers(r)) => {
                std::ptr::eq(l.as_ptr(), r.as_ptr())
            },
            _ => false,
        }
    }
}

impl Eq for ScanSources {}

impl ScanSources {
    pub async fn expand_paths(&self, scan_args: &mut UnifiedScanArgs) -> PolarsResult<Self> {
        match self {
            Self::Paths(paths) => Ok(Self::Paths(
                expand_paths(
                    paths,
                    scan_args.glob,
                    scan_args.hidden_file_prefix.as_deref().unwrap_or_default(),
                    &mut scan_args.cloud_options,
                )
                .await?,
            )),
            v => Ok(v.clone()),
        }
    }

    /// This will update `scan_args.hive_options.enabled` to `true` if the existing value is `None`
    /// and the paths are expanded from a single directory. Otherwise the existing value is maintained.
    #[cfg(any(feature = "ipc", feature = "parquet"))]
    pub async fn expand_paths_with_hive_update(
        &self,
        scan_args: &mut UnifiedScanArgs,
    ) -> PolarsResult<Self> {
        match self {
            Self::Paths(paths) => {
                let (expanded_paths, hive_start_idx) = expand_paths_hive(
                    paths,
                    scan_args.glob,
                    scan_args.hidden_file_prefix.as_deref().unwrap_or_default(),
                    &mut scan_args.cloud_options,
                    scan_args.hive_options.enabled.unwrap_or(false),
                )
                .await?;

                if scan_args.hive_options.enabled.is_none()
                    && expanded_from_single_directory(paths, expanded_paths.as_ref())
                {
                    scan_args.hive_options.enabled = Some(true);
                }
                scan_args.hive_options.hive_start_idx = hive_start_idx;

                Ok(Self::Paths(expanded_paths))
            },
            v => Ok(v.clone()),
        }
    }

    pub fn iter(&self) -> ScanSourceIter<'_> {
        ScanSourceIter {
            sources: self,
            offset: 0,
        }
    }

    /// Are the sources all paths?
    pub fn is_paths(&self) -> bool {
        matches!(self, Self::Paths(_))
    }

    /// Try cast the scan sources to [`ScanSources::Paths`]
    pub fn as_paths(&self) -> Option<&[PlRefPath]> {
        match self {
            Self::Paths(paths) => Some(paths.as_ref()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Try cast the scan sources to [`ScanSources::Paths`] with a clone
    pub fn into_paths(&self) -> Option<Buffer<PlRefPath>> {
        match self {
            Self::Paths(paths) => Some(paths.clone()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Try get the first path in the scan sources
    pub fn first_path(&self) -> Option<&PlRefPath> {
        match self {
            Self::Paths(paths) => paths.first(),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Is the first path a cloud URL?
    pub fn is_cloud_url(&self) -> bool {
        self.first_path().is_some_and(|path| path.has_scheme())
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Paths(s) => s.len(),
            Self::Files(s) => s.len(),
            Self::Buffers(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first(&self) -> Option<ScanSourceRef<'_>> {
        self.get(0)
    }

    pub fn first_or_empty_expand_err(
        &self,
        failed_message: &'static str,
        sources_before_expansion: &ScanSources,
        glob: bool,
        hint: &'static str,
    ) -> PolarsResult<ScanSourceRef<'_>> {
        let hint_padding = if hint.is_empty() { "" } else { " Hint: " };

        self.first().ok_or_else(|| match self {
            Self::Paths(_) if !sources_before_expansion.is_empty() => polars_err!(
                ComputeError:
                "{}: expanded paths were empty \
                (path expansion input: '{:?}', glob: {}).{}{}",
                failed_message, sources_before_expansion, glob, hint_padding, hint
            ),
            _ => polars_err!(
                ComputeError:
                "{}: empty input: {:?}.{}{}",
                failed_message, self, hint_padding, hint
            ),
        })
    }

    /// Turn the [`ScanSources`] into some kind of identifier
    pub fn id(&self) -> PlSmallStr {
        if self.is_empty() {
            return PlSmallStr::from_static("EMPTY");
        }

        match self {
            Self::Paths(paths) => PlSmallStr::from_str(paths.first().unwrap().as_str()),
            Self::Files(_) => PlSmallStr::from_static("OPEN_FILES"),
            Self::Buffers(_) => PlSmallStr::from_static("IN_MEMORY"),
        }
    }

    /// Get the scan source at specific address
    pub fn get(&self, idx: usize) -> Option<ScanSourceRef<'_>> {
        match self {
            Self::Paths(paths) => paths.get(idx).map(ScanSourceRef::Path),
            Self::Files(files) => files.get(idx).map(ScanSourceRef::File),
            Self::Buffers(buffers) => buffers.get(idx).map(ScanSourceRef::Buffer),
        }
    }

    /// Get the scan source at specific address
    ///
    /// # Panics
    ///
    /// If the `idx` is out of range.
    #[track_caller]
    pub fn at(&self, idx: usize) -> ScanSourceRef<'_> {
        self.get(idx).unwrap()
    }

    /// Returns `None` if `self` is a `::File` variant.
    pub fn gather(&self, indices: impl Iterator<Item = usize>) -> Option<Self> {
        Some(match self {
            Self::Paths(paths) => Self::Paths(indices.map(|i| paths[i].clone()).collect()),
            Self::Buffers(buffers) => Self::Buffers(indices.map(|i| buffers[i].clone()).collect()),
            Self::Files(_) => return None,
        })
    }
}

impl ScanSourceRef<'_> {
    /// Get the name for `include_paths`
    pub fn to_include_path_name(&self) -> &str {
        match self {
            Self::Path(path) => path.as_str(),
            Self::File(_) => "open-file",
            Self::Buffer(_) => "in-mem",
        }
    }

    // @TODO: I would like to remove this function eventually.
    pub fn into_owned(&self) -> PolarsResult<ScanSource> {
        Ok(match self {
            ScanSourceRef::Path(path) => ScanSource::Path((*path).clone()),
            ScanSourceRef::File(file) => {
                if let Ok(file) = file.try_clone() {
                    ScanSource::File(Arc::new(file))
                } else {
                    ScanSource::Buffer(self.to_memslice()?)
                }
            },
            ScanSourceRef::Buffer(buffer) => ScanSource::Buffer((*buffer).clone()),
        })
    }

    pub fn as_path(&self) -> Option<&PlRefPath> {
        match self {
            Self::Path(path) => Some(path),
            Self::File(_) | Self::Buffer(_) => None,
        }
    }

    pub fn is_cloud_url(&self) -> bool {
        self.as_path().is_some_and(|x| x.has_scheme())
    }

    /// Turn the scan source into a memory slice
    pub fn to_memslice(&self) -> PolarsResult<Buffer<u8>> {
        self.to_buffer_possibly_async(false, None, 0)
    }

    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "cloud")]
    fn to_buffer_async<F: Fn(Arc<FileCacheEntry>) -> PolarsResult<std::fs::File>>(
        &self,
        open_cache_entry: F,
        run_async: bool,
    ) -> PolarsResult<Buffer<u8>> {
        match self {
            ScanSourceRef::Path(path) => {
                let file = if run_async {
                    open_cache_entry(
                        polars_io::file_cache::FILE_CACHE
                            .get_entry((*path).clone())
                            .unwrap(),
                    )?
                } else {
                    polars_utils::open_file(path.as_std_path())?
                };

                Ok(Buffer::from_owner(MMapSemaphore::new_from_file(&file)?))
            },
            ScanSourceRef::File(file) => {
                Ok(Buffer::from_owner(MMapSemaphore::new_from_file(file)?))
            },
            ScanSourceRef::Buffer(buff) => Ok((*buff).clone()),
        }
    }

    #[cfg(feature = "cloud")]
    pub fn to_buffer_async_assume_latest(&self, run_async: bool) -> PolarsResult<Buffer<u8>> {
        self.to_buffer_async(|entry| entry.try_open_assume_latest(), run_async)
    }

    #[cfg(feature = "cloud")]
    pub fn to_buffer_async_check_latest(&self, run_async: bool) -> PolarsResult<Buffer<u8>> {
        self.to_buffer_async(|entry| entry.try_open_check_latest(), run_async)
    }

    #[cfg(not(feature = "cloud"))]
    #[allow(clippy::wrong_self_convention)]
    fn to_buffer_async(&self, run_async: bool) -> PolarsResult<Buffer<u8>> {
        match self {
            ScanSourceRef::Path(path) => {
                let file = polars_utils::open_file(path.as_std_path())?;
                Ok(Buffer::from_owner(MMapSemaphore::new_from_file(&file)?))
            },
            ScanSourceRef::File(file) => {
                Ok(Buffer::from_owner(MMapSemaphore::new_from_file(file)?))
            },
            ScanSourceRef::Buffer(buff) => Ok((*buff).clone()),
        }
    }

    #[cfg(not(feature = "cloud"))]
    pub fn to_buffer_async_assume_latest(&self, run_async: bool) -> PolarsResult<Buffer<u8>> {
        self.to_buffer_async(run_async)
    }

    #[cfg(not(feature = "cloud"))]
    pub fn to_buffer_async_check_latest(&self, run_async: bool) -> PolarsResult<Buffer<u8>> {
        self.to_buffer_async(run_async)
    }

    pub fn to_buffer_possibly_async(
        &self,
        run_async: bool,
        #[cfg(feature = "cloud")] cache_entries: Option<
            &Vec<Arc<polars_io::file_cache::FileCacheEntry>>,
        >,
        #[cfg(not(feature = "cloud"))] cache_entries: Option<&()>,
        index: usize,
    ) -> PolarsResult<Buffer<u8>> {
        match self {
            Self::Path(path) => {
                let file = if run_async {
                    feature_gated!("cloud", {
                        cache_entries.unwrap()[index].try_open_check_latest()?
                    })
                } else {
                    polars_utils::open_file(path.as_std_path())?
                };

                Ok(Buffer::from_owner(MMapSemaphore::new_from_file(&file)?))
            },
            Self::File(file) => Ok(Buffer::from_owner(MMapSemaphore::new_from_file(file)?)),
            Self::Buffer(buff) => Ok((*buff).clone()),
        }
    }

    pub async fn to_dyn_byte_source(
        &self,
        builder: &DynByteSourceBuilder,
        cloud_options: Option<&CloudOptions>,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<DynByteSource> {
        match self {
            Self::Path(path) => {
                builder
                    .try_build_from_path((*path).clone(), cloud_options, io_metrics)
                    .await
            },
            Self::File(file) => Ok(DynByteSource::from(Buffer::from_owner(
                MMapSemaphore::new_from_file(file)?,
            ))),
            Self::Buffer(buff) => Ok(DynByteSource::from((*buff).clone())),
        }
    }

    pub fn run_async(&self) -> bool {
        matches!(self, Self::Path(p) if p.has_scheme() || polars_config::config().force_async())
    }
}

impl<'a> Iterator for ScanSourceIter<'a> {
    type Item = ScanSourceRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = match self.sources {
            ScanSources::Paths(paths) => ScanSourceRef::Path(paths.get(self.offset)?),
            ScanSources::Files(files) => ScanSourceRef::File(files.get(self.offset)?),
            ScanSources::Buffers(buffers) => ScanSourceRef::Buffer(buffers.get(self.offset)?),
        };

        self.offset += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.sources.len() - self.offset;
        (len, Some(len))
    }
}

impl ExactSizeIterator for ScanSourceIter<'_> {}

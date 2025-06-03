use std::borrow::Cow;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars_core::error::{PolarsResult, feature_gated};
use polars_io::cloud::CloudOptions;
#[cfg(feature = "cloud")]
use polars_io::file_cache::FileCacheEntry;
#[cfg(feature = "cloud")]
use polars_io::utils::byte_source::{DynByteSource, DynByteSourceBuilder};
use polars_io::{expand_paths, expand_paths_hive, expanded_from_single_directory};
use polars_utils::address::{Address, AddressRef};
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;

use super::UnifiedScanArgs;

/// Set of sources to scan from
///
/// This can either be a list of paths to files, opened files or in-memory buffers. Mixing of
/// buffers is not currently possible.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone)]
pub enum ScanSources {
    Addresses(Arc<[Address]>),

    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    Files(Arc<[File]>),
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    Buffers(Arc<[MemSlice]>),
}

impl Debug for ScanSources {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Addresses(p) => write!(f, "addresses: {:?}", p.as_ref()),
            Self::Files(p) => write!(f, "files: {} files", p.len()),
            Self::Buffers(b) => write!(f, "buffers: {} in-memory-buffers", b.len()),
        }
    }
}

/// A reference to a single item in [`ScanSources`]
#[derive(Debug, Clone, Copy)]
pub enum ScanSourceRef<'a> {
    Address(AddressRef<'a>),
    File(&'a File),
    Buffer(&'a MemSlice),
}

/// A single source to scan from
#[derive(Debug, Clone)]
pub enum ScanSource {
    Address(Arc<Address>),
    File(Arc<File>),
    Buffer(MemSlice),
}

impl ScanSource {
    pub fn from_sources(sources: ScanSources) -> Result<Self, ScanSources> {
        if sources.len() == 1 {
            match sources {
                ScanSources::Addresses(ps) => Ok(Self::Address(ps.as_ref()[0].clone().into())),
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
            ScanSource::Address(p) => ScanSources::Addresses([p.as_ref().clone()].into()),
            ScanSource::File(f) => {
                let ptr: *const [File] = std::ptr::slice_from_raw_parts(Arc::into_raw(f), 1);
                // SAFETY: A T can be interpreted as [T] with length 1.
                let fs: Arc<[File]> = unsafe { Arc::from_raw(ptr) };
                ScanSources::Files(fs)
            },
            ScanSource::Buffer(m) => ScanSources::Buffers([m].into()),
        }
    }

    pub fn as_scan_source_ref(&self) -> ScanSourceRef {
        match self {
            ScanSource::Address(addr) => ScanSourceRef::Address(addr.as_ref().as_ref()),
            ScanSource::File(file) => ScanSourceRef::File(file.as_ref()),
            ScanSource::Buffer(mem_slice) => ScanSourceRef::Buffer(mem_slice),
        }
    }

    pub fn run_async(&self) -> bool {
        self.as_scan_source_ref().run_async()
    }

    pub fn is_cloud_url(&self) -> bool {
        if let ScanSource::Address(addr) = self {
            addr.is_cloud_url()
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
        // We need to use `Addresses` here to avoid erroring when doing hive-partitioned scans of empty
        // file lists.
        Self::Addresses(Arc::default())
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
            Self::Addresses(addrs) => addrs.hash(state),
            Self::Files(files) => files.as_ptr().hash(state),
            Self::Buffers(buffers) => buffers.as_ptr().hash(state),
        }
    }
}

impl PartialEq for ScanSources {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScanSources::Addresses(l), ScanSources::Addresses(r)) => l == r,
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
    pub fn expand_paths(
        &self,
        scan_args: &UnifiedScanArgs,
        #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        match self {
            Self::Addresses(addrs) => Ok(Self::Addresses(expand_paths(
                addrs,
                scan_args.glob,
                cloud_options,
            )?)),
            v => Ok(v.clone()),
        }
    }

    /// This will update `scan_args.hive_options.enabled` to `true` if the existing value is `None`
    /// and the paths are expanded from a single directory. Otherwise the existing value is maintained.
    #[cfg(any(feature = "ipc", feature = "parquet"))]
    pub fn expand_paths_with_hive_update(
        &self,
        scan_args: &mut UnifiedScanArgs,
        #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        match self {
            Self::Addresses(addrs) => {
                let (expanded_paths, hive_start_idx) = expand_paths_hive(
                    addrs,
                    scan_args.glob,
                    cloud_options,
                    scan_args.hive_options.enabled.unwrap_or(false),
                )?;

                if scan_args.hive_options.enabled.is_none()
                    && expanded_from_single_directory(addrs, expanded_paths.as_ref())
                {
                    scan_args.hive_options.enabled = Some(true);
                }
                scan_args.hive_options.hive_start_idx = hive_start_idx;

                Ok(Self::Addresses(expanded_paths))
            },
            v => Ok(v.clone()),
        }
    }

    pub fn iter(&self) -> ScanSourceIter {
        ScanSourceIter {
            sources: self,
            offset: 0,
        }
    }

    /// Are the sources all addresses?
    pub fn is_addresses(&self) -> bool {
        matches!(self, Self::Addresses(_))
    }

    /// Try cast the scan sources to [`ScanSources::Addresses`]
    pub fn as_addresses(&self) -> Option<&[Address]> {
        match self {
            Self::Addresses(addrs) => Some(addrs.as_ref()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Try cast the scan sources to [`ScanSources::Addresses`] with a clone
    pub fn into_addresses(&self) -> Option<Arc<[Address]>> {
        match self {
            Self::Addresses(addrs) => Some(addrs.clone()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Try get the first address in the scan sources
    pub fn first_address(&self) -> Option<AddressRef> {
        match self {
            Self::Addresses(addrs) => addrs.first().map(|p| p.as_ref()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Is the first address a cloud URL?
    pub fn is_cloud_url(&self) -> bool {
        self.first_address().is_some_and(|addr| addr.is_cloud_url())
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Addresses(s) => s.len(),
            Self::Files(s) => s.len(),
            Self::Buffers(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first(&self) -> Option<ScanSourceRef> {
        self.get(0)
    }

    /// Turn the [`ScanSources`] into some kind of identifier
    pub fn id(&self) -> PlSmallStr {
        if self.is_empty() {
            return PlSmallStr::from_static("EMPTY");
        }

        match self {
            Self::Addresses(addrs) => PlSmallStr::from_str(&addrs.first().unwrap().to_str()),
            Self::Files(_) => PlSmallStr::from_static("OPEN_FILES"),
            Self::Buffers(_) => PlSmallStr::from_static("IN_MEMORY"),
        }
    }

    /// Get the scan source at specific address
    pub fn get(&self, idx: usize) -> Option<ScanSourceRef> {
        match self {
            Self::Addresses(addrs) => addrs.get(idx).map(|p| ScanSourceRef::Address(p.as_ref())),
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
    pub fn at(&self, idx: usize) -> ScanSourceRef {
        self.get(idx).unwrap()
    }
}

impl ScanSourceRef<'_> {
    /// Get the name for `include_paths`
    pub fn to_include_path_name(&self) -> Cow<str> {
        match self {
            Self::Address(addr) => addr.to_str(),
            Self::File(_) => Cow::Borrowed("open-file"),
            Self::Buffer(_) => Cow::Borrowed("in-mem"),
        }
    }

    // @TODO: I would like to remove this function eventually.
    pub fn into_owned(&self) -> PolarsResult<ScanSource> {
        Ok(match self {
            ScanSourceRef::Address(addr) => ScanSource::Address((*addr).into_owned().into()),
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

    /// Turn the scan source into a memory slice
    pub fn to_memslice(&self) -> PolarsResult<MemSlice> {
        self.to_memslice_possibly_async(false, None, 0)
    }

    #[allow(clippy::wrong_self_convention)]
    #[cfg(feature = "cloud")]
    fn to_memslice_async<F: Fn(Arc<FileCacheEntry>) -> PolarsResult<std::fs::File>>(
        &self,
        assume: F,
        run_async: bool,
    ) -> PolarsResult<MemSlice> {
        match self {
            ScanSourceRef::Address(addr) => {
                let file = if run_async {
                    feature_gated!("cloud", {
                        // This isn't filled if we modified the DSL (e.g. in cloud)
                        let entry = polars_io::file_cache::FILE_CACHE.get_entry(*addr);

                        if let Some(entry) = entry {
                            assume(entry)?
                        } else {
                            polars_utils::open_file(addr.as_local_path().unwrap())?
                        }
                    })
                } else {
                    polars_utils::open_file(addr.as_local_path().unwrap())?
                };

                MemSlice::from_file(&file)
            },
            ScanSourceRef::File(file) => MemSlice::from_file(file),
            ScanSourceRef::Buffer(buff) => Ok((*buff).clone()),
        }
    }

    #[cfg(feature = "cloud")]
    pub fn to_memslice_async_assume_latest(&self, run_async: bool) -> PolarsResult<MemSlice> {
        self.to_memslice_async(|entry| entry.try_open_assume_latest(), run_async)
    }

    #[cfg(feature = "cloud")]
    pub fn to_memslice_async_check_latest(&self, run_async: bool) -> PolarsResult<MemSlice> {
        self.to_memslice_async(|entry| entry.try_open_check_latest(), run_async)
    }

    #[cfg(not(feature = "cloud"))]
    fn to_memslice_async(&self, run_async: bool) -> PolarsResult<MemSlice> {
        match self {
            ScanSourceRef::Address(addr) => {
                let file = polars_utils::open_file(addr.as_local_path().unwrap())?;
                MemSlice::from_file(&file)
            },
            ScanSourceRef::File(file) => MemSlice::from_file(file),
            ScanSourceRef::Buffer(buff) => Ok((*buff).clone()),
        }
    }

    #[cfg(not(feature = "cloud"))]
    pub fn to_memslice_async_assume_latest(&self, run_async: bool) -> PolarsResult<MemSlice> {
        self.to_memslice_async(run_async)
    }

    #[cfg(not(feature = "cloud"))]
    pub fn to_memslice_async_check_latest(&self, run_async: bool) -> PolarsResult<MemSlice> {
        self.to_memslice_async(run_async)
    }

    pub fn to_memslice_possibly_async(
        &self,
        run_async: bool,
        #[cfg(feature = "cloud")] cache_entries: Option<
            &Vec<Arc<polars_io::file_cache::FileCacheEntry>>,
        >,
        #[cfg(not(feature = "cloud"))] cache_entries: Option<&()>,
        index: usize,
    ) -> PolarsResult<MemSlice> {
        match self {
            Self::Address(addr) => {
                let file = if run_async {
                    feature_gated!("cloud", {
                        cache_entries.unwrap()[index].try_open_check_latest()?
                    })
                } else {
                    polars_utils::open_file(addr.as_local_path().unwrap())?
                };

                MemSlice::from_file(&file)
            },
            Self::File(file) => MemSlice::from_file(file),
            Self::Buffer(buff) => Ok((*buff).clone()),
        }
    }

    #[cfg(feature = "cloud")]
    pub async fn to_dyn_byte_source(
        &self,
        builder: &DynByteSourceBuilder,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<DynByteSource> {
        match self {
            Self::Address(addr) => {
                builder
                    .try_build_from_path(&addr.to_str(), cloud_options)
                    .await
            },
            Self::File(file) => Ok(DynByteSource::from(MemSlice::from_file(file)?)),
            Self::Buffer(buff) => Ok(DynByteSource::from((*buff).clone())),
        }
    }

    pub(crate) fn run_async(&self) -> bool {
        matches!(self, Self::Address(p) if p.is_cloud_url() || polars_core::config::force_async())
    }
}

impl<'a> Iterator for ScanSourceIter<'a> {
    type Item = ScanSourceRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = match self.sources {
            ScanSources::Addresses(addrs) => ScanSourceRef::Address(addrs.get(self.offset)?.as_ref()),
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

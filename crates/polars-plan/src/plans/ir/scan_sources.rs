use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars_core::error::{feature_gated, PolarsResult};
use polars_io::cloud::CloudOptions;
#[cfg(feature = "cloud")]
use polars_io::utils::byte_source::{DynByteSource, DynByteSourceBuilder};
use polars_io::{expand_paths, expand_paths_hive, expanded_from_single_directory};
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;

use super::FileScanOptions;

/// Set of sources to scan from
///
/// This can either be a list of paths to files, opened files or in-memory buffers. Mixing of
/// buffers is not currently possible.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub enum ScanSources {
    Paths(Arc<[PathBuf]>),

    #[cfg_attr(feature = "serde", serde(skip))]
    Files(Arc<[File]>),
    #[cfg_attr(feature = "serde", serde(skip))]
    Buffers(Arc<[bytes::Bytes]>),
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
    Path(&'a Path),
    File(&'a File),
    Buffer(&'a bytes::Bytes),
}

/// An iterator for [`ScanSources`]
pub struct ScanSourceIter<'a> {
    sources: &'a ScanSources,
    offset: usize,
}

impl Default for ScanSources {
    fn default() -> Self {
        Self::Buffers(Arc::default())
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
    pub fn expand_paths(
        &self,
        file_options: &FileScanOptions,
        #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        match self {
            Self::Paths(paths) => Ok(Self::Paths(expand_paths(
                paths,
                file_options.glob,
                cloud_options,
            )?)),
            v => Ok(v.clone()),
        }
    }

    #[cfg(any(feature = "ipc", feature = "parquet"))]
    pub fn expand_paths_with_hive_update(
        &self,
        file_options: &mut FileScanOptions,
        #[allow(unused_variables)] cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        match self {
            Self::Paths(paths) => {
                let hive_enabled = file_options.hive_options.enabled;
                let (expanded_paths, hive_start_idx) = expand_paths_hive(
                    paths,
                    file_options.glob,
                    cloud_options,
                    hive_enabled.unwrap_or(false),
                )?;
                let inferred_hive_enabled = hive_enabled.unwrap_or_else(|| {
                    expanded_from_single_directory(paths, expanded_paths.as_ref())
                });

                file_options.hive_options.enabled = Some(inferred_hive_enabled);
                file_options.hive_options.hive_start_idx = hive_start_idx;

                Ok(Self::Paths(expanded_paths))
            },
            v => {
                file_options.hive_options.enabled = Some(false);
                Ok(v.clone())
            },
        }
    }

    pub fn iter(&self) -> ScanSourceIter {
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
    pub fn as_paths(&self) -> Option<&[PathBuf]> {
        match self {
            Self::Paths(paths) => Some(paths.as_ref()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Try cast the scan sources to [`ScanSources::Paths`] with a clone
    pub fn into_paths(&self) -> Option<Arc<[PathBuf]>> {
        match self {
            Self::Paths(paths) => Some(paths.clone()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Try get the first path in the scan sources
    pub fn first_path(&self) -> Option<&Path> {
        match self {
            Self::Paths(paths) => paths.first().map(|p| p.as_path()),
            Self::Files(_) | Self::Buffers(_) => None,
        }
    }

    /// Is the first path a cloud URL?
    pub fn is_cloud_url(&self) -> bool {
        self.first_path().is_some_and(polars_io::is_cloud_url)
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

    pub fn first(&self) -> Option<ScanSourceRef> {
        self.get(0)
    }

    /// Turn the [`ScanSources`] into some kind of identifier
    pub fn id(&self) -> PlSmallStr {
        if self.is_empty() {
            return PlSmallStr::from_static("EMPTY");
        }

        match self {
            Self::Paths(paths) => {
                PlSmallStr::from_str(paths.first().unwrap().to_string_lossy().as_ref())
            },
            Self::Files(_) => PlSmallStr::from_static("OPEN_FILES"),
            Self::Buffers(_) => PlSmallStr::from_static("IN_MEMORY"),
        }
    }

    /// Get the scan source at specific address
    pub fn get(&self, idx: usize) -> Option<ScanSourceRef> {
        match self {
            Self::Paths(paths) => paths.get(idx).map(|p| ScanSourceRef::Path(p)),
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

impl<'a> ScanSourceRef<'a> {
    /// Get the name for `include_paths`
    pub fn to_include_path_name(&self) -> &str {
        match self {
            Self::Path(path) => path.to_str().unwrap(),
            Self::File(_) => "open-file",
            Self::Buffer(_) => "in-mem",
        }
    }

    /// Turn the scan source into a memory slice
    pub fn to_memslice(&self) -> PolarsResult<MemSlice> {
        self.to_memslice_possibly_async(false, None, 0)
    }

    pub fn to_memslice_async_latest(&self, run_async: bool) -> PolarsResult<MemSlice> {
        match self {
            ScanSourceRef::Path(path) => {
                let file = if run_async {
                    feature_gated!("cloud", {
                        polars_io::file_cache::FILE_CACHE
                            .get_entry(path.to_str().unwrap())
                            // Safety: This was initialized by schema inference.
                            .unwrap()
                            .try_open_assume_latest()?
                    })
                } else {
                    polars_utils::open_file(path)?
                };

                MemSlice::from_file(&file)
            },
            ScanSourceRef::File(file) => MemSlice::from_file(file),
            ScanSourceRef::Buffer(buff) => Ok(MemSlice::from_bytes((*buff).clone())),
        }
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
            Self::Path(path) => {
                let file = if run_async {
                    feature_gated!("cloud", {
                        cache_entries.unwrap()[index].try_open_check_latest()?
                    })
                } else {
                    polars_utils::open_file(path)?
                };

                MemSlice::from_file(&file)
            },
            Self::File(file) => MemSlice::from_file(file),
            Self::Buffer(buff) => Ok(MemSlice::from_bytes((*buff).clone())),
        }
    }

    #[cfg(feature = "cloud")]
    pub async fn to_dyn_byte_source(
        &self,
        builder: &DynByteSourceBuilder,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<DynByteSource> {
        match self {
            Self::Path(path) => {
                builder
                    .try_build_from_path(path.to_str().unwrap(), cloud_options)
                    .await
            },
            Self::File(file) => Ok(DynByteSource::from(MemSlice::from_file(file)?)),
            Self::Buffer(buff) => Ok(DynByteSource::from(MemSlice::from_bytes((*buff).clone()))),
        }
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

impl<'a> ExactSizeIterator for ScanSourceIter<'a> {}

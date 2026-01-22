use std::io::SeekFrom;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use polars_core::prelude::{InitHashMaps as _, PlHashMap};
use polars_error::PolarsResult;
use polars_utils::_limit_path_len_io_err;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_path::PlRefPath;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::cloud::{
    CloudLocation, CloudOptions, ObjectStorePath, PolarsObjectStore, build_object_store,
    object_path_from_str,
};

#[allow(async_fn_in_trait)]
pub trait ByteSource: Send + Sync {
    async fn get_size(&self) -> PolarsResult<usize>;
    /// # Panics
    /// Panics if `range` is not in bounds.
    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice>;
    /// Note: This will mutably sort ranges for coalescing.
    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, MemSlice>>;
}

/// Byte source backed by a `MemSlice`, which can potentially be memory-mapped.
pub struct MemSliceByteSource(pub MemSlice);

impl MemSliceByteSource {
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

        Ok(Self(MemSlice::from_file(file.as_ref())?))
    }
}

impl ByteSource for MemSliceByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.0.as_ref().len())
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice> {
        let out = self.0.slice(range);
        Ok(out)
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, MemSlice>> {
        Ok(ranges
            .iter()
            .map(|x| (x.start, self.0.slice(x.clone())))
            .collect())
    }
}

pub struct MmapCopyByteSource(pub MemSlice);

impl MmapCopyByteSource {
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

        Ok(Self(MemSlice::from_file(file.as_ref())?))
    }
}

impl ByteSource for MmapCopyByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.0.as_ref().len())
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice> {
        let out = self.0.slice(range);
        Ok(<[u8]>::to_vec(&out).into())
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, MemSlice>> {
        Ok(ranges
            .iter()
            .map(|x| (x.start, <[u8]>::to_vec(&self.0.slice(x.clone())).into()))
            .collect())
    }
}

pub struct ObjectStoreByteSource {
    store: PolarsObjectStore,
    path: ObjectStorePath,
}

impl ObjectStoreByteSource {
    async fn try_new_from_path(
        path: PlRefPath,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        let (CloudLocation { prefix, .. }, store) =
            build_object_store(path, cloud_options, false).await?;
        let path = object_path_from_str(&prefix)?;

        Ok(Self { store, path })
    }
}

impl ByteSource for ObjectStoreByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.store.head(&self.path).await?.size as usize)
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice> {
        let bytes = self.store.get_range(&self.path, range).await?;
        let mem_slice = MemSlice::from_bytes(bytes);

        Ok(mem_slice)
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, MemSlice>> {
        self.store.get_ranges_sort(&self.path, ranges).await
    }
}

pub struct AsyncFileByteSource(tokio::sync::Mutex<tokio::fs::File>);

impl AsyncFileByteSource {
    async fn try_new_from_path(
        path: &Path,
        _cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        let file = tokio::fs::File::open(path)
            .await
            .map_err(|err| _limit_path_len_io_err(path, err))?;

        Ok(Self(file.into()))
    }
}

impl ByteSource for AsyncFileByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.0.lock().await.metadata().await?.len() as usize)
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice> {
        let mut f = self.0.lock().await;

        f.seek(SeekFrom::Start(range.start as u64)).await?;
        let mut out: Vec<u8> = vec![0; range.len()];
        f.read_exact(&mut out).await?;

        Ok(out.into())
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, MemSlice>> {
        ranges.sort_unstable_by_key(|x| x.start);
        let mut bytes_map: PlHashMap<usize, MemSlice> = PlHashMap::with_capacity(ranges.len());

        let mut f = self.0.lock().await;

        for range in &*ranges {
            f.seek(SeekFrom::Start(range.start as u64)).await?;

            let mut out: Vec<u8> = vec![0; range.len()];

            f.read_exact(&mut out).await?;

            bytes_map.insert(range.start, out.into());
        }

        Ok(bytes_map)
    }
}

/// Dynamic dispatch to async functions.
pub enum DynByteSource {
    MemSlice(MemSliceByteSource),
    Cloud(ObjectStoreByteSource),
    MmapCopy(MmapCopyByteSource),
    AsyncFile(AsyncFileByteSource),
}

impl Default for DynByteSource {
    fn default() -> Self {
        Self::MemSlice(MemSliceByteSource(MemSlice::default()))
    }
}

impl ByteSource for DynByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        match self {
            Self::MemSlice(v) => v.get_size().await,
            Self::Cloud(v) => v.get_size().await,
            Self::MmapCopy(v) => v.get_size().await,
            Self::AsyncFile(v) => v.get_size().await,
        }
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice> {
        match self {
            Self::MemSlice(v) => v.get_range(range).await,
            Self::Cloud(v) => v.get_range(range).await,
            Self::MmapCopy(v) => v.get_range(range).await,
            Self::AsyncFile(v) => v.get_range(range).await,
        }
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, MemSlice>> {
        match self {
            Self::MemSlice(v) => v.get_ranges(ranges).await,
            Self::Cloud(v) => v.get_ranges(ranges).await,
            Self::MmapCopy(v) => v.get_ranges(ranges).await,
            Self::AsyncFile(v) => v.get_ranges(ranges).await,
        }
    }
}

impl From<MemSliceByteSource> for DynByteSource {
    fn from(value: MemSliceByteSource) -> Self {
        Self::MemSlice(value)
    }
}

impl From<ObjectStoreByteSource> for DynByteSource {
    fn from(value: ObjectStoreByteSource) -> Self {
        Self::Cloud(value)
    }
}

impl From<MmapCopyByteSource> for DynByteSource {
    fn from(value: MmapCopyByteSource) -> Self {
        Self::MmapCopy(value)
    }
}

impl From<AsyncFileByteSource> for DynByteSource {
    fn from(value: AsyncFileByteSource) -> Self {
        Self::AsyncFile(value)
    }
}

impl From<MemSlice> for DynByteSource {
    fn from(value: MemSlice) -> Self {
        Self::MemSlice(MemSliceByteSource(value))
    }
}

#[derive(Clone, Debug)]
pub enum DynByteSourceBuilder {
    Mmap,
    /// Supports both cloud and local files.
    ObjectStore,
    MmapCopy,
    AsyncFile,
}

impl DynByteSourceBuilder {
    pub async fn try_build_from_path(
        &self,
        path: PlRefPath,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<DynByteSource> {
        Ok(match self {
            Self::Mmap => {
                MemSliceByteSource::try_new_mmap_from_path(path.as_std_path(), cloud_options)
                    .await?
                    .into()
            },
            Self::ObjectStore => ObjectStoreByteSource::try_new_from_path(path, cloud_options)
                .await?
                .into(),
            Self::MmapCopy => {
                MmapCopyByteSource::try_new_mmap_from_path(path.as_std_path(), cloud_options)
                    .await?
                    .into()
            },
            Self::AsyncFile => {
                AsyncFileByteSource::try_new_from_path(path.as_std_path(), cloud_options)
                    .await?
                    .into()
            },
        })
    }
}

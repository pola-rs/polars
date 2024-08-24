use std::ops::Range;
use std::sync::Arc;

use polars_error::{to_compute_err, PolarsResult};
use polars_utils::_limit_path_len_io_err;
use polars_utils::mmap::MemSlice;

use crate::cloud::{
    build_object_store, object_path_from_str, CloudLocation, CloudOptions, ObjectStorePath,
    PolarsObjectStore,
};

#[allow(async_fn_in_trait)]
pub trait ByteSource: Send + Sync {
    async fn get_size(&self) -> PolarsResult<usize>;
    /// # Panics
    /// Panics if `range` is not in bounds.
    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice>;
    async fn get_ranges(&self, ranges: &[Range<usize>]) -> PolarsResult<Vec<MemSlice>>;
}

/// Byte source backed by a `MemSlice`, which can potentially be memory-mapped.
pub struct MemSliceByteSource(pub MemSlice);

impl MemSliceByteSource {
    async fn try_new_mmap_from_path(
        path: &str,
        _cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        let file = Arc::new(
            tokio::fs::File::open(path)
                .await
                .map_err(|err| _limit_path_len_io_err(path.as_ref(), err))?
                .into_std()
                .await,
        );
        let mmap = Arc::new(unsafe { memmap::Mmap::map(file.as_ref()) }.map_err(to_compute_err)?);

        Ok(Self(MemSlice::from_mmap(mmap)))
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

    async fn get_ranges(&self, ranges: &[Range<usize>]) -> PolarsResult<Vec<MemSlice>> {
        Ok(ranges
            .iter()
            .map(|x| self.0.slice(x.clone()))
            .collect::<Vec<_>>())
    }
}

pub struct ObjectStoreByteSource {
    store: PolarsObjectStore,
    path: ObjectStorePath,
}

impl ObjectStoreByteSource {
    async fn try_new_from_path(
        path: &str,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        let (CloudLocation { prefix, .. }, store) =
            build_object_store(path, cloud_options, false).await?;
        let path = object_path_from_str(&prefix)?;
        let store = PolarsObjectStore::new(store);

        Ok(Self { store, path })
    }
}

impl ByteSource for ObjectStoreByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.store.head(&self.path).await?.size)
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice> {
        let bytes = self.store.get_range(&self.path, range).await?;
        let mem_slice = MemSlice::from_bytes(bytes);

        Ok(mem_slice)
    }

    async fn get_ranges(&self, ranges: &[Range<usize>]) -> PolarsResult<Vec<MemSlice>> {
        let ranges = self.store.get_ranges(&self.path, ranges).await?;
        Ok(ranges.into_iter().map(MemSlice::from_bytes).collect())
    }
}

/// Dynamic dispatch to async functions.
pub enum DynByteSource {
    MemSlice(MemSliceByteSource),
    Cloud(ObjectStoreByteSource),
}

impl DynByteSource {
    pub fn variant_name(&self) -> &str {
        match self {
            Self::MemSlice(_) => "MemSlice",
            Self::Cloud(_) => "Cloud",
        }
    }
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
        }
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<MemSlice> {
        match self {
            Self::MemSlice(v) => v.get_range(range).await,
            Self::Cloud(v) => v.get_range(range).await,
        }
    }

    async fn get_ranges(&self, ranges: &[Range<usize>]) -> PolarsResult<Vec<MemSlice>> {
        match self {
            Self::MemSlice(v) => v.get_ranges(ranges).await,
            Self::Cloud(v) => v.get_ranges(ranges).await,
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

#[derive(Clone, Debug)]
pub enum DynByteSourceBuilder {
    Mmap,
    /// Supports both cloud and local files.
    ObjectStore,
}

impl DynByteSourceBuilder {
    pub async fn try_build_from_path(
        &self,
        path: &str,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<DynByteSource> {
        Ok(match self {
            Self::Mmap => MemSliceByteSource::try_new_mmap_from_path(path, cloud_options)
                .await?
                .into(),
            Self::ObjectStore => ObjectStoreByteSource::try_new_from_path(path, cloud_options)
                .await?
                .into(),
        })
    }
}

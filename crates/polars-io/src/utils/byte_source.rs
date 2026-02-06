use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::prelude::PlHashMap;
use polars_error::{PolarsResult, feature_gated};
use polars_utils::_limit_path_len_io_err;
use polars_utils::mmap::MMapSemaphore;
use polars_utils::pl_path::PlRefPath;

use crate::cloud::options::CloudOptions;
#[cfg(feature = "cloud")]
use crate::cloud::{
    CloudLocation, ObjectStorePath, PolarsObjectStore, build_object_store, object_path_from_str,
};

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

#[cfg(feature = "cloud")]
pub struct ObjectStoreByteSource {
    store: PolarsObjectStore,
    path: ObjectStorePath,
}

#[cfg(feature = "cloud")]
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

#[cfg(feature = "cloud")]
impl ByteSource for ObjectStoreByteSource {
    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.store.head(&self.path).await?.size as usize)
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Buffer<u8>> {
        self.store.get_range(&self.path, range).await
    }

    async fn get_ranges(
        &self,
        ranges: &mut [Range<usize>],
    ) -> PolarsResult<PlHashMap<usize, Buffer<u8>>> {
        self.store.get_ranges_sort(&self.path, ranges).await
    }
}

/// Dynamic dispatch to async functions.
pub enum DynByteSource {
    Buffer(BufferByteSource),
    #[cfg(feature = "cloud")]
    Cloud(ObjectStoreByteSource),
}

impl DynByteSource {
    pub fn variant_name(&self) -> &str {
        match self {
            Self::Buffer(_) => "Buffer",
            #[cfg(feature = "cloud")]
            Self::Cloud(_) => "Cloud",
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
            #[cfg(feature = "cloud")]
            Self::Cloud(v) => v.get_size().await,
        }
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Buffer<u8>> {
        match self {
            Self::Buffer(v) => v.get_range(range).await,
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
    /// Supports both cloud and local files, requires cloud feature.
    ObjectStore,
}

impl DynByteSourceBuilder {
    pub async fn try_build_from_path(
        &self,
        path: PlRefPath,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<DynByteSource> {
        Ok(match self {
            Self::Mmap => {
                BufferByteSource::try_new_mmap_from_path(path.as_std_path(), cloud_options)
                    .await?
                    .into()
            },
            Self::ObjectStore => feature_gated!(
                "cloud",
                ObjectStoreByteSource::try_new_from_path(path, cloud_options)
                    .await?
                    .into()
            ),
        })
    }
}

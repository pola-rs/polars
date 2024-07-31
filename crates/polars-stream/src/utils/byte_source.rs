use std::ops::Range;
use std::sync::Arc;

use polars_error::{to_compute_err, PolarsResult};
use polars_io::cloud::{
    build_object_store, object_path_from_str, CloudLocation, CloudOptions, ObjectStorePath,
    PolarsObjectStore,
};
use polars_utils::{_limit_path_len_io_err, no_call_const};

use crate::async_executor;

mod bytes_private {
    use std::sync::Arc;

    /// This is for keeping the buffers alive.
    #[derive(Clone)]
    #[allow(unused)]
    pub(super) enum BytesKeepAlive {
        Local(Arc<memmap::Mmap>),
        Cloud(bytes::Bytes),
    }

    /// TODO: Add slicing to owned data
    #[derive(Clone)]
    pub struct Bytes {
        // Store the `&[u8]` to make `AsRef<[u8]>` free.
        // `slice` is not 'static - it is backed by `keep_alive`. This is safe as long as `slice` is
        // not directly accessed, and we are in a private module to guarantee that. Access should
        // only be done through `AsRef<[u8]>`, which automatically gives the correct lifetime.
        slice: &'static [u8],
        _keep_alive: BytesKeepAlive,
    }

    impl Bytes {
        pub(super) fn new(slice: &'static [u8], _keep_alive: BytesKeepAlive) -> Self {
            Self { slice, _keep_alive }
        }
    }

    impl AsRef<[u8]> for Bytes {
        fn as_ref(&self) -> &[u8] {
            self.slice
        }
    }
}

pub use bytes_private::Bytes;
use bytes_private::BytesKeepAlive;

pub trait ByteSource: Sized + Send + Sync {
    async fn try_new_from_path(
        path: &str,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self>;
    async fn get_size(&self) -> PolarsResult<usize>;
    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Bytes>;
}

pub struct LocalByteSource {
    mmap: Arc<memmap::Mmap>,
    _file: Arc<std::fs::File>,
}

impl ByteSource for LocalByteSource {
    async fn try_new_from_path(
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

        Ok(Self { mmap, _file: file })
    }

    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.mmap.as_ref().len())
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Bytes> {
        // TODO: Change from mmap to file-read - this currently doesn't trigger any reads.
        let slice = unsafe { std::mem::transmute(self.mmap.as_ref().as_ref().get(range).unwrap()) };

        Ok(Bytes::new(slice, BytesKeepAlive::Local(self.mmap.clone())))
    }
}

pub struct CloudByteSource {
    store: PolarsObjectStore,
    path: ObjectStorePath,
}

impl ByteSource for CloudByteSource {
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

    async fn get_size(&self) -> PolarsResult<usize> {
        Ok(self.store.head(&self.path).await?.size)
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Bytes> {
        let bytes = self.store.get_range(&self.path, range).await?;
        let keep_alive = BytesKeepAlive::Cloud(bytes.clone());

        Ok(Bytes::new(
            unsafe { std::mem::transmute::<_, &'static [u8]>(bytes.as_ref()) },
            keep_alive,
        ))
    }
}

/// We have this because traits with async functions aren't object-safe.
pub enum DynByteSource {
    Local(LocalByteSource),
    Cloud(CloudByteSource),
}

impl ByteSource for DynByteSource {
    async fn try_new_from_path(
        _path: &str,
        _cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Self> {
        // Use `DynByteSourceBuilder` instead.
        no_call_const!()
    }

    async fn get_size(&self) -> PolarsResult<usize> {
        match self {
            Self::Local(v) => v.get_size().await,
            Self::Cloud(v) => v.get_size().await,
        }
    }

    async fn get_range(&self, range: Range<usize>) -> PolarsResult<Bytes> {
        match self {
            Self::Local(v) => v.get_range(range).await,
            Self::Cloud(v) => v.get_range(range).await,
        }
    }
}

impl From<LocalByteSource> for DynByteSource {
    fn from(value: LocalByteSource) -> Self {
        Self::Local(value)
    }
}

impl From<CloudByteSource> for DynByteSource {
    fn from(value: CloudByteSource) -> Self {
        Self::Cloud(value)
    }
}

#[derive(Clone)]
pub enum DynByteSourceBuilder {
    Local,
    Cloud,
}

impl DynByteSourceBuilder {
    pub async fn try_build_from_path(
        &self,
        path: &str,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<DynByteSource> {
        Ok(match self {
            Self::Local => LocalByteSource::try_new_from_path(path, cloud_options)
                .await?
                .into(),
            Self::Cloud => CloudByteSource::try_new_from_path(path, cloud_options)
                .await?
                .into(),
        })
    }
}

mod tests {

    #[test]
    fn test() {
        use polars_io::cloud::CloudOptions;
        use polars_io::pl_async;

        use super::*;
        use crate::async_executor;

        // Needs env vars:
        // * aws_bucket
        // * aws_access_key_id
        // * aws_secret_access_key
        // * aws_region

        let bucket = std::env::var("aws_bucket").unwrap();
        let cloud_path = format!("s3://{bucket}/nxs/hive_date/date1=2024-01-01/date2=2023-01-01%2000%3A00%3A00.000000/ce0249bc8a1d412f87a7ca87977bd1f2-0.parquet");
        let cloud_path = cloud_path.as_str();

        let cloud_options = CloudOptions::from_untyped_config(
            cloud_path,
            vec![
                (
                    "aws_access_key_id",
                    std::env::var("aws_access_key_id").unwrap().as_str(),
                ),
                (
                    "aws_secret_access_key",
                    std::env::var("aws_secret_access_key").unwrap().as_str(),
                ),
                ("aws_region", std::env::var("aws_region").unwrap().as_str()),
            ],
        )
        .unwrap();

        pl_async::get_runtime().block_on(async {
            let local_result = read_parquet_metadata(
                &DynByteSourceBuilder::Local
                    .try_build_from_path("/Users/nxs/git/polars/.env/iris.parquet", None)
                    .await
                    .unwrap(),
                &async_executor::ExecutorScope::new(),
            )
            .await;

            dbg!(local_result.map(|x| x.num_rows));

            let cloud_result = read_parquet_metadata(
                &DynByteSourceBuilder::Cloud
                    .try_build_from_path(cloud_path, Some(&cloud_options))
                    .await
                    .unwrap(),
                &async_executor::ExecutorScope::new(),
            )
            .await;

            dbg!(cloud_result.map(|x| x.num_rows));
        });
    }
}

pub async fn read_parquet_metadata(
    byte_source: &DynByteSource,
    executor_scope: &async_executor::ExecutorScope<'static, 'static>,
) -> PolarsResult<polars_io::prelude::FileMetaData> {
    fn read_n<const N: usize>(reader: &mut &[u8]) -> Option<[u8; N]> {
        if N <= reader.len() {
            let (head, tail) = reader.split_at(N);
            *reader = tail;
            Some(head.try_into().unwrap())
        } else {
            None
        }
    }

    fn read_i32le(reader: &mut &[u8]) -> Option<i32> {
        read_n(reader).map(i32::from_le_bytes)
    }

    let file_byte_length = byte_source.get_size().await?;

    let footer_header_bytes = byte_source
        .get_range(
            file_byte_length
                .checked_sub(polars_parquet::parquet::FOOTER_SIZE as usize)
                .ok_or_else(|| {
                    polars_parquet::parquet::error::ParquetError::OutOfSpec(
                        "not enough bytes to contain parquet footer".to_string(),
                    )
                })?..file_byte_length,
        )
        .await?;

    let footer_byte_length: usize = {
        let reader = &mut footer_header_bytes.as_ref();
        let footer_byte_size = read_i32le(reader).unwrap();
        let magic = read_n(reader).unwrap();
        debug_assert!(reader.is_empty());
        if magic != polars_parquet::parquet::PARQUET_MAGIC {
            return Err(polars_parquet::parquet::error::ParquetError::OutOfSpec(
                "incorrect magic in parquet footer".to_string(),
            )
            .into());
        }
        footer_byte_size.try_into().map_err(|_| {
            polars_parquet::parquet::error::ParquetError::OutOfSpec(
                "negative footer byte length".to_string(),
            )
        })?
    };

    let footer_bytes = byte_source
        .get_range(
            file_byte_length
                .checked_sub(polars_parquet::parquet::FOOTER_SIZE as usize + footer_byte_length)
                .ok_or_else(|| {
                    polars_parquet::parquet::error::ParquetError::OutOfSpec(
                        "not enough bytes to contain parquet footer".to_string(),
                    )
                })?..file_byte_length,
        )
        .await?;

    let out = executor_scope
        .spawn_task(async_executor::TaskPriority::Low, async move {
            polars_parquet::parquet::read::deserialize_metadata(
                std::io::Cursor::new(footer_bytes.as_ref()),
                // TODO: Describe why this makes sense. Taken from the previous
                // implementation which said "a highly nested but sparse struct could
                // result in many allocations".
                footer_bytes.as_ref().len() * 2 + 1024,
            )
        })
        .await?;

    Ok(out)
}

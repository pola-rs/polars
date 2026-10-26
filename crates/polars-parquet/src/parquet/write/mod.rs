mod column_chunk;
mod compression;
mod file;
mod indexes;
pub(crate) mod page;
mod row_group;
mod statistics;

#[cfg(feature = "async")]
mod stream;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use stream::FileStreamer;

mod dyn_iter;
pub use compression::{Compressor, compress};
pub use dyn_iter::{DynIter, DynStreamingIterator};
pub use file::{FileWriter, write_metadata_sidecar};
use polars_parquet_format::thrift::protocol::{TCompactOutputProtocol, TOutputProtocol};
pub use row_group::ColumnOffsetsMetadata;

use crate::parquet::error::ParquetResult;
use crate::parquet::page::CompressedPage;

/// Thrift objects that can be serialized with the compact protocol.
pub(crate) trait WriteThrift {
    fn write_thrift<T: TOutputProtocol>(
        &self,
        protocol: &mut T,
    ) -> polars_parquet_format::thrift::Result<usize>;

    /// Serialize the object to bytes.
    fn to_thrift_bytes(&self) -> ParquetResult<Vec<u8>> {
        let mut buf = vec![];
        let mut protocol = TCompactOutputProtocol::new(&mut buf);
        self.write_thrift(&mut protocol)?;
        Ok(buf)
    }
}

macro_rules! impl_write_thrift {
    ($($t:ty),*) => {
        $(
            impl WriteThrift for $t {
                fn write_thrift<T: TOutputProtocol>(
                    &self,
                    protocol: &mut T,
                ) -> polars_parquet_format::thrift::Result<usize> {
                    self.write_to_out_protocol(protocol)
                }
            }
        )*
    };
}

impl_write_thrift!(
    polars_parquet_format::ColumnIndex,
    polars_parquet_format::ColumnMetaData,
    polars_parquet_format::FileCryptoMetaData,
    polars_parquet_format::FileMetaData,
    polars_parquet_format::OffsetIndex,
    polars_parquet_format::PageHeader
);

pub type RowGroupIterColumns<'a, E> =
    DynIter<'a, Result<DynStreamingIterator<'a, CompressedPage, E>, E>>;

pub type RowGroupIter<'a, E> = DynIter<'a, RowGroupIterColumns<'a, E>>;

/// Write options of different interfaces on this crate
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct WriteOptions {
    /// Whether to write statistics, including indexes
    pub write_statistics: bool,
    /// Which Parquet version to use
    pub version: Version,
}

/// The parquet version to use
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Version {
    V1,
    V2,
}

/// Used to recall the state of the parquet writer - whether sync or async.
#[derive(PartialEq)]
enum State {
    Initialised,
    Started,
    Finished,
}

impl From<Version> for i32 {
    fn from(version: Version) -> Self {
        match version {
            Version::V1 => 1,
            Version::V2 => 2,
        }
    }
}

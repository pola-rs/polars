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
pub use row_group::ColumnOffsetsMetadata;

use crate::parquet::page::CompressedPage;

pub type RowGroupIterColumns<'a, E> =
    DynIter<'a, Result<DynStreamingIterator<'a, CompressedPage, E>, E>>;

pub type RowGroupIter<'a, E> = DynIter<'a, RowGroupIterColumns<'a, E>>;

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

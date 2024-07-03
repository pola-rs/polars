use parquet_format_safe::{ColumnChunk, ColumnMetaData, Encoding};

use super::column_descriptor::ColumnDescriptor;
use crate::parquet::compression::Compression;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::schema::types::PhysicalType;
use crate::parquet::statistics::Statistics;

#[cfg(feature = "serde_types")]
mod serde_types {
    pub use std::io::Cursor;

    pub use parquet_format_safe::thrift::protocol::{
        TCompactInputProtocol, TCompactOutputProtocol,
    };
    pub use serde::de::Error as DeserializeError;
    pub use serde::ser::Error as SerializeError;
    pub use serde::{Deserialize, Deserializer, Serialize, Serializer};
}
#[cfg(feature = "serde_types")]
use serde_types::*;

/// Metadata for a column chunk.
// This contains the `ColumnDescriptor` associated with the chunk so that deserializers have
// access to the descriptor (e.g. physical, converted, logical).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_types", derive(Deserialize, Serialize))]
pub struct ColumnChunkMetaData {
    #[cfg_attr(
        feature = "serde_types",
        serde(serialize_with = "serialize_column_chunk")
    )]
    #[cfg_attr(
        feature = "serde_types",
        serde(deserialize_with = "deserialize_column_chunk")
    )]
    column_chunk: ColumnChunk,
    column_descr: ColumnDescriptor,
}

#[cfg(feature = "serde_types")]
fn serialize_column_chunk<S>(
    column_chunk: &ColumnChunk,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut buf = vec![];
    let cursor = Cursor::new(&mut buf[..]);
    let mut protocol = TCompactOutputProtocol::new(cursor);
    column_chunk
        .write_to_out_protocol(&mut protocol)
        .map_err(S::Error::custom)?;
    serializer.serialize_bytes(&buf)
}

#[cfg(feature = "serde_types")]
fn deserialize_column_chunk<'de, D>(deserializer: D) -> std::result::Result<ColumnChunk, D::Error>
where
    D: Deserializer<'de>,
{
    let buf = Vec::<u8>::deserialize(deserializer)?;
    let mut cursor = Cursor::new(&buf[..]);
    let mut protocol = TCompactInputProtocol::new(&mut cursor, usize::MAX);
    ColumnChunk::read_from_in_protocol(&mut protocol).map_err(D::Error::custom)
}

// Represents common operations for a column chunk.
impl ColumnChunkMetaData {
    /// Returns a new [`ColumnChunkMetaData`]
    pub fn new(column_chunk: ColumnChunk, column_descr: ColumnDescriptor) -> Self {
        Self {
            column_chunk,
            column_descr,
        }
    }

    /// File where the column chunk is stored.
    ///
    /// If not set, assumed to belong to the same file as the metadata.
    /// This path is relative to the current file.
    pub fn file_path(&self) -> &Option<String> {
        &self.column_chunk.file_path
    }

    /// Byte offset in `file_path()`.
    pub fn file_offset(&self) -> i64 {
        self.column_chunk.file_offset
    }

    /// Returns this column's [`ColumnChunk`]
    pub fn column_chunk(&self) -> &ColumnChunk {
        &self.column_chunk
    }

    /// The column's [`ColumnMetaData`]
    pub fn metadata(&self) -> &ColumnMetaData {
        self.column_chunk.meta_data.as_ref().unwrap()
    }

    /// The [`ColumnDescriptor`] for this column. This descriptor contains the physical and logical type
    /// of the pages.
    pub fn descriptor(&self) -> &ColumnDescriptor {
        &self.column_descr
    }

    /// The [`PhysicalType`] of this column.
    pub fn physical_type(&self) -> PhysicalType {
        self.column_descr.descriptor.primitive_type.physical_type
    }

    /// Decodes the raw statistics into [`Statistics`].
    pub fn statistics(&self) -> Option<ParquetResult<Statistics>> {
        self.metadata().statistics.as_ref().map(|x| {
            Statistics::deserialize(x, self.column_descr.descriptor.primitive_type.clone())
        })
    }

    /// Total number of values in this column chunk. Note that this is not necessarily the number
    /// of rows. E.g. the (nested) array `[[1, 2], [3]]` has 2 rows and 3 values.
    pub fn num_values(&self) -> i64 {
        self.metadata().num_values
    }

    /// [`Compression`] for this column.
    pub fn compression(&self) -> Compression {
        self.metadata().codec.try_into().unwrap()
    }

    /// Returns the total compressed data size of this column chunk.
    pub fn compressed_size(&self) -> i64 {
        self.metadata().total_compressed_size
    }

    /// Returns the total uncompressed data size of this column chunk.
    pub fn uncompressed_size(&self) -> i64 {
        self.metadata().total_uncompressed_size
    }

    /// Returns the offset for the column data.
    pub fn data_page_offset(&self) -> i64 {
        self.metadata().data_page_offset
    }

    /// Returns `true` if this column chunk contains a index page, `false` otherwise.
    pub fn has_index_page(&self) -> bool {
        self.metadata().index_page_offset.is_some()
    }

    /// Returns the offset for the index page.
    pub fn index_page_offset(&self) -> Option<i64> {
        self.metadata().index_page_offset
    }

    /// Returns the offset for the dictionary page, if any.
    pub fn dictionary_page_offset(&self) -> Option<i64> {
        self.metadata().dictionary_page_offset
    }

    /// Returns the encoding for this column
    pub fn column_encoding(&self) -> &Vec<Encoding> {
        &self.metadata().encodings
    }

    /// Returns the offset and length in bytes of the column chunk within the file
    pub fn byte_range(&self) -> (u64, u64) {
        let start = if let Some(dict_page_offset) = self.dictionary_page_offset() {
            dict_page_offset as u64
        } else {
            self.data_page_offset() as u64
        };
        let length = self.compressed_size() as u64;
        // this has been validated in [`try_from_thrift`]
        (start, length)
    }

    /// Method to convert from Thrift.
    pub(crate) fn try_from_thrift(
        column_descr: ColumnDescriptor,
        column_chunk: ColumnChunk,
    ) -> ParquetResult<Self> {
        // validate metadata
        if let Some(meta) = &column_chunk.meta_data {
            let _: u64 = meta.total_compressed_size.try_into()?;

            if let Some(offset) = meta.dictionary_page_offset {
                let _: u64 = offset.try_into()?;
            }
            let _: u64 = meta.data_page_offset.try_into()?;

            let _: Compression = meta.codec.try_into()?;
        } else {
            return Err(ParquetError::oos("Column chunk requires metadata"));
        }

        Ok(Self {
            column_chunk,
            column_descr,
        })
    }

    /// Method to convert to Thrift.
    pub fn into_thrift(self) -> ColumnChunk {
        self.column_chunk
    }
}

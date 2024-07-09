//! APIs to read from Avro format to arrow.
use std::io::Read;

use avro_schema::file::FileMetadata;
use avro_schema::read::fallible_streaming_iterator::FallibleStreamingIterator;
use avro_schema::read::{block_iterator, BlockStreamingIterator};
use avro_schema::schema::Field as AvroField;

mod deserialize;
pub use deserialize::deserialize;
use polars_error::PolarsResult;

mod nested;
mod schema;
mod util;

pub use schema::infer_schema;

use crate::array::Array;
use crate::datatypes::Field;
use crate::record_batch::RecordBatchT;

/// Single threaded, blocking reader of Avro; [`Iterator`] of [`RecordBatchT`].
pub struct Reader<R: Read> {
    iter: BlockStreamingIterator<R>,
    avro_fields: Vec<AvroField>,
    fields: Vec<Field>,
    projection: Vec<bool>,
}

impl<R: Read> Reader<R> {
    /// Creates a new [`Reader`].
    pub fn new(
        reader: R,
        metadata: FileMetadata,
        fields: Vec<Field>,
        projection: Option<Vec<bool>>,
    ) -> Self {
        let projection = projection.unwrap_or_else(|| fields.iter().map(|_| true).collect());

        Self {
            iter: block_iterator(reader, metadata.compression, metadata.marker),
            avro_fields: metadata.record.fields,
            fields,
            projection,
        }
    }

    /// Deconstructs itself into its internal reader
    pub fn into_inner(self) -> R {
        self.iter.into_inner()
    }
}

impl<R: Read> Iterator for Reader<R> {
    type Item = PolarsResult<RecordBatchT<Box<dyn Array>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let fields = &self.fields[..];
        let avro_fields = &self.avro_fields;
        let projection = &self.projection;

        self.iter
            .next()
            .transpose()
            .map(|maybe_block| deserialize(maybe_block?, fields, avro_fields, projection))
    }
}

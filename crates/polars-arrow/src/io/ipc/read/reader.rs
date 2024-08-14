use std::io::{Read, Seek};

use polars_error::PolarsResult;
use polars_utils::aliases::PlHashMap;

use super::common::*;
use super::{read_batch, read_file_dictionaries, Dictionaries, FileMetadata};
use crate::array::Array;
use crate::datatypes::ArrowSchema;
use crate::record_batch::RecordBatchT;

/// An iterator of [`RecordBatchT`]s from an Arrow IPC file.
pub struct FileReader<R: Read + Seek> {
    reader: R,
    metadata: FileMetadata,
    // the dictionaries are going to be read
    dictionaries: Option<Dictionaries>,
    current_block: usize,
    projection: Option<(Vec<usize>, PlHashMap<usize, usize>, ArrowSchema)>,
    remaining: usize,
    data_scratch: Vec<u8>,
    message_scratch: Vec<u8>,
}

impl<R: Read + Seek> FileReader<R> {
    /// Creates a new [`FileReader`]. Use `projection` to only take certain columns.
    /// # Panic
    /// Panics iff the projection is not in increasing order (e.g. `[1, 0]` nor `[0, 1, 1]` are valid)
    pub fn new(
        reader: R,
        metadata: FileMetadata,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> Self {
        let projection = projection.map(|projection| {
            let (p, h, fields) = prepare_projection(&metadata.schema.fields, projection);
            let schema = ArrowSchema {
                fields,
                metadata: metadata.schema.metadata.clone(),
            };
            (p, h, schema)
        });
        Self {
            reader,
            metadata,
            dictionaries: Default::default(),
            projection,
            remaining: limit.unwrap_or(usize::MAX),
            current_block: 0,
            data_scratch: Default::default(),
            message_scratch: Default::default(),
        }
    }

    /// Return the schema of the file
    pub fn schema(&self) -> &ArrowSchema {
        self.projection
            .as_ref()
            .map(|x| &x.2)
            .unwrap_or(&self.metadata.schema)
    }

    /// Returns the [`FileMetadata`]
    pub fn metadata(&self) -> &FileMetadata {
        &self.metadata
    }

    /// Consumes this FileReader, returning the underlying reader
    pub fn into_inner(self) -> R {
        self.reader
    }

    /// Get the inner memory scratches so they can be reused in a new writer.
    /// This can be utilized to save memory allocations for performance reasons.
    pub fn get_scratches(&mut self) -> (Vec<u8>, Vec<u8>) {
        (
            std::mem::take(&mut self.data_scratch),
            std::mem::take(&mut self.message_scratch),
        )
    }

    /// Set the inner memory scratches so they can be reused in a new writer.
    /// This can be utilized to save memory allocations for performance reasons.
    pub fn set_scratches(&mut self, scratches: (Vec<u8>, Vec<u8>)) {
        (self.data_scratch, self.message_scratch) = scratches;
    }

    fn read_dictionaries(&mut self) -> PolarsResult<()> {
        if self.dictionaries.is_none() {
            self.dictionaries = Some(read_file_dictionaries(
                &mut self.reader,
                &self.metadata,
                &mut self.data_scratch,
            )?);
        };
        Ok(())
    }
}

impl<R: Read + Seek> Iterator for FileReader<R> {
    type Item = PolarsResult<RecordBatchT<Box<dyn Array>>>;

    fn next(&mut self) -> Option<Self::Item> {
        // get current block
        if self.current_block == self.metadata.blocks.len() {
            return None;
        }

        match self.read_dictionaries() {
            Ok(_) => {},
            Err(e) => return Some(Err(e)),
        };

        let block = self.current_block;
        self.current_block += 1;

        let chunk = read_batch(
            &mut self.reader,
            self.dictionaries.as_ref().unwrap(),
            &self.metadata,
            self.projection.as_ref().map(|x| x.0.as_ref()),
            Some(self.remaining),
            block,
            &mut self.message_scratch,
            &mut self.data_scratch,
        );
        self.remaining -= chunk.as_ref().map(|x| x.len()).unwrap_or_default();

        let chunk = if let Some((_, map, _)) = &self.projection {
            // re-order according to projection
            chunk.map(|chunk| apply_projection(chunk, map))
        } else {
            chunk
        };
        Some(chunk)
    }
}

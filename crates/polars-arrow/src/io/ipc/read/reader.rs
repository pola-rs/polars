use std::io::{Read, Seek};

use polars_error::PolarsResult;

use super::common::*;
use super::file::{get_message_from_block, get_record_batch};
use super::{Dictionaries, FileMetadata, read_batch, read_file_dictionaries};
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
    projection: Option<ProjectionInfo>,
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
        let projection =
            projection.map(|projection| prepare_projection(&metadata.schema, projection));
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

    /// Creates a new [`FileReader`]. Use `projection` to only take certain columns.
    /// # Panic
    /// Panics iff the projection is not in increasing order (e.g. `[1, 0]` nor `[0, 1, 1]` are valid)
    pub fn new_with_projection_info(
        reader: R,
        metadata: FileMetadata,
        projection: Option<ProjectionInfo>,
        limit: Option<usize>,
    ) -> Self {
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
            .map(|x| &x.schema)
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

    pub fn set_current_block(&mut self, idx: usize) {
        self.current_block = idx;
    }

    pub fn get_current_block(&self) -> usize {
        self.current_block
    }

    /// Get the inner memory scratches so they can be reused in a new writer.
    /// This can be utilized to save memory allocations for performance reasons.
    pub fn take_projection_info(&mut self) -> Option<ProjectionInfo> {
        std::mem::take(&mut self.projection)
    }

    /// Get the inner memory scratches so they can be reused in a new writer.
    /// This can be utilized to save memory allocations for performance reasons.
    pub fn take_scratches(&mut self) -> (Vec<u8>, Vec<u8>) {
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

    /// Skip over blocks until we have seen at most `offset` rows, returning how many rows we are
    /// still too see.  
    ///
    /// This will never go over the `offset`. Meaning that if the `offset < current_block.len()`,
    /// the block will not be skipped.
    pub fn skip_blocks_till_limit(&mut self, offset: u64) -> PolarsResult<u64> {
        let mut remaining_offset = offset;

        for (i, block) in self.metadata.blocks.iter().enumerate() {
            let message =
                get_message_from_block(&mut self.reader, block, &mut self.message_scratch)?;
            let record_batch = get_record_batch(message)?;

            let length = record_batch.length()?;
            let length = length as u64;

            if length > remaining_offset {
                self.current_block = i;
                return Ok(remaining_offset);
            }

            remaining_offset -= length;
        }

        self.current_block = self.metadata.blocks.len();
        Ok(remaining_offset)
    }

    pub fn next_record_batch(
        &mut self,
    ) -> Option<PolarsResult<arrow_format::ipc::RecordBatchRef<'_>>> {
        let block = self.metadata.blocks.get(self.current_block)?;
        self.current_block += 1;
        let message = get_message_from_block(&mut self.reader, block, &mut self.message_scratch);
        Some(message.and_then(|m| get_record_batch(m)))
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
            self.projection.as_ref().map(|x| x.columns.as_ref()),
            Some(self.remaining),
            block,
            &mut self.message_scratch,
            &mut self.data_scratch,
        );
        self.remaining -= chunk.as_ref().map(|x| x.len()).unwrap_or_default();

        let chunk = if let Some(ProjectionInfo { map, .. }) = &self.projection {
            // re-order according to projection
            chunk.map(|chunk| apply_projection(chunk, map))
        } else {
            chunk
        };
        Some(chunk)
    }
}

use std::io::{Read, Seek};

use ahash::AHashMap;
use polars_error::PolarsResult;

use super::common::*;
use super::{read_batch, read_file_dictionaries, Dictionaries, FileMetadata};
use crate::array::Array;
use crate::datatypes::ArrowSchema;
use crate::io::ipc::read::file;
use crate::record_batch::RecordBatch;

/// An iterator of [`RecordBatch`]s from an Arrow IPC file.
pub struct FileReader<R: Read + Seek> {
    reader: R,
    metadata: FileMetadata,
    // the dictionaries are going to be read
    dictionaries: Option<Dictionaries>,
    current_block: usize,
    projection: Option<(Vec<usize>, AHashMap<usize, usize>, ArrowSchema)>,
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
    type Item = PolarsResult<RecordBatch<Box<dyn Array>>>;

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

/// An iterator of raw bytes from an Arrow IPC file.
/// Returns the raw header and body of each IPC message without parsing it
/// Useful when efficiently sending data over the wire
#[cfg(feature = "io_flight")]
pub struct FlightFileReader<R: Read + Seek> {
    reader: R,
    has_read_footer: bool,
    record_batch_blocks: std::vec::IntoIter<arrow_format::ipc::Block>,
    dictionaries_blocks: Option<std::vec::IntoIter<arrow_format::ipc::Block>>,
    finished: bool,
}

#[cfg(feature = "io_flight")]
pub struct IPCRawMessage {
    pub data_header: Vec<u8>,
    pub data_body: Vec<u8>,
}

#[cfg(feature = "io_flight")]
impl<R: Read + Seek> FlightFileReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            has_read_footer: false,
            record_batch_blocks: vec![].into_iter(),
            dictionaries_blocks: None,
            finished: false,
        }
    }

    /// Check if the stream is finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Read in the footer data of the IPC file returning the schema
    /// We need to read in the footer data, because the dictionaries do not
    /// necessarily come before the batches which is required for streaming data
    pub fn read_footer(&mut self) -> PolarsResult<arrow_format::ipc::Schema> {
        let (_, footer_len) = file::read_footer_len(&mut self.reader)?;
        let footer_data = file::read_footer(&mut self.reader, footer_len)?;
        let footer = file::deserialize_footer_ref(&footer_data)?;

        self.record_batch_blocks =
            file::deserialize_record_batch_blocks_from_footer(footer)?.into_iter();
        self.dictionaries_blocks =
            file::deserialize_dictionary_blocks_from_footer(footer)?.map(|b| b.into_iter());

        // Get the schema from the footer
        let schema_ref = file::deserialize_schema_ref_from_footer(footer)?;
        let schema: arrow_format::ipc::Schema = schema_ref.try_into()?;
        Ok(schema)
    }

    /// Convert an IPC schema into an IPC Raw Message
    /// The schema comes from the footer and does not have the message format
    fn schema_to_raw_message(&self, schema: arrow_format::ipc::Schema) -> IPCRawMessage {
        // Turn the IPC schema into an encapsulated message
        let message = arrow_format::ipc::Message {
            version: arrow_format::ipc::MetadataVersion::V5,
            header: Some(arrow_format::ipc::MessageHeader::Schema(Box::new(schema))),
            body_length: 0,
            custom_metadata: None, // todo: allow writing custom metadata
        };
        let mut builder = arrow_format::ipc::planus::Builder::new();
        let header = builder.finish(&message, None).to_vec();
        IPCRawMessage {
            data_header: header,
            data_body: vec![],
        }
    }

    fn block_to_raw_message(
        &mut self,
        block: arrow_format::ipc::Block,
    ) -> PolarsResult<IPCRawMessage> {
        let mut header = vec![];
        let mut body = vec![];
        let message = read_ipc_message_from_block(&mut self.reader, &block, &mut header)?;

        let block_length: u64 = message
            .body_length()
            .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferBodyLength(err)))?
            .try_into()
            .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;
        self.reader
            .by_ref()
            .take(block_length)
            .read_to_end(&mut body)?;

        Ok(IPCRawMessage {
            data_header: header,
            data_body: body,
        })
    }

    /// Return the next message
    /// If the the reader is finished return None
    fn maybe_next(&mut self) -> PolarsResult<Option<IPCRawMessage>> {
        if self.finished {
            return Ok(None);
        }
        // Schema as the first message
        if !self.has_read_footer {
            let schema = self.read_footer()?;
            self.has_read_footer = true;
            return Ok(Some(self.schema_to_raw_message(schema)));
        }

        // Second send all the dictionaries
        if let Some(iter) = self.dictionaries_blocks.as_mut() {
            if let Some(block) = iter.next() {
                return self.block_to_raw_message(block).map(Some);
            } else {
                self.dictionaries_blocks = None;
            }
        }

        // Send the record batches
        if let Some(block) = self.record_batch_blocks.next() {
            self.block_to_raw_message(block).map(Some)
        } else {
            self.finished = true;
            Ok(None)
        }
    }
}

#[cfg(feature = "io_flight")]
impl<R: Read + Seek> Iterator for FlightFileReader<R> {
    type Item = PolarsResult<IPCRawMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        self.maybe_next().transpose()
    }
}

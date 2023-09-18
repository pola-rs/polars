use std::io::Write;

use arrow_format::ipc::planus::Builder;

use super::{
    super::IpcField,
    super::ARROW_MAGIC_V2,
    common::{DictionaryTracker, EncodedData, WriteOptions},
    common_sync::{write_continuation, write_message},
    default_ipc_fields, schema, schema_to_bytes,
};

use crate::array::Array;
use crate::chunk::Chunk;
use crate::datatypes::*;
use crate::error::{Error, Result};
use crate::io::ipc::write::common::encode_chunk_amortized;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    None,
    Started,
    Finished,
}

/// Arrow file writer
pub struct FileWriter<W: Write> {
    /// The object to write to
    pub(crate) writer: W,
    /// IPC write options
    pub(crate) options: WriteOptions,
    /// A reference to the schema, used in validating record batches
    pub(crate) schema: Schema,
    pub(crate) ipc_fields: Vec<IpcField>,
    /// The number of bytes between each block of bytes, as an offset for random access
    pub(crate) block_offsets: usize,
    /// Dictionary blocks that will be written as part of the IPC footer
    pub(crate) dictionary_blocks: Vec<arrow_format::ipc::Block>,
    /// Record blocks that will be written as part of the IPC footer
    pub(crate) record_blocks: Vec<arrow_format::ipc::Block>,
    /// Whether the writer footer has been written, and the writer is finished
    pub(crate) state: State,
    /// Keeps track of dictionaries that have been written
    pub(crate) dictionary_tracker: DictionaryTracker,
    /// Buffer/scratch that is reused between writes
    pub(crate) encoded_message: EncodedData,
}

impl<W: Write> FileWriter<W> {
    /// Creates a new [`FileWriter`] and writes the header to `writer`
    pub fn try_new(
        writer: W,
        schema: Schema,
        ipc_fields: Option<Vec<IpcField>>,
        options: WriteOptions,
    ) -> Result<Self> {
        let mut slf = Self::new(writer, schema, ipc_fields, options);
        slf.start()?;

        Ok(slf)
    }

    /// Creates a new [`FileWriter`].
    pub fn new(
        writer: W,
        schema: Schema,
        ipc_fields: Option<Vec<IpcField>>,
        options: WriteOptions,
    ) -> Self {
        let ipc_fields = if let Some(ipc_fields) = ipc_fields {
            ipc_fields
        } else {
            default_ipc_fields(&schema.fields)
        };

        Self {
            writer,
            options,
            schema,
            ipc_fields,
            block_offsets: 0,
            dictionary_blocks: vec![],
            record_blocks: vec![],
            state: State::None,
            dictionary_tracker: DictionaryTracker {
                dictionaries: Default::default(),
                cannot_replace: true,
            },
            encoded_message: Default::default(),
        }
    }

    /// Consumes itself into the inner writer
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Get the inner memory scratches so they can be reused in a new writer.
    /// This can be utilized to save memory allocations for performance reasons.
    pub fn get_scratches(&mut self) -> EncodedData {
        std::mem::take(&mut self.encoded_message)
    }
    /// Set the inner memory scratches so they can be reused in a new writer.
    /// This can be utilized to save memory allocations for performance reasons.
    pub fn set_scratches(&mut self, scratches: EncodedData) {
        self.encoded_message = scratches;
    }

    /// Writes the header and first (schema) message to the file.
    /// # Errors
    /// Errors if the file has been started or has finished.
    pub fn start(&mut self) -> Result<()> {
        if self.state != State::None {
            return Err(Error::oos("The IPC file can only be started once"));
        }
        // write magic to header
        self.writer.write_all(&ARROW_MAGIC_V2[..])?;
        // create an 8-byte boundary after the header
        self.writer.write_all(&[0, 0])?;
        // write the schema, set the written bytes to the schema

        let encoded_message = EncodedData {
            ipc_message: schema_to_bytes(&self.schema, &self.ipc_fields),
            arrow_data: vec![],
        };

        let (meta, data) = write_message(&mut self.writer, &encoded_message)?;
        self.block_offsets += meta + data + 8; // 8 <=> arrow magic + 2 bytes for alignment
        self.state = State::Started;
        Ok(())
    }

    /// Writes [`Chunk`] to the file
    pub fn write(
        &mut self,
        chunk: &Chunk<Box<dyn Array>>,
        ipc_fields: Option<&[IpcField]>,
    ) -> Result<()> {
        if self.state != State::Started {
            return Err(Error::oos(
                "The IPC file must be started before it can be written to. Call `start` before `write`",
            ));
        }

        let ipc_fields = if let Some(ipc_fields) = ipc_fields {
            ipc_fields
        } else {
            self.ipc_fields.as_ref()
        };
        let encoded_dictionaries = encode_chunk_amortized(
            chunk,
            ipc_fields,
            &mut self.dictionary_tracker,
            &self.options,
            &mut self.encoded_message,
        )?;

        // add all dictionaries
        for encoded_dictionary in encoded_dictionaries {
            let (meta, data) = write_message(&mut self.writer, &encoded_dictionary)?;

            let block = arrow_format::ipc::Block {
                offset: self.block_offsets as i64,
                meta_data_length: meta as i32,
                body_length: data as i64,
            };
            self.dictionary_blocks.push(block);
            self.block_offsets += meta + data;
        }

        let (meta, data) = write_message(&mut self.writer, &self.encoded_message)?;
        // add a record block for the footer
        let block = arrow_format::ipc::Block {
            offset: self.block_offsets as i64,
            meta_data_length: meta as i32, // TODO: is this still applicable?
            body_length: data as i64,
        };
        self.record_blocks.push(block);
        self.block_offsets += meta + data;
        Ok(())
    }

    /// Write footer and closing tag, then mark the writer as done
    pub fn finish(&mut self) -> Result<()> {
        if self.state != State::Started {
            return Err(Error::oos(
                "The IPC file must be started before it can be finished. Call `start` before `finish`",
            ));
        }

        // write EOS
        write_continuation(&mut self.writer, 0)?;

        let schema = schema::serialize_schema(&self.schema, &self.ipc_fields);

        let root = arrow_format::ipc::Footer {
            version: arrow_format::ipc::MetadataVersion::V5,
            schema: Some(Box::new(schema)),
            dictionaries: Some(std::mem::take(&mut self.dictionary_blocks)),
            record_batches: Some(std::mem::take(&mut self.record_blocks)),
            custom_metadata: None,
        };
        let mut builder = Builder::new();
        let footer_data = builder.finish(&root, None);
        self.writer.write_all(footer_data)?;
        self.writer
            .write_all(&(footer_data.len() as i32).to_le_bytes())?;
        self.writer.write_all(&ARROW_MAGIC_V2)?;
        self.writer.flush()?;
        self.state = State::Finished;

        Ok(())
    }
}

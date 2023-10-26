use std::convert::TryInto;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use ahash::AHashMap;
use arrow_format::ipc::planus::ReadAsRoot;
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::super::{ARROW_MAGIC_V1, ARROW_MAGIC_V2, CONTINUATION_MARKER};
use super::common::*;
use super::schema::fb_to_schema;
use super::{Dictionaries, OutOfSpecKind};
use crate::array::Array;
use crate::chunk::Chunk;
use crate::datatypes::ArrowSchemaRef;
use crate::io::ipc::IpcSchema;

/// Metadata of an Arrow IPC file, written in the footer of the file.
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// The schema that is read from the file footer
    pub schema: ArrowSchemaRef,

    /// The files' [`IpcSchema`]
    pub ipc_schema: IpcSchema,

    /// The blocks in the file
    ///
    /// A block indicates the regions in the file to read to get data
    pub blocks: Vec<arrow_format::ipc::Block>,

    /// Dictionaries associated to each dict_id
    pub(crate) dictionaries: Option<Vec<arrow_format::ipc::Block>>,

    /// The total size of the file in bytes
    pub size: u64,
}

fn read_dictionary_message<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    data: &mut Vec<u8>,
) -> PolarsResult<()> {
    let mut message_size: [u8; 4] = [0; 4];
    reader.seek(SeekFrom::Start(offset))?;
    reader.read_exact(&mut message_size)?;
    if message_size == CONTINUATION_MARKER {
        reader.read_exact(&mut message_size)?;
    };
    let message_length = i32::from_le_bytes(message_size);

    let message_length: usize = message_length
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    data.clear();
    data.try_reserve(message_length)?;
    reader
        .by_ref()
        .take(message_length as u64)
        .read_to_end(data)?;

    Ok(())
}

pub(crate) fn get_dictionary_batch<'a>(
    message: &'a arrow_format::ipc::MessageRef,
) -> PolarsResult<arrow_format::ipc::DictionaryBatchRef<'a>> {
    let header = message
        .header()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferHeader(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageHeader))?;
    match header {
        arrow_format::ipc::MessageHeaderRef::DictionaryBatch(batch) => Ok(batch),
        _ => polars_bail!(oos = OutOfSpecKind::UnexpectedMessageType),
    }
}

fn read_dictionary_block<R: Read + Seek>(
    reader: &mut R,
    metadata: &FileMetadata,
    block: &arrow_format::ipc::Block,
    dictionaries: &mut Dictionaries,
    message_scratch: &mut Vec<u8>,
    dictionary_scratch: &mut Vec<u8>,
) -> PolarsResult<()> {
    let offset: u64 = block
        .offset
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;
    let length: u64 = block
        .meta_data_length
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;
    read_dictionary_message(reader, offset, message_scratch)?;

    let message = arrow_format::ipc::MessageRef::read_as_root(message_scratch.as_ref())
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))?;

    let batch = get_dictionary_batch(&message)?;

    read_dictionary(
        batch,
        &metadata.schema.fields,
        &metadata.ipc_schema,
        dictionaries,
        reader,
        offset + length,
        metadata.size,
        dictionary_scratch,
    )
}

/// Reads all file's dictionaries, if any
/// This function is IO-bounded
pub fn read_file_dictionaries<R: Read + Seek>(
    reader: &mut R,
    metadata: &FileMetadata,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Dictionaries> {
    let mut dictionaries = Default::default();

    let blocks = if let Some(blocks) = &metadata.dictionaries {
        blocks
    } else {
        return Ok(AHashMap::new());
    };
    // use a temporary smaller scratch for the messages
    let mut message_scratch = Default::default();

    for block in blocks {
        read_dictionary_block(
            reader,
            metadata,
            block,
            &mut dictionaries,
            &mut message_scratch,
            scratch,
        )?;
    }
    Ok(dictionaries)
}

/// Reads the footer's length and magic number in footer
fn read_footer_len<R: Read + Seek>(reader: &mut R) -> PolarsResult<(u64, usize)> {
    // read footer length and magic number in footer
    let end = reader.seek(SeekFrom::End(-10))? + 10;

    let mut footer: [u8; 10] = [0; 10];

    reader.read_exact(&mut footer)?;
    let footer_len = i32::from_le_bytes(footer[..4].try_into().unwrap());

    if footer[4..] != ARROW_MAGIC_V2 {
        return Err(polars_err!(oos = OutOfSpecKind::InvalidFooter));
    }
    let footer_len = footer_len
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    Ok((end, footer_len))
}

pub(super) fn deserialize_footer(footer_data: &[u8], size: u64) -> PolarsResult<FileMetadata> {
    let footer = arrow_format::ipc::FooterRef::read_as_root(footer_data)
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferFooter(err)))?;

    let blocks = footer
        .record_batches()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferRecordBatches(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingRecordBatches))?;

    let blocks = blocks
        .iter()
        .map(|block| {
            block.try_into().map_err(|err| {
                polars_err!(oos = OutOfSpecKind::InvalidFlatbufferRecordBatches(err))
            })
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let ipc_schema = footer
        .schema()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferSchema(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingSchema))?;
    let (schema, ipc_schema) = fb_to_schema(ipc_schema)?;

    let dictionaries = footer
        .dictionaries()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferDictionaries(err)))?
        .map(|dictionaries| {
            dictionaries
                .into_iter()
                .map(|block| {
                    block.try_into().map_err(|err| {
                        polars_err!(oos = OutOfSpecKind::InvalidFlatbufferRecordBatches(err))
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()
        })
        .transpose()?;

    Ok(FileMetadata {
        schema: Arc::new(schema),
        ipc_schema,
        blocks,
        dictionaries,
        size,
    })
}

/// Read the Arrow IPC file's metadata
pub fn read_file_metadata<R: Read + Seek>(reader: &mut R) -> PolarsResult<FileMetadata> {
    // check if header contain the correct magic bytes
    let mut magic_buffer: [u8; 6] = [0; 6];
    let start = reader.stream_position()?;
    reader.read_exact(&mut magic_buffer)?;
    if magic_buffer != ARROW_MAGIC_V2 {
        if magic_buffer[..4] == ARROW_MAGIC_V1 {
            polars_bail!(ComputeError: "feather v1 not supported");
        }
        polars_bail!(oos = OutOfSpecKind::InvalidHeader);
    }

    let (end, footer_len) = read_footer_len(reader)?;

    // read footer
    reader.seek(SeekFrom::End(-10 - footer_len as i64))?;

    let mut serialized_footer = vec![];
    serialized_footer.try_reserve(footer_len)?;
    reader
        .by_ref()
        .take(footer_len as u64)
        .read_to_end(&mut serialized_footer)?;

    deserialize_footer(&serialized_footer, end - start)
}

pub(crate) fn get_record_batch(
    message: arrow_format::ipc::MessageRef,
) -> PolarsResult<arrow_format::ipc::RecordBatchRef> {
    let header = message
        .header()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferHeader(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageHeader))?;
    match header {
        arrow_format::ipc::MessageHeaderRef::RecordBatch(batch) => Ok(batch),
        _ => polars_bail!(oos = OutOfSpecKind::UnexpectedMessageType),
    }
}

/// Reads the record batch at position `index` from the reader.
///
/// This function is useful for random access to the file. For example, if
/// you have indexed the file somewhere else, this allows pruning
/// certain parts of the file.
/// # Panics
/// This function panics iff `index >= metadata.blocks.len()`
#[allow(clippy::too_many_arguments)]
pub fn read_batch<R: Read + Seek>(
    reader: &mut R,
    dictionaries: &Dictionaries,
    metadata: &FileMetadata,
    projection: Option<&[usize]>,
    limit: Option<usize>,
    index: usize,
    message_scratch: &mut Vec<u8>,
    data_scratch: &mut Vec<u8>,
) -> PolarsResult<Chunk<Box<dyn Array>>> {
    let block = metadata.blocks[index];

    let offset: u64 = block
        .offset
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    let length: u64 = block
        .meta_data_length
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    // read length
    reader.seek(SeekFrom::Start(offset))?;
    let mut meta_buf = [0; 4];
    reader.read_exact(&mut meta_buf)?;
    if meta_buf == CONTINUATION_MARKER {
        // continuation marker encountered, read message next
        reader.read_exact(&mut meta_buf)?;
    }
    let meta_len = i32::from_le_bytes(meta_buf)
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;

    message_scratch.clear();
    message_scratch.try_reserve(meta_len)?;
    reader
        .by_ref()
        .take(meta_len as u64)
        .read_to_end(message_scratch)?;

    let message = arrow_format::ipc::MessageRef::read_as_root(message_scratch.as_ref())
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))?;

    let batch = get_record_batch(message)?;

    read_record_batch(
        batch,
        &metadata.schema.fields,
        &metadata.ipc_schema,
        projection,
        limit,
        dictionaries,
        message
            .version()
            .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferVersion(err)))?,
        reader,
        offset + length,
        metadata.size,
        data_scratch,
    )
}

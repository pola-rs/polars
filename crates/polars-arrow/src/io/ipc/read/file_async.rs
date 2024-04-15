//! Async reader for Arrow IPC files
use std::io::SeekFrom;

use ahash::AHashMap;
use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::{Block, MessageHeaderRef};
use futures::stream::BoxStream;
use futures::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt, Stream, StreamExt};
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::common::{apply_projection, prepare_projection, read_dictionary, read_record_batch};
use super::file::{deserialize_footer, get_record_batch};
use super::{Dictionaries, FileMetadata, OutOfSpecKind};
use crate::array::*;
use crate::datatypes::{ArrowSchema, Field};
use crate::io::ipc::read::file;
use crate::io::ipc::write::EncodedData;
use crate::io::ipc::{IpcSchema, ARROW_MAGIC_V2, CONTINUATION_MARKER};
use crate::record_batch::RecordBatch;

/// Async reader for Arrow IPC files
pub struct FileStream<'a> {
    stream: BoxStream<'a, PolarsResult<RecordBatch<Box<dyn Array>>>>,
    schema: Option<ArrowSchema>,
    metadata: FileMetadata,
}

impl<'a> FileStream<'a> {
    /// Create a new IPC file reader.
    ///
    /// # Examples
    /// See [`FileSink`](crate::io::ipc::write::file_async::FileSink).
    pub fn new<R>(
        reader: R,
        metadata: FileMetadata,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> Self
    where
        R: AsyncRead + AsyncSeek + Unpin + Send + 'a,
    {
        let (projection, schema) = if let Some(projection) = projection {
            let (p, h, fields) = prepare_projection(&metadata.schema.fields, projection);
            let schema = ArrowSchema {
                fields,
                metadata: metadata.schema.metadata.clone(),
            };
            (Some((p, h)), Some(schema))
        } else {
            (None, None)
        };

        let stream = Self::stream(reader, None, metadata.clone(), projection, limit);
        Self {
            stream,
            metadata,
            schema,
        }
    }

    /// Get the metadata from the IPC file.
    pub fn metadata(&self) -> &FileMetadata {
        &self.metadata
    }

    /// Get the projected schema from the IPC file.
    pub fn schema(&self) -> &ArrowSchema {
        self.schema.as_ref().unwrap_or(&self.metadata.schema)
    }

    fn stream<R>(
        mut reader: R,
        mut dictionaries: Option<Dictionaries>,
        metadata: FileMetadata,
        projection: Option<(Vec<usize>, AHashMap<usize, usize>)>,
        limit: Option<usize>,
    ) -> BoxStream<'a, PolarsResult<RecordBatch<Box<dyn Array>>>>
    where
        R: AsyncRead + AsyncSeek + Unpin + Send + 'a,
    {
        async_stream::try_stream! {
            // read dictionaries
            cached_read_dictionaries(&mut reader, &metadata, &mut dictionaries).await?;

            let mut meta_buffer = Default::default();
            let mut block_buffer = Default::default();
            let mut scratch = Default::default();
            let mut remaining = limit.unwrap_or(usize::MAX);
            for block in 0..metadata.blocks.len() {
                let chunk = read_batch(
                    &mut reader,
                    dictionaries.as_mut().unwrap(),
                    &metadata,
                    projection.as_ref().map(|x| x.0.as_ref()),
                    Some(remaining),
                    block,
                    &mut meta_buffer,
                    &mut block_buffer,
                    &mut scratch
                ).await?;
                remaining -= chunk.len();

                let chunk = if let Some((_, map)) = &projection {
                    // re-order according to projection
                    apply_projection(chunk, map)
                } else {
                    chunk
                };

                yield chunk;
            }
        }
        .boxed()
    }
}

impl<'a> Stream for FileStream<'a> {
    type Item = PolarsResult<RecordBatch<Box<dyn Array>>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.get_mut().stream.poll_next_unpin(cx)
    }
}

/// Reads the footer's length and magic number in footer
async fn read_footer_len<R: AsyncRead + AsyncSeek + Unpin>(reader: &mut R) -> PolarsResult<usize> {
    // read footer length and magic number in footer
    reader.seek(SeekFrom::End(-10)).await?;
    let mut footer: [u8; 10] = [0; 10];

    reader.read_exact(&mut footer).await?;
    let footer_len = i32::from_le_bytes(footer[..4].try_into().unwrap());

    if footer[4..] != ARROW_MAGIC_V2 {
        polars_bail!(oos = OutOfSpecKind::InvalidFooter)
    }
    footer_len
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))
}

/// Read in the footer of the IPC file
pub async fn read_footer<R>(reader: &mut R, footer_len: usize) -> PolarsResult<Vec<u8>>
where
    R: AsyncRead + AsyncSeek + Unpin,
{
    let mut footer = vec![];
    reader.seek(SeekFrom::End(-10 - footer_len as i64)).await?;

    footer.try_reserve(footer_len)?;
    reader
        .take(footer_len as u64)
        .read_to_end(&mut footer)
        .await?;
    Ok(footer)
}

/// Read the metadata from an IPC file.
pub async fn read_file_metadata_async<R>(reader: &mut R) -> PolarsResult<FileMetadata>
where
    R: AsyncRead + AsyncSeek + Unpin,
{
    let footer_size = read_footer_len(reader).await?;
    let footer = read_footer(reader, footer_size).await?;
    deserialize_footer(&footer, u64::MAX)
}

async fn read_ipc_message_from_block<'a, R: AsyncRead + AsyncSeek + Unpin>(
    reader: &mut R,
    block: arrow_format::ipc::Block,
    scratch: &'a mut Vec<u8>,
) -> PolarsResult<arrow_format::ipc::MessageRef<'a>> {
    let offset: u64 = block
        .offset
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
    reader.seek(SeekFrom::Start(offset)).await?;
    read_ipc_message(reader, scratch).await
}

#[allow(clippy::needless_lifetimes)]
async fn read_ipc_message<'a, R>(
    mut reader: R,
    data: &'a mut Vec<u8>,
) -> PolarsResult<arrow_format::ipc::MessageRef<'a>>
where
    R: AsyncRead + AsyncSeek + Unpin,
{
    let mut message_size = [0; 4];
    reader.read_exact(&mut message_size).await?;
    if message_size == CONTINUATION_MARKER {
        reader.read_exact(&mut message_size).await?;
    }
    let footer_size = i32::from_le_bytes(message_size);

    let footer_size: usize = footer_size
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    data.clear();
    data.try_reserve(footer_size)?;
    (&mut reader)
        .take(footer_size as u64)
        .read_to_end(data)
        .await?;

    arrow_format::ipc::MessageRef::read_as_root(data)
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))
}

#[allow(clippy::too_many_arguments)]
async fn read_batch<R>(
    mut reader: R,
    dictionaries: &mut Dictionaries,
    metadata: &FileMetadata,
    projection: Option<&[usize]>,
    limit: Option<usize>,
    block: usize,
    meta_buffer: &mut Vec<u8>,
    block_buffer: &mut Vec<u8>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<RecordBatch<Box<dyn Array>>>
where
    R: AsyncRead + AsyncSeek + Unpin,
{
    let block = metadata.blocks[block];
    let message = read_ipc_message_from_block(&mut reader, block, meta_buffer).await?;

    let batch = get_record_batch(message)?;

    let block_length: usize = message
        .body_length()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferBodyLength(err)))?
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;

    block_buffer.clear();
    block_buffer.try_reserve(block_length)?;
    reader
        .take(block_length as u64)
        .read_to_end(block_buffer)
        .await?;

    let mut cursor = std::io::Cursor::new(&block_buffer);

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
        &mut cursor,
        0,
        metadata.size,
        scratch,
    )
}

async fn read_dictionaries<R>(
    mut reader: R,
    fields: &[Field],
    ipc_schema: &IpcSchema,
    blocks: &[Block],
    scratch: &mut Vec<u8>,
) -> PolarsResult<Dictionaries>
where
    R: AsyncRead + AsyncSeek + Unpin,
{
    let mut dictionaries = Default::default();
    let mut data: Vec<u8> = vec![];
    let mut buffer: Vec<u8> = vec![];

    for block in blocks {
        let offset: u64 = block
            .offset
            .try_into()
            .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

        let length: usize = block
            .body_length
            .try_into()
            .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

        read_dictionary_message(&mut reader, offset, &mut data).await?;

        let message = arrow_format::ipc::MessageRef::read_as_root(data.as_ref())
            .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))?;

        let header = message
            .header()
            .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferHeader(err)))?
            .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageHeader))?;

        match header {
            MessageHeaderRef::DictionaryBatch(batch) => {
                buffer.clear();
                buffer.try_reserve(length)?;
                (&mut reader)
                    .take(length as u64)
                    .read_to_end(&mut buffer)
                    .await?;
                let mut cursor = std::io::Cursor::new(&buffer);
                read_dictionary(
                    batch,
                    fields,
                    ipc_schema,
                    &mut dictionaries,
                    &mut cursor,
                    0,
                    u64::MAX,
                    scratch,
                )?;
            },
            _ => polars_bail!(oos = OutOfSpecKind::UnexpectedMessageType),
        }
    }
    Ok(dictionaries)
}

async fn read_dictionary_message<R>(
    mut reader: R,
    offset: u64,
    data: &mut Vec<u8>,
) -> PolarsResult<()>
where
    R: AsyncRead + AsyncSeek + Unpin,
{
    let mut message_size = [0; 4];
    reader.seek(SeekFrom::Start(offset)).await?;
    reader.read_exact(&mut message_size).await?;
    if message_size == CONTINUATION_MARKER {
        reader.read_exact(&mut message_size).await?;
    }
    let footer_size = i32::from_le_bytes(message_size);

    let footer_size: usize = footer_size
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    data.clear();
    data.try_reserve(footer_size)?;
    (&mut reader)
        .take(footer_size as u64)
        .read_to_end(data)
        .await?;

    Ok(())
}

async fn cached_read_dictionaries<R: AsyncRead + AsyncSeek + Unpin>(
    reader: &mut R,
    metadata: &FileMetadata,
    dictionaries: &mut Option<Dictionaries>,
) -> PolarsResult<()> {
    match (&dictionaries, metadata.dictionaries.as_deref()) {
        (None, Some(blocks)) => {
            let new_dictionaries = read_dictionaries(
                reader,
                &metadata.schema.fields,
                &metadata.ipc_schema,
                blocks,
                &mut Default::default(),
            )
            .await?;
            *dictionaries = Some(new_dictionaries);
        },
        (None, None) => {
            *dictionaries = Some(Default::default());
        },
        _ => {},
    };
    Ok(())
}

#[cfg(feature = "io_flight")]
pub struct FlightAsyncRawReader {}

/// Read in the footer data of the IPC file returning the schema
/// We need to read in the footer data, because the dictionaries do not
/// necessarily come before the batches which is required for streaming data over flight
#[cfg(feature = "io_flight")]
pub async fn read_footer_raw<R>(
    reader: &mut R,
) -> PolarsResult<(
    arrow_format::ipc::Schema,
    std::vec::IntoIter<Block>,
    Option<std::vec::IntoIter<Block>>,
)>
where
    R: AsyncRead + AsyncSeek + Unpin + Send,
{
    let footer_len = read_footer_len(reader).await?;
    let footer_data = read_footer(reader, footer_len).await?;
    let footer = file::deserialize_footer_ref(&footer_data)?;

    // TODO now we do this at once, we could keep the iterator
    let record_batch_blocks =
        file::deserialize_record_batch_blocks_from_footer(footer)?.into_iter();
    let dictionaries_blocks =
        file::deserialize_dictionary_blocks_from_footer(footer)?.map(|b| b.into_iter());

    // Get the schema from the footer
    let schema_ref = file::deserialize_schema_ref_from_footer(footer)?;
    let schema: arrow_format::ipc::Schema = schema_ref.try_into()?;
    Ok((schema, record_batch_blocks, dictionaries_blocks))
}

/// Convert an IPC schema into an IPC Raw Message
/// The schema comes from the footer and does not have the message format
#[cfg(feature = "io_flight")]
fn schema_to_raw_message(schema: arrow_format::ipc::Schema) -> EncodedData {
    // Turn the IPC schema into an encapsulated message
    let message = arrow_format::ipc::Message {
        version: arrow_format::ipc::MetadataVersion::V5,
        header: Some(arrow_format::ipc::MessageHeader::Schema(Box::new(schema))),
        body_length: 0,
        custom_metadata: None, // todo: allow writing custom metadata
    };
    let mut builder = arrow_format::ipc::planus::Builder::new();
    let header = builder.finish(&message, None).to_vec();
    EncodedData {
        ipc_message: header,
        arrow_data: vec![],
    }
}

#[cfg(feature = "io_flight")]
async fn block_to_raw_message<'a, R>(reader: &mut R, block: Block) -> PolarsResult<EncodedData>
where
    R: AsyncRead + AsyncSeek + Unpin + Send + 'a,
{
    let mut header = vec![];
    let mut body = vec![];
    let message = read_ipc_message_from_block(reader, block, &mut header).await?;

    let block_length: u64 = message
        .body_length()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferBodyLength(err)))?
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;
    reader.take(block_length).read_to_end(&mut body).await?;

    Ok(EncodedData {
        ipc_message: header,
        arrow_data: body,
    })
}

#[cfg(feature = "io_flight")]
impl FlightAsyncRawReader {
    /// Return a stream of messages
    pub fn stream<R>(mut reader: R) -> impl Stream<Item = PolarsResult<EncodedData>>
    where
        R: AsyncRead + AsyncSeek + Unpin + Send,
    {
        async_stream::try_stream! {
            let (schema, record_batch_blocks, dictionary_blocks) = read_footer_raw(&mut reader).await?;

            // Schema as the first message
            yield schema_to_raw_message(schema);

            // Second send all the dictionaries
            if let Some(iter) = dictionary_blocks {
                for block in iter{
                    yield block_to_raw_message(&mut reader, block).await?;
                }
            }

            // Send the record batches
            for block in record_batch_blocks{
                yield block_to_raw_message(&mut reader, block).await?
            }
        }
    }
}

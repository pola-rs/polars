use std::io::SeekFrom;
use std::sync::Arc;

use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::MessageHeaderRef;
use futures::{Stream, StreamExt};
use polars_error::{polars_bail, polars_err, PolarsResult};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt};

use crate::io::ipc::read::file::{
    decode_footer_len, deserialize_schema_ref_from_footer, iter_dictionary_blocks_from_footer,
    iter_recordbatch_blocks_from_footer,
};
use crate::io::ipc::read::schema::deserialize_stream_metadata;
use crate::io::ipc::read::{Dictionaries, OutOfSpecKind, StreamMetadata};
use crate::io::ipc::write::common::EncodedData;
use crate::mmap::{mmap_dictionary_from_batch, mmap_record};
use crate::record_batch::RecordBatch;

async fn read_ipc_message_from_block<'a, R: AsyncRead + AsyncSeek + Unpin>(
    reader: &mut R,
    block: &arrow_format::ipc::Block,
    scratch: &'a mut Vec<u8>,
) -> PolarsResult<arrow_format::ipc::MessageRef<'a>> {
    let offset: u64 = block
        .offset
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
    reader.seek(SeekFrom::Start(offset)).await?;
    read_ipc_message(reader, scratch).await
}

/// Read an encapsulated IPC Message from the reader
async fn read_ipc_message<'a, R: AsyncRead + Unpin>(
    reader: &mut R,
    scratch: &'a mut Vec<u8>,
) -> PolarsResult<arrow_format::ipc::MessageRef<'a>> {
    let mut message_size: [u8; 4] = [0; 4];

    reader.read_exact(&mut message_size).await?;
    if message_size == crate::io::ipc::CONTINUATION_MARKER {
        reader.read_exact(&mut message_size).await?;
    };
    let message_length = i32::from_le_bytes(message_size);

    let message_length: usize = message_length
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    scratch.clear();
    scratch.try_reserve(message_length)?;
    reader
        .take(message_length as u64)
        .read_to_end(scratch)
        .await?;

    arrow_format::ipc::MessageRef::read_as_root(scratch)
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))
}

async fn read_footer_len<R: AsyncRead + AsyncSeek + Unpin>(
    reader: &mut R,
) -> PolarsResult<(u64, usize)> {
    // read footer length and magic number in footer
    let end = reader.seek(SeekFrom::End(-10)).await? + 10;

    let mut footer: [u8; 10] = [0; 10];
    reader.read_exact(&mut footer).await?;

    decode_footer_len(footer, end)
}

async fn read_footer<R: AsyncRead + AsyncSeek + Unpin>(
    reader: &mut R,
    footer_len: usize,
) -> PolarsResult<Vec<u8>> {
    // read footer
    reader.seek(SeekFrom::End(-10 - footer_len as i64)).await?;

    let mut serialized_footer = vec![];
    serialized_footer.try_reserve(footer_len)?;

    reader
        .take(footer_len as u64)
        .read_to_end(&mut serialized_footer)
        .await?;
    Ok(serialized_footer)
}

fn schema_to_raw_message(schema: arrow_format::ipc::SchemaRef) -> EncodedData {
    // Turn the IPC schema into an encapsulated message
    let message = arrow_format::ipc::Message {
        version: arrow_format::ipc::MetadataVersion::V5,
        // Assumed the conversion is infallible.
        header: Some(arrow_format::ipc::MessageHeader::Schema(Box::new(
            schema.try_into().unwrap(),
        ))),
        body_length: 0,
        custom_metadata: None, // todo: allow writing custom metadata
    };
    let mut builder = arrow_format::ipc::planus::Builder::new();
    let header = builder.finish(&message, None).to_vec();

    // Use `EncodedData` directly instead of `FlightData`. In FlightData we would only use
    // `data_header` and `data_body`.
    EncodedData {
        ipc_message: header,
        arrow_data: vec![],
    }
}

async fn block_to_raw_message<'a, R>(
    reader: &mut R,
    block: &arrow_format::ipc::Block,
) -> PolarsResult<EncodedData>
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

// TODO! optimize this by passing an `EncodedData` to the functions and reuse the same allocation
pub async fn into_flight_stream<R: AsyncRead + AsyncSeek + Unpin + Send>(
    reader: &mut R,
) -> PolarsResult<impl Stream<Item = PolarsResult<EncodedData>> + '_> {
    Ok(async_stream::try_stream! {
        let (_end, len) = read_footer_len(reader).await?;
        let footer_data = read_footer(reader, len).await?;
        let footer = arrow_format::ipc::FooterRef::read_as_root(&footer_data)
            .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferFooter(err)))?;
        let data_blocks = iter_recordbatch_blocks_from_footer(footer)?;
        let dict_blocks = iter_dictionary_blocks_from_footer(footer)?;

        let schema_ref = deserialize_schema_ref_from_footer(footer)?;
        let schema = schema_to_raw_message(schema_ref);

        yield schema;

        if let Some(dict_blocks_iter) = dict_blocks {
            for d in dict_blocks_iter {
                yield block_to_raw_message(reader, &d?).await?;
            }
        };

        for d in data_blocks {
            yield block_to_raw_message(reader, &d?).await?;
        }
    })
}

pub struct FlightstreamConsumer<S: Stream<Item = PolarsResult<EncodedData>> + Unpin> {
    dictionaries: Dictionaries,
    md: StreamMetadata,
    stream: S,
}

impl<S: Stream<Item = PolarsResult<EncodedData>> + Unpin> FlightstreamConsumer<S> {
    pub async fn new(mut stream: S) -> PolarsResult<Self> {
        let Some(first) = stream.next().await else {
            polars_bail!(ComputeError: "expected the schema")
        };
        let first = first?;

        let md = deserialize_stream_metadata(&first.ipc_message)?;
        Ok(FlightstreamConsumer {
            dictionaries: Default::default(),
            md,
            stream,
        })
    }

    pub async fn next_batch(&mut self) -> PolarsResult<Option<RecordBatch>> {
        while let Some(msg) = self.stream.next().await {
            let msg = msg?;

            // Parse the header
            let message = arrow_format::ipc::MessageRef::read_as_root(&msg.ipc_message)
                .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))?;

            let header = message
                .header()
                .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferHeader(err)))?
                .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageHeader))?;

            // Needed to memory map.
            let arrow_data = Arc::new(msg.arrow_data);

            // Either append to the dictionaries and return None or return Some(ArrowChunk)
            match header {
                MessageHeaderRef::Schema(_) => {
                    polars_bail!(ComputeError: "Unexpected schema message while parsing Stream");
                },
                // Add to dictionary state and continue iteration
                MessageHeaderRef::DictionaryBatch(batch) => unsafe {
                    mmap_dictionary_from_batch(
                        &self.md.schema,
                        &self.md.ipc_schema.fields,
                        &arrow_data,
                        batch,
                        &mut self.dictionaries,
                        0,
                    )?
                },
                // Return Batch
                MessageHeaderRef::RecordBatch(batch) => {
                    return unsafe {
                        mmap_record(
                            &self.md.schema,
                            &self.md.ipc_schema.fields,
                            arrow_data.clone(),
                            batch,
                            0,
                            &self.dictionaries,
                        )
                        .map(Some)
                    }
                },
                _ => unimplemented!(),
            }
        }
        Ok(None)
    }
}

use std::io::SeekFrom;
use std::pin::Pin;
use std::sync::Arc;

use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::{Block, FooterRef, MessageHeaderRef};
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
    encoded_data: &mut EncodedData,
) -> PolarsResult<()>
where
    R: AsyncRead + AsyncSeek + Unpin + Send + 'a,
{
    debug_assert!(encoded_data.arrow_data.is_empty() && encoded_data.ipc_message.is_empty());
    let message = read_ipc_message_from_block(reader, block, &mut encoded_data.ipc_message).await?;

    let block_length: u64 = message
        .body_length()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferBodyLength(err)))?
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;
    reader
        .take(block_length)
        .read_to_end(&mut encoded_data.arrow_data)
        .await?;

    Ok(())
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
                let mut ed: EncodedData = Default::default();
                block_to_raw_message(reader, &d?, &mut ed).await?;
                yield ed
            }
        };

        for d in data_blocks {
                let mut ed: EncodedData = Default::default();
                block_to_raw_message(reader, &d?, &mut ed).await?;
                yield ed
        }
    })
}

pub struct FlightStreamProducer<'a, R: AsyncRead + AsyncSeek + Unpin + Send> {
    footer: Option<*const FooterRef<'static>>,
    footer_data: Vec<u8>,
    dict_blocks: Option<Box<dyn Iterator<Item = PolarsResult<Block>>>>,
    data_blocks: Option<Box<dyn Iterator<Item = PolarsResult<Block>>>>,
    reader: &'a mut R,
}

impl<'a, R: AsyncRead + AsyncSeek + Unpin + Send> Drop for FlightStreamProducer<'a, R> {
    fn drop(&mut self) {
        if let Some(p) = self.footer {
            unsafe {
                let _ = Box::from_raw(p as *mut FooterRef<'static>);
            }
        }
    }
}

impl<'a, R: AsyncRead + AsyncSeek + Unpin + Send> FlightStreamProducer<'a, R> {
    pub async fn new(reader: &'a mut R) -> PolarsResult<Self> {
        let (_end, len) = read_footer_len(reader).await?;
        let footer_data = read_footer(reader, len).await?;

        Ok(Self {
            footer: None,
            footer_data,
            dict_blocks: None,
            data_blocks: None,
            reader,
        })
    }

    pub fn init(self: &mut Pin<&mut Self>) -> PolarsResult<()> {
        let footer = arrow_format::ipc::FooterRef::read_as_root(&self.footer_data)
            .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferFooter(err)))?;

        let footer = Box::new(footer);

        #[allow(clippy::unnecessary_cast)]
        let ptr = Box::leak(footer) as *const _ as *const FooterRef<'static>;

        self.footer = Some(ptr);
        let footer = &unsafe { **self.footer.as_ref().unwrap() };

        self.data_blocks =
            Some(Box::new(iter_recordbatch_blocks_from_footer(*footer)?)
                as Box<dyn Iterator<Item = _>>);
        self.dict_blocks = iter_dictionary_blocks_from_footer(*footer)?
            .map(|i| Box::new(i) as Box<dyn Iterator<Item = _>>);

        Ok(())
    }

    pub fn get_schema(self: &Pin<&mut Self>) -> PolarsResult<EncodedData> {
        let footer = &unsafe { **self.footer.as_ref().expect("init must be called first") };

        let schema_ref = deserialize_schema_ref_from_footer(*footer)?;
        let schema = schema_to_raw_message(schema_ref);

        Ok(schema)
    }

    pub async fn next_dict(
        self: &mut Pin<&mut Self>,
        encoded_data: &mut EncodedData,
    ) -> PolarsResult<Option<()>> {
        assert!(self.data_blocks.is_some(), "init must be called first");
        encoded_data.ipc_message.clear();
        encoded_data.arrow_data.clear();

        if let Some(iter) = &mut self.dict_blocks {
            let Some(value) = iter.next() else {
                return Ok(None);
            };
            let block = value?;

            block_to_raw_message(&mut self.reader, &block, encoded_data).await?;
            Ok(Some(()))
        } else {
            Ok(None)
        }
    }

    pub async fn next_data(
        self: &mut Pin<&mut Self>,
        encoded_data: &mut EncodedData,
    ) -> PolarsResult<Option<()>> {
        encoded_data.ipc_message.clear();
        encoded_data.arrow_data.clear();

        let iter = self
            .data_blocks
            .as_mut()
            .expect("init must be called first");
        let Some(value) = iter.next() else {
            return Ok(None);
        };
        let block = value?;

        block_to_raw_message(&mut self.reader, &block, encoded_data).await?;
        Ok(Some(()))
    }
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

#[cfg(test)]
mod test {
    use std::path::{Path, PathBuf};

    use tokio::fs::File;

    use super::*;
    use crate::record_batch::RecordBatch;

    fn get_file_path() -> PathBuf {
        let polars_arrow = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        std::path::Path::new(&polars_arrow).join("../../py-polars/tests/unit/io/files/foods1.ipc")
    }

    fn read_file(path: &Path) -> RecordBatch {
        let mut file = std::fs::File::open(path).unwrap();
        let md = crate::io::ipc::read::read_file_metadata(&mut file).unwrap();
        let mut ipc_reader = crate::io::ipc::read::FileReader::new(&mut file, md, None, None);
        ipc_reader.next().unwrap().unwrap()
    }

    #[tokio::test]
    #[allow(clippy::needless_return)]
    async fn test_file_flight_simple() {
        let path = &get_file_path();
        let mut file = tokio::fs::File::open(path).await.unwrap();
        let stream = into_flight_stream(&mut file).await.unwrap();

        let mut c = FlightstreamConsumer::new(Box::pin(stream)).await.unwrap();
        let b = c.next_batch().await.unwrap().unwrap();

        assert_eq!(b, read_file(path));
    }

    #[tokio::test]
    #[allow(clippy::needless_return)]
    async fn test_file_flight_amortized() {
        let path = &get_file_path();
        let mut file = File::open(path).await.unwrap();
        let mut p = FlightStreamProducer::new(&mut file).await.unwrap();
        let mut p = std::pin::pin!(p);
        p.init().unwrap();

        let mut batches = vec![];

        let schema = p.get_schema().unwrap();
        batches.push(schema);

        let mut ed = EncodedData::default();
        if p.next_dict(&mut ed).await.unwrap().is_some() {
            batches.push(ed);
        }

        let mut ed = EncodedData::default();
        p.next_data(&mut ed).await.unwrap();
        batches.push(ed);

        let mut c =
            FlightstreamConsumer::new(Box::pin(futures::stream::iter(batches.into_iter().map(Ok))))
                .await
                .unwrap();
        let b = c.next_batch().await.unwrap().unwrap();

        assert_eq!(b, read_file(path));
    }
}

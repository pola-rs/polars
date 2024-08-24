use std::io::Read;

use arrow_format::ipc::planus::ReadAsRoot;
use polars_error::{polars_bail, polars_err, PolarsError, PolarsResult};
use polars_utils::aliases::PlHashMap;

use super::super::CONTINUATION_MARKER;
use super::common::*;
use super::schema::deserialize_stream_metadata;
use super::{Dictionaries, OutOfSpecKind};
use crate::array::Array;
use crate::datatypes::ArrowSchema;
use crate::io::ipc::IpcSchema;
use crate::record_batch::RecordBatchT;

/// Metadata of an Arrow IPC stream, written at the start of the stream
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    /// The schema that is read from the stream's first message
    pub schema: ArrowSchema,

    /// The IPC version of the stream
    pub version: arrow_format::ipc::MetadataVersion,

    /// The IPC fields tracking dictionaries
    pub ipc_schema: IpcSchema,
}

/// Reads the metadata of the stream
pub fn read_stream_metadata<R: Read>(reader: &mut R) -> PolarsResult<StreamMetadata> {
    // determine metadata length
    let mut meta_size: [u8; 4] = [0; 4];
    reader.read_exact(&mut meta_size)?;
    let meta_length = {
        // If a continuation marker is encountered, skip over it and read
        // the size from the next four bytes.
        if meta_size == CONTINUATION_MARKER {
            reader.read_exact(&mut meta_size)?;
        }
        i32::from_le_bytes(meta_size)
    };

    let length: usize = meta_length
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    let mut buffer = vec![];
    buffer.try_reserve(length)?;
    reader
        .by_ref()
        .take(length as u64)
        .read_to_end(&mut buffer)?;

    deserialize_stream_metadata(&buffer)
}

/// Encodes the stream's status after each read.
///
/// A stream is an iterator, and an iterator returns `Option<Item>`. The `Item`
/// type in the [`StreamReader`] case is `StreamState`, which means that an Arrow
/// stream may yield one of three values: (1) `None`, which signals that the stream
/// is done; (2) [`StreamState::Some`], which signals that there was
/// data waiting in the stream and we read it; and finally (3)
/// [`Some(StreamState::Waiting)`], which means that the stream is still "live", it
/// just doesn't hold any data right now.
pub enum StreamState {
    /// A live stream without data
    Waiting,
    /// Next item in the stream
    Some(RecordBatchT<Box<dyn Array>>),
}

impl StreamState {
    /// Return the data inside this wrapper.
    ///
    /// # Panics
    ///
    /// If the `StreamState` was `Waiting`.
    pub fn unwrap(self) -> RecordBatchT<Box<dyn Array>> {
        if let StreamState::Some(batch) = self {
            batch
        } else {
            panic!("The batch is not available")
        }
    }
}

/// Reads the next item, yielding `None` if the stream is done,
/// and a [`StreamState`] otherwise.
fn read_next<R: Read>(
    reader: &mut R,
    metadata: &StreamMetadata,
    dictionaries: &mut Dictionaries,
    message_buffer: &mut Vec<u8>,
    data_buffer: &mut Vec<u8>,
    projection: &Option<(Vec<usize>, PlHashMap<usize, usize>, ArrowSchema)>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Option<StreamState>> {
    // determine metadata length
    let mut meta_length: [u8; 4] = [0; 4];

    match reader.read_exact(&mut meta_length) {
        Ok(()) => (),
        Err(e) => {
            return if e.kind() == std::io::ErrorKind::UnexpectedEof {
                // Handle EOF without the "0xFFFFFFFF 0x00000000"
                // valid according to:
                // https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format
                Ok(Some(StreamState::Waiting))
            } else {
                Err(PolarsError::from(e))
            };
        },
    }

    let meta_length = {
        // If a continuation marker is encountered, skip over it and read
        // the size from the next four bytes.
        if meta_length == CONTINUATION_MARKER {
            reader.read_exact(&mut meta_length)?;
        }
        i32::from_le_bytes(meta_length)
    };

    let meta_length: usize = meta_length
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    if meta_length == 0 {
        // the stream has ended, mark the reader as finished
        return Ok(None);
    }

    message_buffer.clear();
    message_buffer.try_reserve(meta_length)?;
    reader
        .by_ref()
        .take(meta_length as u64)
        .read_to_end(message_buffer)?;

    let message = arrow_format::ipc::MessageRef::read_as_root(message_buffer.as_ref())
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))?;

    let header = message
        .header()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferHeader(err)))?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageHeader))?;

    let block_length: usize = message
        .body_length()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferBodyLength(err)))?
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::UnexpectedNegativeInteger))?;

    match header {
        arrow_format::ipc::MessageHeaderRef::RecordBatch(batch) => {
            data_buffer.clear();
            data_buffer.try_reserve(block_length)?;
            reader
                .by_ref()
                .take(block_length as u64)
                .read_to_end(data_buffer)?;

            let file_size = data_buffer.len() as u64;

            let mut reader = std::io::Cursor::new(data_buffer);

            let chunk = read_record_batch(
                batch,
                &metadata.schema.fields,
                &metadata.ipc_schema,
                projection.as_ref().map(|x| x.0.as_ref()),
                None,
                dictionaries,
                metadata.version,
                &mut reader,
                0,
                file_size,
                scratch,
            );

            if let Some((_, map, _)) = projection {
                // re-order according to projection
                chunk
                    .map(|chunk| apply_projection(chunk, map))
                    .map(|x| Some(StreamState::Some(x)))
            } else {
                chunk.map(|x| Some(StreamState::Some(x)))
            }
        },
        arrow_format::ipc::MessageHeaderRef::DictionaryBatch(batch) => {
            data_buffer.clear();
            data_buffer.try_reserve(block_length)?;
            reader
                .by_ref()
                .take(block_length as u64)
                .read_to_end(data_buffer)?;

            let file_size = data_buffer.len() as u64;
            let mut dict_reader = std::io::Cursor::new(&data_buffer);

            read_dictionary(
                batch,
                &metadata.schema.fields,
                &metadata.ipc_schema,
                dictionaries,
                &mut dict_reader,
                0,
                file_size,
                scratch,
            )?;

            // read the next message until we encounter a RecordBatch message
            read_next(
                reader,
                metadata,
                dictionaries,
                message_buffer,
                data_buffer,
                projection,
                scratch,
            )
        },
        _ => polars_bail!(oos = OutOfSpecKind::UnexpectedMessageType),
    }
}

/// Arrow Stream reader.
///
/// An [`Iterator`] over an Arrow stream that yields a result of [`StreamState`]s.
/// This is the recommended way to read an arrow stream (by iterating over its data).
///
/// For a more thorough walkthrough consult [this example](https://github.com/jorgecarleitao/polars_arrow/tree/main/examples/ipc_pyarrow).
pub struct StreamReader<R: Read> {
    reader: R,
    metadata: StreamMetadata,
    dictionaries: Dictionaries,
    finished: bool,
    data_buffer: Vec<u8>,
    message_buffer: Vec<u8>,
    projection: Option<(Vec<usize>, PlHashMap<usize, usize>, ArrowSchema)>,
    scratch: Vec<u8>,
}

impl<R: Read> StreamReader<R> {
    /// Try to create a new stream reader
    ///
    /// The first message in the stream is the schema, the reader will fail if it does not
    /// encounter a schema.
    /// To check if the reader is done, use `is_finished(self)`
    pub fn new(reader: R, metadata: StreamMetadata, projection: Option<Vec<usize>>) -> Self {
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
            finished: false,
            data_buffer: Default::default(),
            message_buffer: Default::default(),
            projection,
            scratch: Default::default(),
        }
    }

    /// Return the schema of the stream
    pub fn metadata(&self) -> &StreamMetadata {
        &self.metadata
    }

    /// Return the schema of the file
    pub fn schema(&self) -> &ArrowSchema {
        self.projection
            .as_ref()
            .map(|x| &x.2)
            .unwrap_or(&self.metadata.schema)
    }

    /// Check if the stream is finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn maybe_next(&mut self) -> PolarsResult<Option<StreamState>> {
        if self.finished {
            return Ok(None);
        }
        let batch = read_next(
            &mut self.reader,
            &self.metadata,
            &mut self.dictionaries,
            &mut self.message_buffer,
            &mut self.data_buffer,
            &self.projection,
            &mut self.scratch,
        )?;
        if batch.is_none() {
            self.finished = true;
        }
        Ok(batch)
    }
}

impl<R: Read> Iterator for StreamReader<R> {
    type Item = PolarsResult<StreamState>;

    fn next(&mut self) -> Option<Self::Item> {
        self.maybe_next().transpose()
    }
}

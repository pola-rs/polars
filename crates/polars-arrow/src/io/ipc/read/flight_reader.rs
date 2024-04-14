use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::MessageHeaderRef;
use polars_error::PolarsError::ComputeError;
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::OutOfSpecKind;
use crate::array::Array;
use crate::datatypes::ArrowSchema;
use crate::io::ipc::read::schema::deserialize_stream_metadata;
use crate::io::ipc::read::{read_dictionary, read_record_batch, Dictionaries, StreamMetadata};
use crate::io::ipc::write::common::EncodedData;
use crate::record_batch::RecordBatch;
pub struct FlightStreamReader {
    metadata: Option<StreamMetadata>,
    dictionaries: Dictionaries,
}

impl Default for FlightStreamReader {
    fn default() -> Self {
        Self {
            metadata: None,
            dictionaries: Default::default(),
        }
    }
}

impl FlightStreamReader {
    pub fn get_metadata(&self) -> PolarsResult<&ArrowSchema> {
        self.metadata
            .as_ref()
            .ok_or(ComputeError("Unknown schema".into()))
            .map(|metadata| &metadata.schema)
    }

    pub fn parse(
        &mut self,
        data: EncodedData,
    ) -> PolarsResult<Option<RecordBatch<Box<dyn Array>>>> {
        // First message should be the schema
        if self.metadata.is_none() {
            let message = arrow_format::ipc::MessageRef::read_as_root(data.ipc_message.as_ref())
                .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))?;
            let metadata = deserialize_stream_metadata(&message)?;
            let _ = std::mem::replace(&mut self.metadata, Some(metadata));
            Ok(None)
        } else {
            // Parse dictionary or batch
            // TODO: read_dictionary / read_record batch should also be async
            match convert_to_arrow_chunk(
                self.metadata.as_ref().unwrap(),
                &mut self.dictionaries,
                data,
            ) {
                // We parsed a dictionary so return None
                Ok(None) => Ok(None),
                // We parsed a record batch return the result
                Ok(Some(chunk)) => Ok(Some(chunk)),
                Err(e) => Err(e),
            }
        }
    }
}

fn convert_to_arrow_chunk(
    metadata: &StreamMetadata,
    dictionaries: &mut Dictionaries,
    flight_data: EncodedData,
) -> PolarsResult<Option<RecordBatch<Box<dyn Array>>>> {
    let EncodedData {
        ipc_message,
        arrow_data,
    } = flight_data;

    // Parse the header
    let message = arrow_format::ipc::MessageRef::read_as_root(ipc_message.as_ref())
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferMessage(err)))?;

    let header = message
            .header()
            .map_err(|err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidFlatbufferHeader(err)))?
            .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingMessageHeader))?;

    let data_size = arrow_data.len() as u64;
    let mut data_reader = std::io::Cursor::new(arrow_data);
    let mut scratch = vec![];

    // Either append to the dictionaries and return None or return Some(ArrowChunk)
    match header {
        MessageHeaderRef::Schema(_) => {
            polars_bail!(ComputeError: "Unexpected schema message while parsing Stream");
        },
        MessageHeaderRef::DictionaryBatch(batch) => {
            read_dictionary(
                batch,
                &metadata.schema.fields,
                &metadata.ipc_schema,
                dictionaries,
                &mut data_reader,
                0,
                data_size,
                &mut scratch,
            )?;
            Ok(None)
        },
        MessageHeaderRef::RecordBatch(batch) => read_record_batch(
            batch,
            &metadata.schema.fields,
            &metadata.ipc_schema,
            None,
            None,
            dictionaries,
            metadata.version,
            &mut data_reader,
            0,
            data_size,
            &mut scratch,
        )
        .map(Some),
        _ => {
            unimplemented!()
        },
    }
}

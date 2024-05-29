//! Serialization and deserialization to Arrow's flight protocol

use arrow_format::flight::data::{FlightData, SchemaResult};
use arrow_format::ipc;
use arrow_format::ipc::planus::ReadAsRoot;
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::ipc::read::Dictionaries;
pub use super::ipc::write::default_ipc_fields;
use super::ipc::{IpcField, IpcSchema};
use crate::array::Array;
use crate::datatypes::*;
pub use crate::io::ipc::write::common::WriteOptions;
use crate::io::ipc::write::common::{encode_chunk, DictionaryTracker, EncodedData};
use crate::io::ipc::{read, write};
use crate::record_batch::RecordBatchT;

/// Serializes [`RecordBatchT`] to a vector of [`FlightData`] representing the serialized dictionaries
/// and a [`FlightData`] representing the batch.
/// # Errors
/// This function errors iff `fields` is not consistent with `columns`
pub fn serialize_batch(
    chunk: &RecordBatchT<Box<dyn Array>>,
    fields: &[IpcField],
    options: &WriteOptions,
) -> PolarsResult<(Vec<FlightData>, FlightData)> {
    if fields.len() != chunk.arrays().len() {
        polars_bail!(oos = "The argument `fields` must be consistent with the columns' schema. Use e.g. &polars_arrow::io::flight::default_ipc_fields(&schema.fields)");
    }

    let mut dictionary_tracker = DictionaryTracker {
        dictionaries: Default::default(),
        cannot_replace: false,
    };

    let (encoded_dictionaries, encoded_batch) =
        encode_chunk(chunk, fields, &mut dictionary_tracker, options)
            .expect("DictionaryTracker configured above to not error on replacement");

    let flight_dictionaries = encoded_dictionaries.into_iter().map(Into::into).collect();
    let flight_batch = encoded_batch.into();

    Ok((flight_dictionaries, flight_batch))
}

impl From<EncodedData> for FlightData {
    fn from(data: EncodedData) -> Self {
        FlightData {
            data_header: data.ipc_message,
            data_body: data.arrow_data,
            ..Default::default()
        }
    }
}

/// Serializes a [`ArrowSchema`] to [`SchemaResult`].
pub fn serialize_schema_to_result(
    schema: &ArrowSchema,
    ipc_fields: Option<&[IpcField]>,
) -> SchemaResult {
    SchemaResult {
        schema: _serialize_schema(schema, ipc_fields),
    }
}

/// Serializes a [`ArrowSchema`] to [`FlightData`].
pub fn serialize_schema(schema: &ArrowSchema, ipc_fields: Option<&[IpcField]>) -> FlightData {
    FlightData {
        data_header: _serialize_schema(schema, ipc_fields),
        ..Default::default()
    }
}

/// Convert a [`ArrowSchema`] to bytes in the format expected in [`arrow_format::flight::data::FlightInfo`].
pub fn serialize_schema_to_info(
    schema: &ArrowSchema,
    ipc_fields: Option<&[IpcField]>,
) -> PolarsResult<Vec<u8>> {
    let encoded_data = if let Some(ipc_fields) = ipc_fields {
        schema_as_encoded_data(schema, ipc_fields)
    } else {
        let ipc_fields = default_ipc_fields(&schema.fields);
        schema_as_encoded_data(schema, &ipc_fields)
    };

    let mut schema = vec![];
    write::common_sync::write_message(&mut schema, &encoded_data)?;
    Ok(schema)
}

fn _serialize_schema(schema: &ArrowSchema, ipc_fields: Option<&[IpcField]>) -> Vec<u8> {
    if let Some(ipc_fields) = ipc_fields {
        write::schema_to_bytes(schema, ipc_fields)
    } else {
        let ipc_fields = default_ipc_fields(&schema.fields);
        write::schema_to_bytes(schema, &ipc_fields)
    }
}

fn schema_as_encoded_data(schema: &ArrowSchema, ipc_fields: &[IpcField]) -> EncodedData {
    EncodedData {
        ipc_message: write::schema_to_bytes(schema, ipc_fields),
        arrow_data: vec![],
    }
}

/// Deserialize an IPC message into [`ArrowSchema`], [`IpcSchema`].
/// Use to deserialize [`FlightData::data_header`] and [`SchemaResult::schema`].
pub fn deserialize_schemas(bytes: &[u8]) -> PolarsResult<(ArrowSchema, IpcSchema)> {
    read::deserialize_schema(bytes)
}

/// Deserializes [`FlightData`] representing a record batch message to [`RecordBatchT`].
pub fn deserialize_batch(
    data: &FlightData,
    fields: &[Field],
    ipc_schema: &IpcSchema,
    dictionaries: &read::Dictionaries,
) -> PolarsResult<RecordBatchT<Box<dyn Array>>> {
    // check that the data_header is a record batch message
    let message = arrow_format::ipc::MessageRef::read_as_root(&data.data_header)
        .map_err(|_err| polars_err!(oos = "Unable to get root as message: {err:?}"))?;

    let length = data.data_body.len();
    let mut reader = std::io::Cursor::new(&data.data_body);

    match message.header()?.ok_or_else(|| {
        polars_err!(oos = "Unable to convert flight data header to a record batch".to_string())
    })? {
        ipc::MessageHeaderRef::RecordBatch(batch) => read::read_record_batch(
            batch,
            fields,
            ipc_schema,
            None,
            None,
            dictionaries,
            message.version()?,
            &mut reader,
            0,
            length as u64,
            &mut Default::default(),
        ),
        _ => polars_bail!(oos = "flight currently only supports reading RecordBatch messages"),
    }
}

/// Deserializes [`FlightData`], assuming it to be a dictionary message, into `dictionaries`.
pub fn deserialize_dictionary(
    data: &FlightData,
    fields: &[Field],
    ipc_schema: &IpcSchema,
    dictionaries: &mut read::Dictionaries,
) -> PolarsResult<()> {
    let message = ipc::MessageRef::read_as_root(&data.data_header)?;

    let chunk = if let ipc::MessageHeaderRef::DictionaryBatch(chunk) = message
        .header()?
        .ok_or_else(|| polars_err!(oos = "Header is required"))?
    {
        chunk
    } else {
        return Ok(());
    };

    let length = data.data_body.len();
    let mut reader = std::io::Cursor::new(&data.data_body);
    read::read_dictionary(
        chunk,
        fields,
        ipc_schema,
        dictionaries,
        &mut reader,
        0,
        length as u64,
        &mut Default::default(),
    )?;

    Ok(())
}

/// Deserializes [`FlightData`] into either a [`RecordBatchT`] (when the message is a record batch)
/// or by upserting into `dictionaries` (when the message is a dictionary)
pub fn deserialize_message(
    data: &FlightData,
    fields: &[Field],
    ipc_schema: &IpcSchema,
    dictionaries: &mut Dictionaries,
) -> PolarsResult<Option<RecordBatchT<Box<dyn Array>>>> {
    let FlightData {
        data_header,
        data_body,
        ..
    } = data;

    let message = arrow_format::ipc::MessageRef::read_as_root(data_header)?;
    let header = message
        .header()?
        .ok_or_else(|| polars_err!(oos = "IPC Message must contain a header"))?;

    match header {
        ipc::MessageHeaderRef::RecordBatch(batch) => {
            let length = data_body.len();
            let mut reader = std::io::Cursor::new(data_body);

            let chunk = read::read_record_batch(
                batch,
                fields,
                ipc_schema,
                None,
                None,
                dictionaries,
                arrow_format::ipc::MetadataVersion::V5,
                &mut reader,
                0,
                length as u64,
                &mut Default::default(),
            )?;

            Ok(chunk.into())
        },
        ipc::MessageHeaderRef::DictionaryBatch(dict_batch) => {
            let length = data_body.len();
            let mut reader = std::io::Cursor::new(data_body);

            read::read_dictionary(
                dict_batch,
                fields,
                ipc_schema,
                dictionaries,
                &mut reader,
                0,
                length as u64,
                &mut Default::default(),
            )?;
            Ok(None)
        },
        t => polars_bail!(ComputeError:
            "Reading types other than record batches not yet supported, unable to read {t:?}"
        ),
    }
}

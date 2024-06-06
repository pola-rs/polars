use std::io::Cursor;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{ArrowSchema, ArrowSchemaRef, Field};
use arrow::io::ipc::read::{read_file_metadata, FileReader};
use arrow::io::ipc::write::*;
use arrow::io::ipc::IpcField;
use arrow::record_batch::RecordBatchT;
use polars_error::*;

pub(crate) fn write(
    batches: &[RecordBatchT<Box<dyn Array>>],
    schema: &ArrowSchemaRef,
    ipc_fields: Option<Vec<IpcField>>,
    compression: Option<Compression>,
) -> PolarsResult<Vec<u8>> {
    let result = vec![];
    let options = WriteOptions { compression };
    let mut writer = FileWriter::try_new(result, schema.clone(), ipc_fields.clone(), options)?;
    for batch in batches {
        writer.write(batch, ipc_fields.as_ref().map(|x| x.as_ref()))?;
    }
    writer.finish()?;
    Ok(writer.into_inner())
}

fn round_trip(
    columns: RecordBatchT<Box<dyn Array>>,
    schema: ArrowSchemaRef,
    ipc_fields: Option<Vec<IpcField>>,
    compression: Option<Compression>,
) -> PolarsResult<()> {
    let (expected_schema, expected_batches) = (schema.clone(), vec![columns]);

    let result = write(&expected_batches, &schema, ipc_fields, compression)?;
    let mut reader = Cursor::new(result);
    let metadata = read_file_metadata(&mut reader)?;
    let schema = metadata.schema.clone();

    let reader = FileReader::new(reader, metadata, None, None);

    assert_eq!(schema, expected_schema);

    let batches = reader.collect::<PolarsResult<Vec<_>>>()?;

    assert_eq!(batches, expected_batches);
    Ok(())
}

fn prep_schema(array: &dyn Array) -> ArrowSchemaRef {
    let fields = vec![Field::new("a", array.data_type().clone(), true)];
    Arc::new(ArrowSchema::from(fields))
}

#[test]
fn write_boolean() -> PolarsResult<()> {
    let array = BooleanArray::from([Some(true), Some(false), None, Some(true)]).boxed();
    let schema = prep_schema(array.as_ref());
    let columns = RecordBatchT::try_new(vec![array])?;
    round_trip(columns, schema, None, Some(Compression::ZSTD))
}

#[test]
fn write_sliced_utf8() -> PolarsResult<()> {
    let array = Utf8Array::<i32>::from_slice(["aa", "bb"])
        .sliced(1, 1)
        .boxed();
    let schema = prep_schema(array.as_ref());
    let columns = RecordBatchT::try_new(vec![array])?;
    round_trip(columns, schema, None, Some(Compression::ZSTD))
}

#[test]
fn write_binview() -> PolarsResult<()> {
    let array = Utf8ViewArray::from_slice([Some("foo"), Some("bar"), None, Some("hamlet")]).boxed();
    let schema = prep_schema(array.as_ref());
    let columns = RecordBatchT::try_new(vec![array])?;
    round_trip(columns, schema, None, Some(Compression::ZSTD))
}

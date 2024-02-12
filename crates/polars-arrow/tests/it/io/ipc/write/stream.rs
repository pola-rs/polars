use std::io::Cursor;

use polars_arrow::array::Array;
use polars_arrow::chunk::Chunk;
use polars_arrow::datatypes::ArrowSchema;
use polars_arrow::error::Result;
use polars_arrow::io::ipc::read::{read_stream_metadata, StreamReader};
use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};
use polars_arrow::io::ipc::IpcField;

use crate::io::ipc::common::{read_arrow_stream, read_gzip_json};

fn write_(
    schema: &ArrowSchema,
    ipc_fields: Option<Vec<IpcField>>,
    batches: &[Chunk<Box<dyn Array>>],
) -> Vec<u8> {
    let mut result = vec![];

    let options = WriteOptions { compression: None };
    let mut writer = StreamWriter::new(&mut result, options);
    writer.start(schema, ipc_fields).unwrap();
    for batch in batches {
        writer.write(batch, None).unwrap();
    }
    writer.finish().unwrap();
    result
}

fn test_file(version: &str, file_name: &str) {
    let (schema, ipc_fields, batches) = read_arrow_stream(version, file_name, None);

    let result = write_(&schema, Some(ipc_fields), &batches);

    let mut reader = Cursor::new(result);
    let metadata = read_stream_metadata(&mut reader).unwrap();
    let reader = StreamReader::new(reader, metadata, None);

    let schema = reader.metadata().schema.clone();
    let ipc_fields = reader.metadata().ipc_schema.fields.clone();

    // read expected JSON output
    let (expected_schema, expected_ipc_fields, expected_batches) =
        read_gzip_json(version, file_name).unwrap();

    assert_eq!(schema, expected_schema);
    assert_eq!(ipc_fields, expected_ipc_fields);

    let batches = reader
        .map(|x| x.map(|x| x.unwrap()))
        .collect::<Result<Vec<_>>>()
        .unwrap();

    assert_eq!(batches, expected_batches);
}

#[test]
fn write_100_primitive() {
    test_file("1.0.0-littleendian", "generated_primitive");
}

#[test]
fn write_100_datetime() {
    test_file("1.0.0-littleendian", "generated_datetime");
}

#[test]
fn write_100_dictionary_unsigned() {
    test_file("1.0.0-littleendian", "generated_dictionary_unsigned");
}

#[test]
fn write_100_dictionary() {
    test_file("1.0.0-littleendian", "generated_dictionary");
}

#[test]
fn write_100_interval() {
    test_file("1.0.0-littleendian", "generated_interval");
}

#[test]
fn write_100_large_batch() {
    // this takes too long for unit-tests. It has been passing...
    //test_file("1.0.0-littleendian", "generated_large_batch");
}

#[test]
fn write_100_nested() {
    test_file("1.0.0-littleendian", "generated_nested");
}

#[test]
fn write_100_nested_large_offsets() {
    test_file("1.0.0-littleendian", "generated_nested_large_offsets");
}

#[test]
fn write_100_null_trivial() {
    test_file("1.0.0-littleendian", "generated_null_trivial");
}

#[test]
fn write_100_null() {
    test_file("1.0.0-littleendian", "generated_null");
}

#[test]
fn write_100_primitive_large_offsets() {
    test_file("1.0.0-littleendian", "generated_primitive_large_offsets");
}

#[test]
fn write_100_union() {
    test_file("1.0.0-littleendian", "generated_union");
}

#[test]
fn write_generated_017_union() {
    test_file("0.17.1", "generated_union");
}

//#[test]
//fn write_100_recursive_nested() {
//test_file("1.0.0-littleendian", "generated_recursive_nested");
//}

#[test]
fn write_100_primitive_no_batches() {
    test_file("1.0.0-littleendian", "generated_primitive_no_batches");
}

#[test]
fn write_100_primitive_zerolength() {
    test_file("1.0.0-littleendian", "generated_primitive_zerolength");
}

#[test]
fn write_100_custom_metadata() {
    test_file("1.0.0-littleendian", "generated_custom_metadata");
}

#[test]
fn write_100_decimal() {
    test_file("1.0.0-littleendian", "generated_decimal");
}

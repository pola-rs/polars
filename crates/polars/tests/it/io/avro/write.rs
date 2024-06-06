use std::io::Cursor;

use arrow::array::*;
use arrow::datatypes::*;
use arrow::io::avro::avro_schema::file::{Block, CompressedBlock, Compression};
use arrow::io::avro::avro_schema::write::{compress, write_block, write_metadata};
use arrow::io::avro::write;
use arrow::record_batch::RecordBatchT;
use avro_schema::schema::{Field as AvroField, Record, Schema as AvroSchema};
use polars::io::avro::{AvroReader, AvroWriter};
use polars::io::{SerReader, SerWriter};
use polars::prelude::df;
use polars_error::PolarsResult;

use super::read::read_avro;

pub(super) fn schema() -> ArrowSchema {
    ArrowSchema::from(vec![
        Field::new("int64", ArrowDataType::Int64, false),
        Field::new("int64 nullable", ArrowDataType::Int64, true),
        Field::new("utf8", ArrowDataType::Utf8, false),
        Field::new("utf8 nullable", ArrowDataType::Utf8, true),
        Field::new("int32", ArrowDataType::Int32, false),
        Field::new("int32 nullable", ArrowDataType::Int32, true),
        Field::new("date", ArrowDataType::Date32, false),
        Field::new("date nullable", ArrowDataType::Date32, true),
        Field::new("binary", ArrowDataType::Binary, false),
        Field::new("binary nullable", ArrowDataType::Binary, true),
        Field::new("float32", ArrowDataType::Float32, false),
        Field::new("float32 nullable", ArrowDataType::Float32, true),
        Field::new("float64", ArrowDataType::Float64, false),
        Field::new("float64 nullable", ArrowDataType::Float64, true),
        Field::new("boolean", ArrowDataType::Boolean, false),
        Field::new("boolean nullable", ArrowDataType::Boolean, true),
        Field::new(
            "list",
            ArrowDataType::List(Box::new(Field::new("item", ArrowDataType::Int32, true))),
            false,
        ),
        Field::new(
            "list nullable",
            ArrowDataType::List(Box::new(Field::new("item", ArrowDataType::Int32, true))),
            true,
        ),
    ])
}

pub(super) fn data() -> RecordBatchT<Box<dyn Array>> {
    let list_dt = ArrowDataType::List(Box::new(Field::new("item", ArrowDataType::Int32, true)));
    let list_dt1 = ArrowDataType::List(Box::new(Field::new("item", ArrowDataType::Int32, true)));

    let columns = vec![
        Box::new(Int64Array::from_slice([27, 47])) as Box<dyn Array>,
        Box::new(Int64Array::from([Some(27), None])),
        Box::new(Utf8Array::<i32>::from_slice(["foo", "bar"])),
        Box::new(Utf8Array::<i32>::from([Some("foo"), None])),
        Box::new(Int32Array::from_slice([1, 1])),
        Box::new(Int32Array::from([Some(1), None])),
        Box::new(Int32Array::from_slice([1, 2]).to(ArrowDataType::Date32)),
        Box::new(Int32Array::from([Some(1), None]).to(ArrowDataType::Date32)),
        Box::new(BinaryArray::<i32>::from_slice([b"foo", b"bar"])),
        Box::new(BinaryArray::<i32>::from([Some(b"foo"), None])),
        Box::new(PrimitiveArray::<f32>::from_slice([1.0, 2.0])),
        Box::new(PrimitiveArray::<f32>::from([Some(1.0), None])),
        Box::new(PrimitiveArray::<f64>::from_slice([1.0, 2.0])),
        Box::new(PrimitiveArray::<f64>::from([Some(1.0), None])),
        Box::new(BooleanArray::from_slice([true, false])),
        Box::new(BooleanArray::from([Some(true), None])),
        Box::new(ListArray::<i32>::new(
            list_dt,
            vec![0, 2, 5].try_into().unwrap(),
            Box::new(PrimitiveArray::<i32>::from([
                None,
                Some(1),
                None,
                Some(3),
                Some(4),
            ])),
            None,
        )),
        Box::new(ListArray::<i32>::new(
            list_dt1,
            vec![0, 2, 2].try_into().unwrap(),
            Box::new(PrimitiveArray::<i32>::from([None, Some(1)])),
            Some([true, false].into()),
        )),
    ];

    RecordBatchT::new(columns)
}

pub(super) fn serialize_to_block<R: AsRef<dyn Array>>(
    columns: &RecordBatchT<R>,
    schema: &ArrowSchema,
    compression: Option<Compression>,
) -> PolarsResult<CompressedBlock> {
    let record = write::to_record(schema, "".to_string())?;

    let mut serializers = columns
        .arrays()
        .iter()
        .map(|x| x.as_ref())
        .zip(record.fields.iter())
        .map(|(array, field)| write::new_serializer(array, &field.schema))
        .collect::<Vec<_>>();
    let mut block = Block::new(columns.len(), vec![]);

    write::serialize(&mut serializers, &mut block);

    let mut compressed_block = CompressedBlock::default();

    compress(&mut block, &mut compressed_block, compression)?;

    Ok(compressed_block)
}

fn write_avro<R: AsRef<dyn Array>>(
    columns: &RecordBatchT<R>,
    schema: &ArrowSchema,
    compression: Option<Compression>,
) -> PolarsResult<Vec<u8>> {
    let compressed_block = serialize_to_block(columns, schema, compression)?;

    let avro_fields = write::to_record(schema, "".to_string())?;
    let mut file = vec![];

    write_metadata(&mut file, avro_fields, compression)?;

    write_block(&mut file, &compressed_block)?;

    Ok(file)
}

fn roundtrip(compression: Option<Compression>) -> PolarsResult<()> {
    let expected = data();
    let expected_schema = schema();

    let data = write_avro(&expected, &expected_schema, compression)?;

    let (result, read_schema) = read_avro(&data, None)?;

    assert_eq!(expected_schema, read_schema);
    for (c1, c2) in result.columns().iter().zip(expected.columns().iter()) {
        assert_eq!(c1.as_ref(), c2.as_ref());
    }
    Ok(())
}

#[test]
fn no_compression() -> PolarsResult<()> {
    roundtrip(None)
}

#[test]
fn snappy() -> PolarsResult<()> {
    roundtrip(Some(Compression::Snappy))
}

#[test]
fn deflate() -> PolarsResult<()> {
    roundtrip(Some(Compression::Deflate))
}

fn large_format_schema() -> ArrowSchema {
    ArrowSchema::from(vec![
        Field::new("large_utf8", ArrowDataType::LargeUtf8, false),
        Field::new("large_utf8_nullable", ArrowDataType::LargeUtf8, true),
        Field::new("large_binary", ArrowDataType::LargeBinary, false),
        Field::new("large_binary_nullable", ArrowDataType::LargeBinary, true),
    ])
}

fn large_format_data() -> RecordBatchT<Box<dyn Array>> {
    let columns = vec![
        Box::new(Utf8Array::<i64>::from_slice(["a", "b"])) as Box<dyn Array>,
        Box::new(Utf8Array::<i64>::from([Some("a"), None])),
        Box::new(BinaryArray::<i64>::from_slice([b"foo", b"bar"])),
        Box::new(BinaryArray::<i64>::from([Some(b"foo"), None])),
    ];
    RecordBatchT::new(columns)
}

fn large_format_expected_schema() -> ArrowSchema {
    ArrowSchema::from(vec![
        Field::new("large_utf8", ArrowDataType::Utf8, false),
        Field::new("large_utf8_nullable", ArrowDataType::Utf8, true),
        Field::new("large_binary", ArrowDataType::Binary, false),
        Field::new("large_binary_nullable", ArrowDataType::Binary, true),
    ])
}

fn large_format_expected_data() -> RecordBatchT<Box<dyn Array>> {
    let columns = vec![
        Box::new(Utf8Array::<i32>::from_slice(["a", "b"])) as Box<dyn Array>,
        Box::new(Utf8Array::<i32>::from([Some("a"), None])),
        Box::new(BinaryArray::<i32>::from_slice([b"foo", b"bar"])),
        Box::new(BinaryArray::<i32>::from([Some(b"foo"), None])),
    ];
    RecordBatchT::new(columns)
}

#[test]
fn check_large_format() -> PolarsResult<()> {
    let write_schema = large_format_schema();
    let write_data = large_format_data();

    let data = write_avro(&write_data, &write_schema, None)?;
    let (result, read_schame) = read_avro(&data, None)?;

    let expected_schema = large_format_expected_schema();
    assert_eq!(read_schame, expected_schema);

    let expected_data = large_format_expected_data();
    for (c1, c2) in result.columns().iter().zip(expected_data.columns().iter()) {
        assert_eq!(c1.as_ref(), c2.as_ref());
    }

    Ok(())
}

fn struct_schema() -> ArrowSchema {
    ArrowSchema::from(vec![
        Field::new(
            "struct",
            ArrowDataType::Struct(vec![
                Field::new("item1", ArrowDataType::Int32, false),
                Field::new("item2", ArrowDataType::Int32, true),
            ]),
            false,
        ),
        Field::new(
            "struct nullable",
            ArrowDataType::Struct(vec![
                Field::new("item1", ArrowDataType::Int32, false),
                Field::new("item2", ArrowDataType::Int32, true),
            ]),
            true,
        ),
    ])
}

fn struct_data() -> RecordBatchT<Box<dyn Array>> {
    let struct_dt = ArrowDataType::Struct(vec![
        Field::new("item1", ArrowDataType::Int32, false),
        Field::new("item2", ArrowDataType::Int32, true),
    ]);

    RecordBatchT::new(vec![
        Box::new(StructArray::new(
            struct_dt.clone(),
            vec![
                Box::new(PrimitiveArray::<i32>::from_slice([1, 2])),
                Box::new(PrimitiveArray::<i32>::from([None, Some(1)])),
            ],
            None,
        )),
        Box::new(StructArray::new(
            struct_dt,
            vec![
                Box::new(PrimitiveArray::<i32>::from_slice([1, 2])),
                Box::new(PrimitiveArray::<i32>::from([None, Some(1)])),
            ],
            Some([true, false].into()),
        )),
    ])
}

fn avro_record() -> Record {
    Record {
        name: "".to_string(),
        namespace: None,
        doc: None,
        aliases: vec![],
        fields: vec![
            AvroField {
                name: "struct".to_string(),
                doc: None,
                schema: AvroSchema::Record(Record {
                    name: "r1".to_string(),
                    namespace: None,
                    doc: None,
                    aliases: vec![],
                    fields: vec![
                        AvroField {
                            name: "item1".to_string(),
                            doc: None,
                            schema: AvroSchema::Int(None),
                            default: None,
                            order: None,
                            aliases: vec![],
                        },
                        AvroField {
                            name: "item2".to_string(),
                            doc: None,
                            schema: AvroSchema::Union(vec![
                                AvroSchema::Null,
                                AvroSchema::Int(None),
                            ]),
                            default: None,
                            order: None,
                            aliases: vec![],
                        },
                    ],
                }),
                default: None,
                order: None,
                aliases: vec![],
            },
            AvroField {
                name: "struct nullable".to_string(),
                doc: None,
                schema: AvroSchema::Union(vec![
                    AvroSchema::Null,
                    AvroSchema::Record(Record {
                        name: "r2".to_string(),
                        namespace: None,
                        doc: None,
                        aliases: vec![],
                        fields: vec![
                            AvroField {
                                name: "item1".to_string(),
                                doc: None,
                                schema: AvroSchema::Int(None),
                                default: None,
                                order: None,
                                aliases: vec![],
                            },
                            AvroField {
                                name: "item2".to_string(),
                                doc: None,
                                schema: AvroSchema::Union(vec![
                                    AvroSchema::Null,
                                    AvroSchema::Int(None),
                                ]),
                                default: None,
                                order: None,
                                aliases: vec![],
                            },
                        ],
                    }),
                ]),
                default: None,
                order: None,
                aliases: vec![],
            },
        ],
    }
}

#[test]
fn avro_record_schema() -> PolarsResult<()> {
    let arrow_schema = struct_schema();
    let record = write::to_record(&arrow_schema, "".to_string())?;
    assert_eq!(record, avro_record());
    Ok(())
}

#[test]
fn struct_() -> PolarsResult<()> {
    let write_schema = struct_schema();
    let write_data = struct_data();

    let data = write_avro(&write_data, &write_schema, None)?;
    let (result, read_schema) = read_avro(&data, None)?;

    let expected_schema = struct_schema();
    assert_eq!(read_schema, expected_schema);

    let expected_data = struct_data();
    for (c1, c2) in result.columns().iter().zip(expected_data.columns().iter()) {
        assert_eq!(c1.as_ref(), c2.as_ref());
    }

    Ok(())
}

#[test]
fn test_write_and_read_with_compression() -> PolarsResult<()> {
    let mut write_df = df!(
        "i64" => &[1, 2],
        "f64" => &[0.1, 0.2],
        "string" => &["a", "b"]
    )?;

    let compressions = vec![None, Some(Compression::Deflate), Some(Compression::Snappy)];

    for compression in compressions.into_iter() {
        let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

        AvroWriter::new(&mut buf)
            .with_compression(compression)
            .finish(&mut write_df)?;
        buf.set_position(0);

        let read_df = AvroReader::new(buf).finish()?;
        assert!(write_df.equals(&read_df));
    }

    Ok(())
}

#[test]
fn test_with_projection() -> PolarsResult<()> {
    let mut df = df!(
        "i64" => &[1, 2],
        "f64" => &[0.1, 0.2],
        "string" => &["a", "b"]
    )?;

    let expected_df = df!(
        "i64" => &[1, 2],
        "f64" => &[0.1, 0.2]
    )?;

    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

    AvroWriter::new(&mut buf).finish(&mut df)?;
    buf.set_position(0);

    let read_df = AvroReader::new(buf)
        .with_projection(Some(vec![0, 1]))
        .finish()?;

    assert!(expected_df.equals(&read_df));

    Ok(())
}

#[test]
fn test_with_columns() -> PolarsResult<()> {
    let mut df = df!(
        "i64" => &[1, 2],
        "f64" => &[0.1, 0.2],
        "string" => &["a", "b"]
    )?;

    let expected_df = df!(
        "i64" => &[1, 2],
        "string" => &["a", "b"]
    )?;

    let mut buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());

    AvroWriter::new(&mut buf).finish(&mut df)?;
    buf.set_position(0);

    let read_df = AvroReader::new(buf)
        .with_columns(Some(vec!["i64".to_string(), "string".to_string()]))
        .finish()?;

    assert!(expected_df.equals(&read_df));

    Ok(())
}

use std::sync::Arc;

use apache_avro::types::{Record, Value};
use apache_avro::{Codec, Days, Duration, Millis, Months, Schema as AvroSchema, Writer};
use arrow::array::*;
use arrow::datatypes::*;
use arrow::io::avro::avro_schema::read::read_metadata;
use arrow::io::avro::read;
use arrow::record_batch::RecordBatchT;
use polars_error::PolarsResult;

pub(super) fn schema() -> (AvroSchema, ArrowSchema) {
    let raw_schema = r#"
    {
        "type": "record",
        "name": "test",
        "fields": [
            {"name": "a", "type": "long"},
            {"name": "b", "type": "string"},
            {"name": "c", "type": "int"},
            {
                "name": "date",
                "type": "int",
                "logicalType": "date"
            },
            {"name": "d", "type": "bytes"},
            {"name": "e", "type": "double"},
            {"name": "f", "type": "boolean"},
            {"name": "g", "type": ["null", "string"], "default": null},
            {"name": "h", "type": {
                "type": "array",
                "items": {
                    "name": "item",
                    "type": ["null", "int"],
                    "default": null
                }
            }},
            {"name": "i", "type": {
                "type": "record",
                "name": "bla",
                "fields": [
                    {"name": "e", "type": "double"}
                ]
            }},
            {"name": "nullable_struct", "type": [
                "null", {
                    "type": "record",
                    "name": "foo",
                    "fields": [
                        {"name": "e", "type": "double"}
                    ]
                }]
                , "default": null
            }
        ]
    }
"#;

    let schema = ArrowSchema::from_iter([
        Field::new("a".into(), ArrowDataType::Int64, false),
        Field::new("b".into(), ArrowDataType::Utf8, false),
        Field::new("c".into(), ArrowDataType::Int32, false),
        Field::new("date".into(), ArrowDataType::Date32, false),
        Field::new("d".into(), ArrowDataType::Binary, false),
        Field::new("e".into(), ArrowDataType::Float64, false),
        Field::new("f".into(), ArrowDataType::Boolean, false),
        Field::new("g".into(), ArrowDataType::Utf8, true),
        Field::new(
            "h".into(),
            ArrowDataType::List(Box::new(Field::new(
                "item".into(),
                ArrowDataType::Int32,
                true,
            ))),
            false,
        ),
        Field::new(
            "i".into(),
            ArrowDataType::Struct(vec![Field::new("e".into(), ArrowDataType::Float64, false)]),
            false,
        ),
        Field::new(
            "nullable_struct".into(),
            ArrowDataType::Struct(vec![Field::new("e".into(), ArrowDataType::Float64, false)]),
            true,
        ),
    ]);

    (AvroSchema::parse_str(raw_schema).unwrap(), schema)
}

pub(super) fn data() -> RecordBatchT<Box<dyn Array>> {
    let data = vec![
        Some(vec![Some(1i32), None, Some(3)]),
        Some(vec![Some(1i32), None, Some(3)]),
    ];

    let mut array = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    array.try_extend(data).unwrap();

    let columns = vec![
        Int64Array::from_slice([27, 47]).boxed(),
        Utf8Array::<i32>::from_slice(["foo", "bar"]).boxed(),
        Int32Array::from_slice([1, 1]).boxed(),
        Int32Array::from_slice([1, 2])
            .to(ArrowDataType::Date32)
            .boxed(),
        BinaryArray::<i32>::from_slice([b"foo", b"bar"]).boxed(),
        PrimitiveArray::<f64>::from_slice([1.0, 2.0]).boxed(),
        BooleanArray::from_slice([true, false]).boxed(),
        Utf8Array::<i32>::from([Some("foo"), None]).boxed(),
        array.into_box(),
        StructArray::new(
            ArrowDataType::Struct(vec![Field::new("e".into(), ArrowDataType::Float64, false)]),
            2,
            vec![PrimitiveArray::<f64>::from_slice([1.0, 2.0]).boxed()],
            None,
        )
        .boxed(),
        StructArray::new(
            ArrowDataType::Struct(vec![Field::new("e".into(), ArrowDataType::Float64, false)]),
            2,
            vec![PrimitiveArray::<f64>::from_slice([1.0, 0.0]).boxed()],
            Some([true, false].into()),
        )
        .boxed(),
    ];

    let (_, schema) = schema();

    RecordBatchT::try_new(2, Arc::new(schema), columns).unwrap()
}

pub(super) fn write_avro(codec: Codec) -> Result<Vec<u8>, Box<apache_avro::Error>> {
    let (avro, _) = schema();
    // a writer needs a schema and something to write to
    let mut writer = Writer::with_codec(&avro, Vec::new(), codec);

    // the Record type models our Record schema
    let mut record = Record::new(writer.schema()).unwrap();
    record.put("a", 27i64);
    record.put("b", "foo");
    record.put("c", 1i32);
    record.put("date", 1i32);
    record.put("d", b"foo".as_ref());
    record.put("e", 1.0f64);
    record.put("f", true);
    record.put("g", Some("foo"));
    record.put(
        "h",
        Value::Array(vec![
            Value::Union(1, Box::new(Value::Int(1))),
            Value::Union(0, Box::new(Value::Null)),
            Value::Union(1, Box::new(Value::Int(3))),
        ]),
    );
    record.put(
        "i",
        Value::Record(vec![("e".to_string(), Value::Double(1.0f64))]),
    );
    record.put(
        "duration",
        Value::Duration(Duration::new(Months::new(1), Days::new(1), Millis::new(1))),
    );
    record.put(
        "nullable_struct",
        Value::Union(
            1,
            Box::new(Value::Record(vec![(
                "e".to_string(),
                Value::Double(1.0f64),
            )])),
        ),
    );
    writer.append(record)?;

    let mut record = Record::new(writer.schema()).unwrap();
    record.put("b", "bar");
    record.put("a", 47i64);
    record.put("c", 1i32);
    record.put("date", 2i32);
    record.put("d", b"bar".as_ref());
    record.put("e", 2.0f64);
    record.put("f", false);
    record.put("g", None::<&str>);
    record.put(
        "i",
        Value::Record(vec![("e".to_string(), Value::Double(2.0f64))]),
    );
    record.put(
        "h",
        Value::Array(vec![
            Value::Union(1, Box::new(Value::Int(1))),
            Value::Union(0, Box::new(Value::Null)),
            Value::Union(1, Box::new(Value::Int(3))),
        ]),
    );
    record.put("nullable_struct", Value::Union(0, Box::new(Value::Null)));
    writer.append(record)?;
    writer.into_inner().map_err(Box::new)
}

pub(super) fn read_avro(
    mut avro: &[u8],
    projection: Option<Vec<bool>>,
) -> PolarsResult<(RecordBatchT<Box<dyn Array>>, ArrowSchema)> {
    let file = &mut avro;

    let metadata = read_metadata(file)?;
    let schema = read::infer_schema(&metadata.record)?;

    let mut reader = read::Reader::new(file, metadata, schema.clone(), projection.clone());

    let schema = if let Some(projection) = projection {
        schema
            .into_iter_values()
            .zip(projection.iter())
            .filter_map(|x| if *x.1 { Some(x.0) } else { None })
            .collect()
    } else {
        schema
    };

    reader.next().unwrap().map(|x| (x, schema))
}

fn test(codec: Codec) -> PolarsResult<()> {
    let avro = write_avro(codec).unwrap();
    let expected = data();
    let (_, expected_schema) = schema();

    let (result, schema) = read_avro(&avro, None)?;

    assert_eq!(schema, expected_schema);
    assert_eq!(result, expected);
    Ok(())
}

#[test]
fn read_without_codec() -> PolarsResult<()> {
    test(Codec::Null)
}

#[test]
fn read_deflate() -> PolarsResult<()> {
    test(Codec::Deflate)
}

#[test]
fn read_snappy() -> PolarsResult<()> {
    test(Codec::Snappy)
}

#[test]
fn test_projected() -> PolarsResult<()> {
    let expected = data();
    let expected_schema = expected.schema();

    let avro = write_avro(Codec::Null).unwrap();

    for i in 0..expected_schema.len() {
        let mut projection = vec![false; expected_schema.len()];
        projection[i] = true;

        let length = expected.first().map_or(0, |arr| arr.len());
        let (expected_schema_2, expected_arrays) = expected.clone().into_schema_and_arrays();
        let expected_schema_2 = expected_schema_2
            .as_ref()
            .clone()
            .into_iter()
            .zip(projection.iter())
            .filter_map(|x| if *x.1 { Some(x.0) } else { None })
            .collect();
        let expected_arrays = expected_arrays
            .into_iter()
            .zip(projection.iter())
            .filter_map(|x| if *x.1 { Some(x.0) } else { None })
            .collect();
        let expected = RecordBatchT::new(length, Arc::new(expected_schema_2), expected_arrays);

        let expected_schema = expected_schema
            .clone()
            .into_iter_values()
            .zip(projection.iter())
            .filter_map(|x| if *x.1 { Some(x.0) } else { None })
            .collect();

        let (result, schema) = read_avro(&avro, Some(projection))?;

        assert_eq!(schema, expected_schema);
        assert_eq!(result, expected);
    }
    Ok(())
}

fn schema_list() -> (AvroSchema, ArrowSchema) {
    let raw_schema = r#"
    {
        "type": "record",
        "name": "test",
        "fields": [
            {"name": "h", "type": {
                "type": "array",
                "items": {
                    "name": "item",
                    "type": "int"
                }
            }}
        ]
    }
"#;

    let schema = ArrowSchema::from_iter([Field::new(
        "h".into(),
        ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            ArrowDataType::Int32,
            false,
        ))),
        false,
    )]);

    (AvroSchema::parse_str(raw_schema).unwrap(), schema)
}

pub(super) fn data_list() -> RecordBatchT<Box<dyn Array>> {
    let data = [Some(vec![Some(1i32), Some(2), Some(3)]), Some(vec![])];

    let mut array = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new_from(
        Default::default(),
        ArrowDataType::List(Box::new(Field::new(
            "item".into(),
            ArrowDataType::Int32,
            false,
        ))),
        0,
    );
    array.try_extend(data).unwrap();

    let length = array.len();
    let (_, schema) = schema_list();
    let columns = vec![array.into_box()];

    RecordBatchT::try_new(length, Arc::new(schema), columns).unwrap()
}

pub(super) fn write_list(codec: Codec) -> Result<Vec<u8>, Box<apache_avro::Error>> {
    let (avro, _) = schema_list();
    // a writer needs a schema and something to write to
    let mut writer = Writer::with_codec(&avro, Vec::new(), codec);

    // the Record type models our Record schema
    let mut record = Record::new(writer.schema()).unwrap();
    record.put(
        "h",
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
    );
    writer.append(record)?;

    let mut record = Record::new(writer.schema()).unwrap();
    record.put("h", Value::Array(vec![]));
    writer.append(record)?;
    Ok(writer.into_inner().unwrap())
}

#[test]
fn test_list() -> PolarsResult<()> {
    let avro = write_list(Codec::Null).unwrap();
    let expected = data_list();
    let expected_schema = expected.schema();

    let (result, schema) = read_avro(&avro, None)?;

    assert_eq!(&schema, expected_schema);
    assert_eq!(result, expected);
    Ok(())
}

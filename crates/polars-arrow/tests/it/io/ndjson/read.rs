use std::io::Cursor;

use polars_arrow::array::*;
use polars_arrow::datatypes::{ArrowDataType, Field};
use polars_arrow::error::{Error, Result};
use polars_arrow::io::ndjson::read as ndjson_read;
use polars_arrow::io::ndjson::read::FallibleStreamingIterator;

use super::*;

fn test_case(case_: &str) -> Result<()> {
    let (ndjson, expected) = case(case_);

    let data_type = expected.data_type().clone();

    let mut arrays = read_and_deserialize(&ndjson, &data_type, 1000)?;

    assert_eq!(arrays.len(), 1);
    assert_eq!(expected, arrays.pop().unwrap());
    Ok(())
}

pub fn infer(ndjson: &str) -> Result<ArrowDataType> {
    ndjson_read::infer(&mut Cursor::new(ndjson), None)
}

pub fn read_and_deserialize(
    ndjson: &str,
    data_type: &ArrowDataType,
    batch_size: usize,
) -> Result<Vec<Box<dyn Array>>> {
    let reader = Cursor::new(ndjson);

    let mut reader = ndjson_read::FileReader::new(reader, vec!["".to_string(); batch_size], None);

    let mut chunks = vec![];
    while let Some(rows) = reader.next()? {
        chunks.push(ndjson_read::deserialize(rows, data_type.clone())?);
    }

    Ok(chunks)
}

#[test]
fn infer_nullable() -> Result<()> {
    let ndjson = r#"true
    false
    null
    true
    "#;
    let expected = ArrowDataType::Boolean;

    let result = infer(ndjson)?;

    assert_eq!(result, expected);
    Ok(())
}

#[test]
fn read_null() -> Result<()> {
    let ndjson = "null";
    let expected_data_type = ArrowDataType::Null;

    let data_type = infer(ndjson)?;
    assert_eq!(expected_data_type, data_type);

    let arrays = read_and_deserialize(ndjson, &data_type, 1000)?;
    let expected = NullArray::new(data_type, 1);
    assert_eq!(expected, arrays[0].as_ref());
    Ok(())
}

#[test]
fn read_empty_reader() -> Result<()> {
    let ndjson = "";

    let infer_error = infer(ndjson);
    assert!(matches!(infer_error, Err(Error::ExternalFormat(_))));

    let deserialize_error = ndjson_read::deserialize(&[], ArrowDataType::Null);
    assert!(matches!(deserialize_error, Err(Error::ExternalFormat(_))));
    Ok(())
}

fn case_nested_struct() -> (String, Box<dyn Array>) {
    let ndjson = r#"{"a": {"a": 2.0, "b": 2}}
    {"a": {"b": 2}}
    {"a": {"a": 2.0, "b": 2, "c": true}}
    {"a": {"a": 2.0, "b": 2}}
    "#;

    let inner = ArrowDataType::Struct(vec![
        Field::new("a", ArrowDataType::Float64, true),
        Field::new("b", ArrowDataType::Int64, true),
        Field::new("c", ArrowDataType::Boolean, true),
    ]);

    let data_type = ArrowDataType::Struct(vec![Field::new("a", inner.clone(), true)]);

    let values = vec![
        Float64Array::from([Some(2.0), None, Some(2.0), Some(2.0)]).boxed(),
        Int64Array::from([Some(2), Some(2), Some(2), Some(2)]).boxed(),
        BooleanArray::from([None, None, Some(true), None]).boxed(),
    ];

    let values = vec![StructArray::new(inner, values, None).boxed()];

    let array = StructArray::new(data_type, values, None).boxed();

    (ndjson.to_string(), array)
}

#[test]
fn infer_nested_struct() -> Result<()> {
    let (ndjson, array) = case_nested_struct();

    let result = infer(&ndjson)?;

    assert_eq!(&result, array.data_type());
    Ok(())
}

#[test]
fn read_nested_struct() -> Result<()> {
    let (ndjson, expected) = case_nested_struct();

    let data_type = infer(&ndjson)?;

    let result = read_and_deserialize(&ndjson, &data_type, 100)?;

    assert_eq!(result, vec![expected]);
    Ok(())
}

#[test]
fn read_nested_struct_batched() -> Result<()> {
    let (ndjson, expected) = case_nested_struct();
    let batch_size = 2;

    // create a chunked array by batch_size from the (un-chunked) expected
    let expected: Vec<Box<dyn Array>> = (0..(expected.len() + batch_size - 1) / batch_size)
        .map(|offset| expected.sliced(offset * batch_size, batch_size))
        .collect();

    let data_type = infer(&ndjson)?;

    let result = read_and_deserialize(&ndjson, &data_type, batch_size)?;

    assert_eq!(result, expected,);
    Ok(())
}

#[test]
fn invalid_infer_schema() -> Result<()> {
    let re = ndjson_read::infer(&mut Cursor::new("city,lat,lng"), None);
    assert_eq!(
        re.err().unwrap().to_string(),
        "External format error: InvalidToken(99)",
    );
    Ok(())
}

#[test]
fn infer_schema_mixed_list() -> Result<()> {
    let ndjson = r#"{"a":1, "b":[2.0, 1.3, -6.1], "c":[false, true], "d":4.1}
    {"a":-10, "b":[2.0, 1.3, -6.1], "c":null, "d":null}
    {"a":2, "b":[2.0, null, -6.1], "c":[false, null], "d":"text"}
    {"a":3, "b":4, "c": true, "d":[1, false, "array", 2.4]}
    "#;

    let expected = ArrowDataType::Struct(vec![
        Field::new("a", ArrowDataType::Int64, true),
        Field::new(
            "b",
            ArrowDataType::List(Box::new(Field::new("item", ArrowDataType::Float64, true))),
            true,
        ),
        Field::new(
            "c",
            ArrowDataType::List(Box::new(Field::new("item", ArrowDataType::Boolean, true))),
            true,
        ),
        Field::new("d", ArrowDataType::Utf8, true),
    ]);

    let result = infer(ndjson)?;

    assert_eq!(result, expected);
    Ok(())
}

#[test]
fn basic() -> Result<()> {
    test_case("basics")
}

#[test]
fn projection() -> Result<()> {
    test_case("projection")
}

#[test]
fn dictionary() -> Result<()> {
    test_case("dict")
}

#[test]
fn list() -> Result<()> {
    test_case("list")
}

#[test]
fn nested_struct() -> Result<()> {
    test_case("struct")
}

#[test]
fn nested_list() -> Result<()> {
    test_case("nested_list")
}

#[test]
fn line_break_in_values() -> Result<()> {
    let ndjson = r#"
    "aa\n\n"
    "aa\n"
    null
    "#;

    let data_type = ArrowDataType::Utf8;
    let arrays = read_and_deserialize(ndjson, &data_type, 1000)?;

    let expected = Utf8Array::<i32>::from([Some("aa\n\n"), Some("aa\n"), None]);

    assert_eq!(expected, arrays[0].as_ref());
    Ok(())
}

#[test]
fn invalid_read_record() -> Result<()> {
    let fields = vec![Field::new(
        "a",
        ArrowDataType::Struct(vec![Field::new("a", ArrowDataType::Utf8, true)]),
        true,
    )];
    let data_type = ArrowDataType::Struct(fields);
    let arrays = read_and_deserialize("city,lat,lng", &data_type, 1000);

    assert_eq!(
        arrays.err().unwrap().to_string(),
        "External format error: InvalidToken(99)",
    );
    Ok(())
}

#[test]
fn skip_empty_lines() -> Result<()> {
    let ndjson = "
    {\"a\": 1}

    {\"a\": 2}

    {\"a\": 3}";

    let data_type = ArrowDataType::Struct(vec![Field::new("a", ArrowDataType::Int64, true)]);
    let arrays = read_and_deserialize(ndjson, &data_type, 1000)?;

    assert_eq!(1, arrays.len());
    assert_eq!(3, arrays[0].len());
    Ok(())
}

#[test]
fn utf8_array() -> Result<()> {
    let array = Utf8Array::<i64>::from([
        Some(r#"{"a": 1, "b": [{"c": 0}, {"c": 1}]}"#),
        None,
        Some(r#"{"a": 2, "b": [{"c": 2}, {"c": 5}]}"#),
        None,
    ]);
    let data_type = ndjson_read::infer_iter(array.iter().map(|x| x.unwrap_or("null"))).unwrap();
    let new_array =
        ndjson_read::deserialize_iter(array.iter().map(|x| x.unwrap_or("null")), data_type)
            .unwrap();

    // Explicitly cast as StructArray
    let new_array = new_array.as_any().downcast_ref::<StructArray>().unwrap();

    assert_eq!(array.len(), new_array.len());
    assert_eq!(array.null_count(), new_array.null_count());
    assert_eq!(array.validity().unwrap(), new_array.validity().unwrap());

    let field_names: Vec<String> = new_array.fields().iter().map(|f| f.name.clone()).collect();
    assert_eq!(field_names, vec!["a".to_string(), "b".to_string()]);
    Ok(())
}

use std::io::Cursor;
use std::num::NonZeroUsize;

use super::*;

#[test]
fn read_json() {
    let basic_json = r#"{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":-10, "b":-3.5, "c":true, "d":"4"}
{"a":2, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":7, "b":-3.5, "c":true, "d":"4"}
{"a":1, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":5, "b":-3.5, "c":true, "d":"4"}
{"a":1, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":1, "b":-3.5, "c":true, "d":"4"}
{"a":100000000000000, "b":0.6, "c":false, "d":"text"}
"#;
    let file = Cursor::new(basic_json);
    let df = JsonReader::new(file)
        .infer_schema_len(NonZeroUsize::new(3))
        .with_json_format(JsonFormat::JsonLines)
        .with_batch_size(NonZeroUsize::new(3).unwrap())
        .finish()
        .unwrap();
    assert_eq!("a", df.get_columns()[0].name());
    assert_eq!("d", df.get_columns()[3].name());
    assert_eq!((12, 4), df.shape());
}
#[test]
fn read_json_with_whitespace() {
    let basic_json = r#"{   "a":1, "b":2.0, "c"   :false  , "d":"4"}
{"a":-10, "b":-3.5, "c":true, "d":"4"}
{"a":2, "b":0.6, "c":false, "d":"text"   }
{"a":1, "b":2.0, "c":false, "d":"4"}


{"a":      7, "b":-3.5, "c":true, "d":"4"}
{"a":1, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d"  :"4"}
{"a":5, "b":-3.5, "c":true  , "d":"4"}

{"a":1, "b":0.6,   "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":1, "b":32.5,   "c":false, "d":"99"}
{  "a":100000000000000, "b":0.6, "c":false, "d":"text"}"#;
    let file = Cursor::new(basic_json);
    let df = JsonReader::new(file)
        .infer_schema_len(NonZeroUsize::new(3))
        .with_json_format(JsonFormat::JsonLines)
        .with_batch_size(NonZeroUsize::new(3).unwrap())
        .finish()
        .unwrap();
    assert_eq!("a", df.get_columns()[0].name());
    assert_eq!("d", df.get_columns()[3].name());
    assert_eq!((12, 4), df.shape());
}
#[test]
fn read_json_with_escapes() {
    let escaped_json = r#"{"id": 1, "text": "\""}
    {"text": "\n{\n\t\t\"inner\": \"json\n}\n", "id": 10}
    {"id": 0, "text":"\"","date":"2013-08-03 15:17:23"}
    {"id": 1, "text":"\"123\"","date":"2009-05-19 21:07:53"}
    {"id": 2, "text":"/....","date":"2009-05-19 21:07:53"}
    {"id": 3, "text":"\n\n..","date":"2"}
    {"id": 4, "text":"\"'/\n...","date":"2009-05-19 21:07:53"}
    {"id": 5, "text":".h\"h1hh\\21hi1e2emm...","date":"2009-05-19 21:07:53"}
    {"id": 6, "text":"xxxx....","date":"2009-05-19 21:07:53"}
    {"id": 7, "text":".\"quoted text\".","date":"2009-05-19 21:07:53"}

"#;
    let file = Cursor::new(escaped_json);
    let df = JsonLineReader::new(file)
        .infer_schema_len(NonZeroUsize::new(6))
        .finish()
        .unwrap();
    assert_eq!("id", df.get_columns()[0].name());
    assert_eq!(
        AnyValue::String("\""),
        df.column("text").unwrap().get(0).unwrap()
    );
    assert_eq!("text", df.get_columns()[1].name());
    assert_eq!((10, 3), df.shape());
}

#[test]
fn read_unordered_json() {
    let unordered_json = r#"{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":-10, "b":-3.5, "c":true, "d":"4"}
{"a":2, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":7, "b":-3.5, "c":true, "d":"4"}
{"a":1, "b":0.6, "c":false, "d":"text"}
{"d":1, "c":false, "d":"4", "b":2.0}
{"b":-3.5, "c":true, "d":"4", "a":5}
{"d":"text", "a":1, "c":false, "b":0.6}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":1, "b":-3.5, "c":true, "d":"4"}
{"a":100000000000000, "b":0.6, "c":false, "d":"text"}
"#;
    let file = Cursor::new(unordered_json);
    let df = JsonReader::new(file)
        .infer_schema_len(NonZeroUsize::new(3))
        .with_json_format(JsonFormat::JsonLines)
        .with_batch_size(NonZeroUsize::new(3).unwrap())
        .finish()
        .unwrap();
    assert_eq!("a", df.get_columns()[0].name());
    assert_eq!("d", df.get_columns()[3].name());
    assert_eq!((12, 4), df.shape());
}

#[test]
fn read_ndjson_with_trailing_newline() {
    let data = r#"{"Column1":"Value1"}
"#;

    let file = Cursor::new(data);
    let df = JsonReader::new(file)
        .with_json_format(JsonFormat::JsonLines)
        .finish()
        .unwrap();

    let expected = df! {
        "Column1" => ["Value1"]
    }
    .unwrap();
    assert!(expected.equals(&df));
}
#[test]
#[cfg(feature = "dtype-struct")]
fn test_read_ndjson_iss_5875() {
    let jsonlines = r#"
    {"struct": {"int_inner": [1, 2, 3], "float_inner": 5.0, "str_inner": ["a", "b", "c"]}}
    {"struct": {"int_inner": [4, 5, 6]}, "float": 4.0}
    "#;
    let cursor = Cursor::new(jsonlines);

    let df = JsonLineReader::new(cursor).finish();
    assert!(df.is_ok());

    let field_int_inner = Field::new("int_inner", DataType::List(Box::new(DataType::Int64)));
    let field_float_inner = Field::new("float_inner", DataType::Float64);
    let field_str_inner = Field::new("str_inner", DataType::List(Box::new(DataType::String)));

    let mut schema = Schema::new();
    schema.with_column(
        "struct".into(),
        DataType::Struct(vec![field_int_inner, field_float_inner, field_str_inner]),
    );
    schema.with_column("float".into(), DataType::Float64);

    assert_eq!(schema, df.unwrap().schema());
}

#[test]
#[cfg(feature = "dtype-struct")]
fn test_read_ndjson_iss_5875_part3() {
    let jsonlines = r#"
    {"key1":"value1", "key2": "value2", "key3": {"k1": 2, "k3": "value5", "k10": 5}}
    {"key1":"value5", "key2": "value4", "key3": {"k1": 2, "k5": "value5", "k10": 4}}
    {"key1":"value6", "key3": {"k1": 5, "k3": "value5"}}"#;

    let cursor = Cursor::new(jsonlines);

    let df = JsonLineReader::new(cursor).finish();
    assert!(df.is_ok());
}

#[test]
#[cfg(feature = "dtype-struct")]
fn test_read_ndjson_iss_6148() {
    let json = b"{\"a\":1,\"b\":{}}\n{\"a\":2,\"b\":{}}\n";

    let cursor = Cursor::new(json);

    let df = JsonLineReader::new(cursor).finish();
    assert!(df.is_ok());
}

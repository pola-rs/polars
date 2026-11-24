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
    assert_eq!("a", df.columns()[0].name().as_str());
    assert_eq!("d", df.columns()[3].name().as_str());
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
    assert_eq!("a", df.columns()[0].name().as_str());
    assert_eq!("d", df.columns()[3].name().as_str());
    assert_eq!((12, 4), df.shape());
}

#[test]
fn read_unordered_json() {
    let unordered_json = r#"{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":-10, "b":-3.5, "c":true, "d":"4"}
{"a":2, "b":0.6, "c":false, "d":"text"}
{"a":1, "b":2.0, "c":false, "d":"4"}
{"a":7, "b":-3.5, "c":true, "d":"4"}
{"a":1, "b":0.6, "c":false, "d":"text"}
{"d":"1", "c":false, "d":"4", "b":2.0}
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
    assert_eq!("a", df.columns()[0].name().as_str());
    assert_eq!("d", df.columns()[3].name().as_str());
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
fn write_ndjson_empty_chunk() -> PolarsResult<()> {
    let mut df = df!("x" => [1i64, 2])?;
    df.vstack_mut(&df!("x" => [3i64, 4])?)?;
    df.vstack_mut(&df!("x" => [5i64, 6])?)?;
    let mask = BooleanChunked::new("".into(), [true, true, false, false, true, true]);
    let mut df = df.filter(&mask)?;
    let chunk_lens = df.column("x")?.i64()?.chunk_lengths();
    assert_eq!(chunk_lens.collect::<Vec<_>>(), [2, 0, 2]);

    let mut buf = Vec::new();
    JsonWriter::new(&mut buf)
        .with_json_format(JsonFormat::JsonLines)
        .finish(&mut df)?;
    assert_eq!(
        std::str::from_utf8(&buf).unwrap(),
        "{\"x\":1}\n{\"x\":2}\n{\"x\":5}\n{\"x\":6}\n"
    );
    Ok(())
}

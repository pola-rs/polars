mod read;

use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::*;
use polars_arrow::error::Result;
use polars_arrow::io::ndjson::write as ndjson_write;
use read::{infer, read_and_deserialize};

fn round_trip(ndjson: String) -> Result<()> {
    let data_type = infer(&ndjson)?;

    let expected = read_and_deserialize(&ndjson, &data_type, 1000)?;

    let arrays = expected.clone().into_iter().map(Ok);

    let serializer = ndjson_write::Serializer::new(arrays, vec![]);

    let mut writer = ndjson_write::FileWriter::new(vec![], serializer);
    writer.by_ref().collect::<Result<()>>()?; // write
    let buf = writer.into_inner().0;

    let new_chunk = read_and_deserialize(std::str::from_utf8(&buf).unwrap(), &data_type, 1000)?;

    assert_eq!(expected, new_chunk);
    Ok(())
}

#[test]
fn round_trip_basics() -> Result<()> {
    let (data, _) = case_basics();
    round_trip(data)
}

#[test]
fn round_trip_list() -> Result<()> {
    let (data, _) = case_list();
    round_trip(data)
}

fn case_list() -> (String, Box<dyn Array>) {
    let data = r#"{"a":1, "b":[2.0, 1.3, -6.1], "c":[false, true], "d":"4"}
            {"a":-10, "b":null, "c":[true, true]}
            {"a":null, "b":[2.1, null, -6.2], "c":[false, null], "d":"text"}
            "#
    .to_string();

    let data_type = ArrowDataType::Struct(vec![
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

    let a = Int64Array::from(&[Some(1), Some(-10), None]);
    let mut b = MutableListArray::<i32, MutablePrimitiveArray<f64>>::new();
    b.try_extend(vec![
        Some(vec![Some(2.0), Some(1.3), Some(-6.1)]),
        None,
        Some(vec![Some(2.1), None, Some(-6.2)]),
    ])
    .unwrap();
    let b: ListArray<i32> = b.into();

    let mut c = MutableListArray::<i32, MutableBooleanArray>::new();
    c.try_extend(vec![
        Some(vec![Some(false), Some(true)]),
        Some(vec![Some(true), Some(true)]),
        Some(vec![Some(false), None]),
    ])
    .unwrap();
    let c: ListArray<i32> = c.into();

    let d = Utf8Array::<i32>::from([Some("4"), None, Some("text")]);

    let array = StructArray::new(
        data_type,
        vec![a.boxed(), b.boxed(), c.boxed(), d.boxed()],
        None,
    );

    (data, array.boxed())
}

fn case_dict() -> (String, Box<dyn Array>) {
    let data = r#"{"machine": "a", "events": [null, "Elect Leader", "Do Ballot"]}
    {"machine": "b", "events": ["Do Ballot", null, "Send Data", "Elect Leader"]}
    {"machine": "c", "events": ["Send Data"]}
    {"machine": "c"}
    {"machine": "c", "events": null}
    "#
    .to_string();

    let data_type = ArrowDataType::List(Box::new(Field::new(
        "item",
        ArrowDataType::Dictionary(u64::KEY_TYPE, Box::new(ArrowDataType::Utf8), false),
        true,
    )));

    let fields = vec![Field::new("events", data_type, true)];

    type A = MutableDictionaryArray<u64, MutableUtf8Array<i32>>;

    let mut array = MutableListArray::<i32, A>::new();
    array
        .try_extend(vec![
            Some(vec![None, Some("Elect Leader"), Some("Do Ballot")]),
            Some(vec![
                Some("Do Ballot"),
                None,
                Some("Send Data"),
                Some("Elect Leader"),
            ]),
            Some(vec![Some("Send Data")]),
            None,
            None,
        ])
        .unwrap();

    let array: ListArray<i32> = array.into();

    (
        data,
        StructArray::new(ArrowDataType::Struct(fields), vec![array.boxed()], None).boxed(),
    )
}

fn case_basics() -> (String, Box<dyn Array>) {
    let data = r#"{"a":1, "b":2.0, "c":false, "d":"4"}
    {"a":-10, "b":-3.5, "c":true, "d":null}
    {"a":100000000, "b":0.6, "d":"text"}"#
        .to_string();
    let data_type = ArrowDataType::Struct(vec![
        Field::new("a", ArrowDataType::Int64, true),
        Field::new("b", ArrowDataType::Float64, true),
        Field::new("c", ArrowDataType::Boolean, true),
        Field::new("d", ArrowDataType::Utf8, true),
    ]);
    let array = StructArray::new(
        data_type,
        vec![
            Int64Array::from_slice([1, -10, 100000000]).boxed(),
            Float64Array::from_slice([2.0, -3.5, 0.6]).boxed(),
            BooleanArray::from(&[Some(false), Some(true), None]).boxed(),
            Utf8Array::<i32>::from([Some("4"), None, Some("text")]).boxed(),
        ],
        None,
    );
    (data, array.boxed())
}

fn case_projection() -> (String, Box<dyn Array>) {
    let data = r#"{"a":1, "b":2.0, "c":false, "d":"4", "e":"4"}
    {"a":10, "b":-3.5, "c":true, "d":null, "e":"text"}
    {"a":100000000, "b":0.6, "d":"text"}"#
        .to_string();
    let data_type = ArrowDataType::Struct(vec![
        Field::new("a", ArrowDataType::UInt32, true),
        Field::new("b", ArrowDataType::Float32, true),
        Field::new("c", ArrowDataType::Boolean, true),
        // note how "d" is not here
        Field::new("e", ArrowDataType::Binary, true),
    ]);
    let array = StructArray::new(
        data_type,
        vec![
            UInt32Array::from_slice([1, 10, 100000000]).boxed(),
            Float32Array::from_slice([2.0, -3.5, 0.6]).boxed(),
            BooleanArray::from(&[Some(false), Some(true), None]).boxed(),
            BinaryArray::<i32>::from([Some(b"4".as_ref()), Some(b"text".as_ref()), None]).boxed(),
        ],
        None,
    );
    (data, array.boxed())
}

fn case_struct() -> (String, Box<dyn Array>) {
    let data = r#"{"a": {"b": true, "c": {"d": "text"}}}
    {"a": {"b": false, "c": null}}
    {"a": {"b": true, "c": {"d": "text"}}}
    {"a": 1}"#
        .to_string();

    let d_field = Field::new("d", ArrowDataType::Utf8, true);
    let c_field = Field::new("c", ArrowDataType::Struct(vec![d_field.clone()]), true);
    let a_field = Field::new(
        "a",
        ArrowDataType::Struct(vec![
            Field::new("b", ArrowDataType::Boolean, true),
            c_field.clone(),
        ]),
        true,
    );
    let fields = vec![a_field];

    // build expected output
    let d = Utf8Array::<i32>::from([Some("text"), None, Some("text"), None]);
    let c = StructArray::new(
        ArrowDataType::Struct(vec![d_field]),
        vec![d.boxed()],
        Some([true, false, true, true].into()),
    );

    let b = BooleanArray::from(vec![Some(true), Some(false), Some(true), None]);
    let inner = ArrowDataType::Struct(vec![Field::new("b", ArrowDataType::Boolean, true), c_field]);
    let expected = StructArray::new(
        inner,
        vec![b.boxed(), c.boxed()],
        Some([true, true, true, false].into()),
    );

    let data_type = ArrowDataType::Struct(fields);

    (
        data,
        StructArray::new(data_type, vec![expected.boxed()], None).boxed(),
    )
}

fn case_nested_list() -> (String, Box<dyn Array>) {
    let d_field = Field::new("d", ArrowDataType::Utf8, true);
    let c_field = Field::new("c", ArrowDataType::Struct(vec![d_field.clone()]), true);
    let b_field = Field::new("b", ArrowDataType::Boolean, true);
    let a_struct_field = Field::new(
        "a",
        ArrowDataType::Struct(vec![b_field.clone(), c_field.clone()]),
        true,
    );
    let a_list_data_type = ArrowDataType::List(Box::new(a_struct_field));
    let a_field = Field::new("a", a_list_data_type.clone(), true);

    let data = r#"
    {"a": [{"b": true, "c": {"d": "a_text"}}, {"b": false, "c": {"d": "b_text"}}]}
    {"a": [{"b": false, "c": null}]}
    {"a": [{"b": true, "c": {"d": "c_text"}}, {"b": null, "c": {"d": "d_text"}}, {"b": true, "c": {"d": null}}]}
    {"a": null}
    {"a": []}
    "#.to_string();

    // build expected output
    let d = Utf8Array::<i32>::from([
        Some("a_text"),
        Some("b_text"),
        None,
        Some("c_text"),
        Some("d_text"),
        None,
    ]);

    let c = StructArray::new(
        ArrowDataType::Struct(vec![d_field]),
        vec![d.boxed()],
        Some(Bitmap::from_u8_slice([0b11111011], 6)),
    );

    let b = BooleanArray::from(vec![
        Some(true),
        Some(false),
        Some(false),
        Some(true),
        None,
        Some(true),
    ]);
    let a_struct = StructArray::new(
        ArrowDataType::Struct(vec![b_field, c_field]),
        vec![b.boxed(), c.boxed()],
        None,
    );
    let expected = ListArray::new(
        a_list_data_type,
        vec![0i32, 2, 3, 6, 6, 6].try_into().unwrap(),
        a_struct.boxed(),
        Some([true, true, true, false, true].into()),
    );

    let array = StructArray::new(
        ArrowDataType::Struct(vec![a_field]),
        vec![expected.boxed()],
        None,
    )
    .boxed();

    (data, array)
}

fn case(case: &str) -> (String, Box<dyn Array>) {
    match case {
        "basics" => case_basics(),
        "projection" => case_projection(),
        "list" => case_list(),
        "dict" => case_dict(),
        "struct" => case_struct(),
        "nested_list" => case_nested_list(),
        _ => todo!(),
    }
}

#[test]
fn infer_object() -> Result<()> {
    let data = r#"{"i64": 1, "f64": 0.1, "utf8": "foo1", "bools": true}
    {"i64": 2, "f64": 0.2, "utf8": "foo2", "bools": false}
    {"i64": 3, "f64": 0.3, "utf8": "foo3"}
    {"i64": 4, "f64": 0.4, "utf8": "foo4", "bools": false}
    "#;
    let u64_fld = Field::new("i64", ArrowDataType::Int64, true);
    let f64_fld = Field::new("f64", ArrowDataType::Float64, true);
    let utf8_fld = Field::new("utf8", ArrowDataType::Utf8, true);
    let bools_fld = Field::new("bools", ArrowDataType::Boolean, true);

    let expected = ArrowDataType::Struct(vec![u64_fld, f64_fld, utf8_fld, bools_fld]);
    let actual = infer(data)?;

    assert_eq!(expected, actual);
    Ok(())
}

use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::buffer::Buffer;
use polars_arrow::datatypes::{ArrowDataType, Field, IntegerType, Metadata, Schema, TimeUnit};
use polars_arrow::error::Result;

use super::*;

macro_rules! test {
    ($array:expr, $expected:expr) => {{
        let buf = write_batch(Box::new($array))?;
        assert_eq!(String::from_utf8(buf).unwrap(), $expected);
        Ok(())
    }};
}

#[test]
fn int32() -> Result<()> {
    let array = Int32Array::from([Some(1), Some(2), Some(3), None, Some(5)]);

    let expected = r#"[1,2,3,null,5]"#;

    test!(array, expected)
}

#[test]
fn null() -> Result<()> {
    let array = NullArray::new(ArrowDataType::Null, 3);

    let expected = r#"[null,null,null]"#;

    test!(array, expected)
}

#[test]
fn f32() -> Result<()> {
    let array = Float32Array::from([
        Some(1.5),
        Some(2.5),
        Some(f32::NAN),
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
        None,
        Some(5.5),
    ]);

    let expected = r#"[1.5,2.5,null,null,null,null,5.5]"#;

    test!(array, expected)
}

#[test]
fn f64() -> Result<()> {
    let array = Float64Array::from([
        Some(1.5),
        Some(2.5),
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        None,
        Some(5.5),
    ]);

    let expected = r#"[1.5,2.5,null,null,null,null,5.5]"#;

    test!(array, expected)
}

#[test]
fn utf8() -> Result<()> {
    let array = Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c"), Some("d"), None]);

    let expected = r#"["a","b","c","d",null]"#;

    test!(array, expected)
}

#[test]
fn dictionary_utf8() -> Result<()> {
    let values = Utf8Array::<i64>::from([Some("a"), Some("b"), Some("c"), Some("d")]);
    let keys = PrimitiveArray::from_slice([0u32, 1, 2, 3, 1]);
    let array = DictionaryArray::try_new(
        ArrowDataType::Dictionary(
            IntegerType::UInt32,
            Box::new(ArrowDataType::LargeUtf8),
            false,
        ),
        keys,
        Box::new(values),
    )
    .unwrap();

    let expected = r#"["a","b","c","d","b"]"#;

    test!(array, expected)
}

#[test]
fn struct_() -> Result<()> {
    let c1 = Int32Array::from([Some(1), Some(2), Some(3), None, Some(5)]);
    let c2 = Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c"), Some("d"), None]);

    let data_type = ArrowDataType::Struct(vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("c2", c2.data_type().clone(), true),
    ]);
    let array = StructArray::new(data_type, vec![Box::new(c1) as _, Box::new(c2)], None);

    let expected = r#"[{"c1":1,"c2":"a"},{"c1":2,"c2":"b"},{"c1":3,"c2":"c"},{"c1":null,"c2":"d"},{"c1":5,"c2":null}]"#;

    test!(array, expected)
}

#[test]
fn nested_struct_with_validity() -> Result<()> {
    let inner = vec![
        Field::new("c121", ArrowDataType::Utf8, false),
        Field::new("c122", ArrowDataType::Int32, false),
    ];
    let fields = vec![
        Field::new("c11", ArrowDataType::Int32, false),
        Field::new("c12", ArrowDataType::Struct(inner.clone()), false),
    ];

    let c1 = StructArray::new(
        ArrowDataType::Struct(fields),
        vec![
            Int32Array::from(&[Some(1), None, Some(5)]).boxed(),
            StructArray::new(
                ArrowDataType::Struct(inner),
                vec![
                    Utf8Array::<i32>::from([None, Some("f"), Some("g")]).boxed(),
                    Int32Array::from(&[Some(20), None, Some(43)]).boxed(),
                ],
                Some(Bitmap::from([false, true, true])),
            )
            .boxed(),
        ],
        Some(Bitmap::from([true, true, false])),
    );
    let c2 = Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]);

    let data_type = ArrowDataType::Struct(vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("c2", c2.data_type().clone(), true),
    ]);
    let array = StructArray::new(data_type, vec![c1.boxed(), c2.boxed()], None);

    let expected = r#"[{"c1":{"c11":1,"c12":null},"c2":"a"},{"c1":{"c11":null,"c12":{"c121":"f","c122":null}},"c2":"b"},{"c1":null,"c2":"c"}]"#;

    test!(array, expected)
}

#[test]
fn nested_struct() -> Result<()> {
    let c121 = Field::new("c121", ArrowDataType::Utf8, false);
    let fields = vec![
        Field::new("c11", ArrowDataType::Int32, false),
        Field::new("c12", ArrowDataType::Struct(vec![c121.clone()]), false),
    ];

    let c1 = StructArray::new(
        ArrowDataType::Struct(fields),
        vec![
            Int32Array::from(&[Some(1), None, Some(5)]).boxed(),
            StructArray::new(
                ArrowDataType::Struct(vec![c121]),
                vec![Box::new(Utf8Array::<i32>::from([
                    Some("e"),
                    Some("f"),
                    Some("g"),
                ]))],
                None,
            )
            .boxed(),
        ],
        None,
    );

    let c2 = Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]);

    let data_type = ArrowDataType::Struct(vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("c2", c2.data_type().clone(), true),
    ]);
    let array = StructArray::new(data_type, vec![c1.boxed(), c2.boxed()], None);

    let expected = r#"[{"c1":{"c11":1,"c12":{"c121":"e"}},"c2":"a"},{"c1":{"c11":null,"c12":{"c121":"f"}},"c2":"b"},{"c1":{"c11":5,"c12":{"c121":"g"}},"c2":"c"}]"#;

    test!(array, expected)
}

#[test]
fn struct_with_list_field() -> Result<()> {
    let iter = vec![vec!["a", "a1"], vec!["b"], vec!["c"], vec!["d"], vec!["e"]];

    let iter = iter
        .into_iter()
        .map(|x| x.into_iter().map(Some).collect::<Vec<_>>())
        .map(Some);
    let mut a = MutableListArray::<i32, MutableUtf8Array<i32>>::new_with_field(
        MutableUtf8Array::<i32>::new(),
        "c_list",
        false,
    );
    a.try_extend(iter).unwrap();
    let c1: ListArray<i32> = a.into();

    let c2 = PrimitiveArray::from_slice([1, 2, 3, 4, 5]);

    let data_type = ArrowDataType::Struct(vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("c2", c2.data_type().clone(), true),
    ]);
    let array = StructArray::new(data_type, vec![c1.boxed(), c2.boxed()], None);

    let expected = r#"[{"c1":["a","a1"],"c2":1},{"c1":["b"],"c2":2},{"c1":["c"],"c2":3},{"c1":["d"],"c2":4},{"c1":["e"],"c2":5}]"#;

    test!(array, expected)
}

#[test]
fn nested_list() -> Result<()> {
    let iter = vec![
        vec![Some(vec![Some(1), Some(2)]), Some(vec![Some(3)])],
        vec![],
        vec![Some(vec![Some(4), Some(5), Some(6)])],
    ];

    let iter = iter.into_iter().map(Some);

    let inner = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new_with_field(
        MutablePrimitiveArray::<i32>::new(),
        "b",
        false,
    );
    let mut c1 =
        MutableListArray::<i32, MutableListArray<i32, MutablePrimitiveArray<i32>>>::new_with_field(
            inner, "a", false,
        );
    c1.try_extend(iter).unwrap();
    let c1: ListArray<i32> = c1.into();

    let c2 = Utf8Array::<i32>::from([Some("foo"), Some("bar"), None]);

    let data_type = ArrowDataType::Struct(vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("c2", c2.data_type().clone(), true),
    ]);
    let array = StructArray::new(data_type, vec![c1.boxed(), c2.boxed()], None);

    let expected =
        r#"[{"c1":[[1,2],[3]],"c2":"foo"},{"c1":[],"c2":"bar"},{"c1":[[4,5,6]],"c2":null}]"#;

    test!(array, expected)
}

#[test]
fn nested_list_records() -> Result<()> {
    let iter = vec![
        vec![Some(vec![Some(1), Some(2)]), Some(vec![Some(3)])],
        vec![],
        vec![Some(vec![Some(4), Some(5), Some(6)])],
    ];

    let iter = iter.into_iter().map(Some);

    let inner = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new_with_field(
        MutablePrimitiveArray::<i32>::new(),
        "b",
        false,
    );
    let mut c1 =
        MutableListArray::<i32, MutableListArray<i32, MutablePrimitiveArray<i32>>>::new_with_field(
            inner, "c1", false,
        );
    c1.try_extend(iter).unwrap();
    let c1: ListArray<i32> = c1.into();

    let c2 = Utf8Array::<i32>::from([Some("foo"), Some("bar"), None]);

    let schema: Schema = vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("c2", c2.data_type().clone(), true),
    ]
    .into();

    let arrays: Vec<Box<dyn Array>> = vec![Box::new(c1), Box::new(c2)];
    let chunk = Chunk::new(arrays);

    let expected =
        r#"[{"c1":[[1,2],[3]],"c2":"foo"},{"c1":[],"c2":"bar"},{"c1":[[4,5,6]],"c2":null}]"#;

    let buf = write_record_batch(schema, chunk)?;
    assert_eq!(String::from_utf8(buf).unwrap(), expected);
    Ok(())
}

#[test]
fn fixed_size_list_records() -> Result<()> {
    let iter = vec![
        vec![Some(1), Some(2), Some(3)],
        vec![Some(4), Some(5), Some(6)],
    ];

    let iter = iter.into_iter().map(Some);

    let mut inner = MutableFixedSizeListArray::<MutablePrimitiveArray<i32>>::new_with_field(
        MutablePrimitiveArray::new(),
        "vs",
        false,
        3,
    );
    inner.try_extend(iter).unwrap();
    let inner: FixedSizeListArray = inner.into();

    let schema = Schema {
        fields: vec![Field::new("vs", inner.data_type().clone(), true)],
        metadata: Metadata::default(),
    };

    let arrays: Vec<Box<dyn Array>> = vec![Box::new(inner)];
    let chunk = Chunk::new(arrays);

    let expected = r#"[{"vs":[1,2,3]},{"vs":[4,5,6]}]"#;

    let buf = write_record_batch(schema, chunk)?;
    assert_eq!(String::from_utf8(buf).unwrap(), expected);
    Ok(())
}

#[test]
fn list_of_struct() -> Result<()> {
    let inner = vec![Field::new("c121", ArrowDataType::Utf8, false)];
    let fields = vec![
        Field::new("c11", ArrowDataType::Int32, false),
        Field::new("c12", ArrowDataType::Struct(inner.clone()), false),
    ];
    let c1_datatype = ArrowDataType::List(Box::new(Field::new(
        "s",
        ArrowDataType::Struct(fields.clone()),
        false,
    )));

    let s = StructArray::new(
        ArrowDataType::Struct(fields),
        vec![
            Int32Array::from(&[Some(1), None, Some(5)]).boxed(),
            StructArray::new(
                ArrowDataType::Struct(inner),
                vec![Box::new(Utf8Array::<i32>::from([
                    Some("e"),
                    Some("f"),
                    Some("g"),
                ]))],
                Some(Bitmap::from([false, true, true])),
            )
            .boxed(),
        ],
        Some(Bitmap::from([true, true, false])),
    );

    // list column rows (c1):
    // [{"c11": 1, "c12": {"c121": "e"}}, {"c12": {"c121": "f"}}],
    // null,
    // [{"c11": 5, "c12": {"c121": "g"}}]
    let c1 = ListArray::<i32>::new(
        c1_datatype,
        Buffer::from(vec![0, 2, 2, 3]).try_into().unwrap(),
        s.boxed(),
        Some(Bitmap::from([true, false, true])),
    );

    let c2 = Int32Array::from_slice([1, 2, 3]);

    let data_type = ArrowDataType::Struct(vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("c2", c2.data_type().clone(), true),
    ]);
    let array = StructArray::new(data_type, vec![c1.boxed(), c2.boxed()], None);

    let expected = r#"[{"c1":[{"c11":1,"c12":null},{"c11":null,"c12":{"c121":"f"}}],"c2":1},{"c1":null,"c2":2},{"c1":[null],"c2":3}]"#;

    test!(array, expected)
}

#[test]
fn escaped_end_of_line_in_utf8() -> Result<()> {
    let array = Utf8Array::<i32>::from([Some("a\na"), None]);

    let expected = r#"["a\na",null]"#;

    test!(array, expected)
}

#[test]
fn escaped_quotation_marks_in_utf8() -> Result<()> {
    let array = Utf8Array::<i32>::from([Some("a\"a"), None]);

    let expected = r#"["a\"a",null]"#;

    test!(array, expected)
}

#[test]
fn write_date32() -> Result<()> {
    let array = PrimitiveArray::new(
        ArrowDataType::Date32,
        vec![1000i32, 8000, 10000].into(),
        None,
    );

    let expected = r#"["1972-09-27","1991-11-27","1997-05-19"]"#;

    test!(array, expected)
}

#[test]
fn write_timestamp() -> Result<()> {
    let array = PrimitiveArray::new(
        ArrowDataType::Timestamp(TimeUnit::Second, None),
        vec![10i64, 1 << 32, 1 << 33].into(),
        None,
    );

    let expected = r#"["1970-01-01 00:00:10","2106-02-07 06:28:16","2242-03-16 12:56:32"]"#;

    test!(array, expected)
}

#[test]
fn write_timestamp_with_tz_secs() -> Result<()> {
    let array = PrimitiveArray::new(
        ArrowDataType::Timestamp(TimeUnit::Second, Some("UTC".to_owned())),
        vec![10i64, 1 << 32, 1 << 33].into(),
        None,
    );

    let expected =
        r#"["1970-01-01T00:00:10+00:00","2106-02-07T06:28:16+00:00","2242-03-16T12:56:32+00:00"]"#;
    test!(array, expected)
}

#[test]
fn write_timestamp_with_tz_micros() -> Result<()> {
    let array = PrimitiveArray::new(
        ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("+02:00".to_owned())),
        vec![
            10i64 * 1_000_000,
            (1 << 32) * 1_000_000,
            (1 << 33) * 1_000_000,
            1_234_567_890_123_450,
            1_234_567_890_120_000,
        ]
        .into(),
        None,
    );
    // Note, default chrono DateTime string conversion strips off milli/micro/nanoseconds parts
    // if they are zero
    let expected = r#"["1970-01-01T02:00:10+02:00","2106-02-07T08:28:16+02:00","2242-03-16T14:56:32+02:00","2009-02-14T01:31:30.123450+02:00","2009-02-14T01:31:30.120+02:00"]"#;

    test!(array, expected)
}
#[cfg(feature = "chrono-tz")]
#[test]
fn write_timestamp_with_chrono_tz_millis() -> Result<()> {
    let array = PrimitiveArray::new(
        ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("Europe/Oslo".to_owned())),
        vec![
            10i64 * 1_000,
            (1 << 32) * 1_000,
            (1 << 33) * 1_000,
            1_234_567_890_123,
            1_239_874_560_120,
        ]
        .into(),
        None,
    );
    // Note, default chrono DateTime string conversion strips off milli/micro/nanoseconds parts
    // if they are zero
    let expected = r#"["1970-01-01T01:00:10+01:00","2106-02-07T07:28:16+01:00","2242-03-16T13:56:32+01:00","2009-02-14T00:31:30.123+01:00","2009-04-16T11:36:00.120+02:00"]"#;

    test!(array, expected)
}

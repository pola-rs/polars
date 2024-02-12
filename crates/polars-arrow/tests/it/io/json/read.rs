use polars_arrow::array::*;
use polars_arrow::datatypes::*;
use polars_arrow::error::Result;
use polars_arrow::io::json::read;

#[test]
fn read_json() -> Result<()> {
    let data = br#"[
        {
            "a": 1
        },
        {
            "a": 2
        },
        {
            "a": 3
        }
    ]"#;

    let json = json_deserializer::parse(data)?;

    let data_type = read::infer(&json)?;

    let result = read::deserialize(&json, data_type)?;

    let expected = StructArray::new(
        ArrowDataType::Struct(vec![Field::new("a", ArrowDataType::Int64, true)]),
        vec![Box::new(Int64Array::from_slice([1, 2, 3])) as _],
        None,
    );

    assert_eq!(expected, result.as_ref());

    Ok(())
}

#[test]
fn read_json_records() -> Result<()> {
    let data = br#"[
        {
            "a": [
                [1.1, 2, 3],
                [2, 3],
                [4, 5, 6]
            ],
            "b": [1, 2, 3],
            "c": ["test"],
            "d": [true]
        },
        {
            "a": [
                [3, 2, 1],
                [3, 2],
                [6, 5, 4]
            ]
        },
        {
            "b": [7, 8, 9],
            "c": ["string"],
            "d": [false]
        }
    ]"#;

    let a_iter = vec![
        vec![
            Some(vec![Some(1.1), Some(2.), Some(3.)]),
            Some(vec![Some(2.), Some(3.)]),
            Some(vec![Some(4.), Some(5.), Some(6.)]),
        ],
        vec![
            Some(vec![Some(3.), Some(2.), Some(1.)]),
            Some(vec![Some(3.), Some(2.)]),
            Some(vec![Some(6.), Some(5.), Some(4.)]),
        ],
    ];

    let a_iter = a_iter.into_iter().map(Some);
    let a_inner = MutableListArray::<i32, MutablePrimitiveArray<f64>>::new_with_field(
        MutablePrimitiveArray::<f64>::new(),
        "item",
        true,
    );
    let mut a_outer =
        MutableListArray::<i32, MutableListArray<i32, MutablePrimitiveArray<f64>>>::new_with_field(
            a_inner, "item", true,
        );
    a_outer.try_extend(a_iter).unwrap();
    let a_expected: ListArray<i32> = a_outer.into();

    let b_iter = vec![
        vec![Some(1), Some(2), Some(3)],
        vec![Some(7), Some(8), Some(9)],
    ];
    let b_iter = b_iter.into_iter().map(Some);
    let mut b = MutableListArray::<i32, MutablePrimitiveArray<i64>>::new_with_field(
        MutablePrimitiveArray::<i64>::new(),
        "item",
        true,
    );
    b.try_extend(b_iter).unwrap();
    let b_expected: ListArray<i32> = b.into();

    let c_iter = vec![vec![Some("test")], vec![Some("string")]];

    let c_iter = c_iter.into_iter().map(Some);
    let mut c = MutableListArray::<i32, MutableUtf8Array<i32>>::new_with_field(
        MutableUtf8Array::<i32>::new(),
        "item",
        true,
    );

    c.try_extend(c_iter).unwrap();
    let c_expected: ListArray<i32> = c.into();

    let d_iter = vec![vec![Some(true)], vec![Some(false)]];

    let d_iter = d_iter.into_iter().map(Some);
    let mut d = MutableListArray::<i32, MutableBooleanArray>::new_with_field(
        MutableBooleanArray::new(),
        "item",
        true,
    );

    d.try_extend(d_iter).unwrap();
    let d_expected: ListArray<i32> = d.into();

    let json = json_deserializer::parse(data)?;

    let schema = read::infer_records_schema(&json)?;
    let actual = read::deserialize_records(&json, &schema)?;

    for (f, arr) in schema.fields.iter().zip(actual.arrays().iter()) {
        let (expected, actual) = if f.name == "a" {
            (&a_expected, arr.as_ref())
        } else if f.name == "b" {
            (&b_expected, arr.as_ref())
        } else if f.name == "c" {
            (&c_expected, arr.as_ref())
        } else if f.name == "d" {
            (&d_expected, arr.as_ref())
        } else {
            panic!("unexpected field found: {}", f.name);
        };

        assert_eq!(expected.to_boxed().as_ref(), actual);
    }

    Ok(())
}

#[test]
fn read_json_fixed_size_records() -> Result<()> {
    let data = br#"[
        {
            "a": [1, 2.2, 3, 4]
        },
        {
            "a": [5, 6, 7, 8]
        },
        {
            "a": [7, 8, 9]
        }
    ]"#;

    let a_iter = vec![
        Some(vec![Some(1.), Some(2.2), Some(3.), Some(4.)]),
        Some(vec![Some(5.), Some(6.), Some(7.), Some(8.)]),
        None,
    ];

    let a_iter = a_iter.into_iter();
    let mut a = MutableFixedSizeListArray::<MutablePrimitiveArray<f64>>::new_with_field(
        MutablePrimitiveArray::<f64>::new(),
        "inner",
        true,
        4,
    );
    a.try_extend(a_iter).unwrap();
    let a_expected: FixedSizeListArray = a.into();

    let json = json_deserializer::parse(data)?;

    let schema: ArrowSchema = vec![Field::new("a", a_expected.data_type().clone(), true)].into();
    let actual = read::deserialize_records(&json, &schema)?;

    for (f, arr) in schema.fields.iter().zip(actual.arrays().iter()) {
        let (expected, actual) = if f.name == "a" {
            (&a_expected, arr.as_ref())
        } else {
            panic!("unexpected field found: {}", f.name);
        };

        assert_eq!(expected.to_boxed().as_ref(), actual);
    }

    Ok(())
}

#[test]
fn read_json_records_with_schema() -> Result<()> {
    let raw = b"[{\"matrix\":[0.0,2.0]},{\"matrix\":[0.0,0.0,2.1,3.0]}]";
    let schema = ArrowSchema {
        fields: vec![Field::new(
            "matrix",
            ArrowDataType::List(Box::new(Field::new("inner", ArrowDataType::Float32, false))),
            false,
        )],
        metadata: Metadata::default(),
    };

    let json = json_deserializer::parse(raw)?;
    let actual = read::deserialize_records(&json, &schema)?;
    assert_eq!(
        format!("{:?}", actual.arrays()[0]),
        "ListArray[[0, 2], [0, 0, 2.1, 3]]"
    );

    let schema = read::infer_records_schema(&json)?;
    let actual = read::deserialize_records(&json, &schema)?;
    assert_eq!(
        format!("{:?}", actual.arrays()[0]),
        "ListArray[[0, 2], [0, 0, 2.1, 3]]"
    );

    Ok(())
}

#[test]
fn deserialize_timestamp_string_ns() -> Result<()> {
    let data = br#"["2023-04-07T12:23:34.000000001Z"]"#;

    let json = json_deserializer::parse(data)?;

    let data_type = ArrowDataType::List(Box::new(Field::new(
        "item",
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
    )));

    let result = read::deserialize(&json, data_type)?;

    let expected = Int64Array::from([Some(1680870214000000001)])
        .to(ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));

    assert_eq!(expected, result.as_ref());

    Ok(())
}

#[test]
fn deserialize_timestamp_string_us() -> Result<()> {
    let data = br#"["2023-04-07T12:23:34.000000001Z"]"#;

    let json = json_deserializer::parse(data)?;

    let data_type = ArrowDataType::List(Box::new(Field::new(
        "item",
        ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )));

    let result = read::deserialize(&json, data_type)?;

    let expected = Int64Array::from([Some(1680870214000000)])
        .to(ArrowDataType::Timestamp(TimeUnit::Microsecond, None));

    assert_eq!(expected, result.as_ref());

    Ok(())
}

#[test]
fn deserialize_timestamp_string_ms() -> Result<()> {
    let data = br#"["2023-04-07T12:23:34.000000001Z"]"#;

    let json = json_deserializer::parse(data)?;

    let data_type = ArrowDataType::List(Box::new(Field::new(
        "item",
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
        false,
    )));

    let result = read::deserialize(&json, data_type)?;

    let expected = Int64Array::from([Some(1680870214000)])
        .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None));

    assert_eq!(expected, result.as_ref());

    Ok(())
}

#[test]
fn deserialize_timestamp_string_s() -> Result<()> {
    let data = br#"["2023-04-07T12:23:34.000000001Z"]"#;

    let json = json_deserializer::parse(data)?;

    let data_type = ArrowDataType::List(Box::new(Field::new(
        "item",
        ArrowDataType::Timestamp(TimeUnit::Second, None),
        false,
    )));

    let result = read::deserialize(&json, data_type)?;

    let expected =
        Int64Array::from([Some(1680870214)]).to(ArrowDataType::Timestamp(TimeUnit::Second, None));

    assert_eq!(expected, result.as_ref());

    Ok(())
}

#[test]
fn deserialize_timestamp_string_tz_s() -> Result<()> {
    let data = br#"["2023-04-07T12:23:34.000000001+00:00"]"#;

    let json = json_deserializer::parse(data)?;

    let data_type = ArrowDataType::List(Box::new(Field::new(
        "item",
        ArrowDataType::Timestamp(TimeUnit::Second, Some("+01:00".to_string())),
        false,
    )));

    let result = read::deserialize(&json, data_type)?;

    let expected = Int64Array::from([Some(1680870214)]).to(ArrowDataType::Timestamp(
        TimeUnit::Second,
        Some("+01:00".to_string()),
    ));

    assert_eq!(expected, result.as_ref());

    Ok(())
}

#[test]
fn deserialize_timestamp_int_ns() -> Result<()> {
    let data = br#"[1680870214000000001]"#;

    let json = json_deserializer::parse(data)?;

    let data_type = ArrowDataType::List(Box::new(Field::new(
        "item",
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
    )));

    let result = read::deserialize(&json, data_type)?;

    let expected = Int64Array::from([Some(1680870214000000001)])
        .to(ArrowDataType::Timestamp(TimeUnit::Nanosecond, None));

    assert_eq!(expected, result.as_ref());

    Ok(())
}

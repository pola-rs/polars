use polars_arrow::array::*;
use polars_arrow::compute::cast::{can_cast_types, cast, CastOptions};
use polars_arrow::datatypes::DataType::LargeList;
use polars_arrow::datatypes::*;
use polars_arrow::types::{days_ms, months_days_ns, NativeType};

#[test]
fn i32_to_f64() {
    let array = Int32Array::from_slice([5, 6, 7, 8, 9]);
    let b = cast(&array, &DataType::Float64, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<Float64Array>().unwrap();
    assert!((5.0 - c.value(0)).abs() < f64::EPSILON);
    assert!((6.0 - c.value(1)).abs() < f64::EPSILON);
    assert!((7.0 - c.value(2)).abs() < f64::EPSILON);
    assert!((8.0 - c.value(3)).abs() < f64::EPSILON);
    assert!((9.0 - c.value(4)).abs() < f64::EPSILON);
}

#[test]
fn i32_as_f64_no_overflow() {
    let array = Int32Array::from_slice([5, 6, 7, 8, 9]);
    let b = cast(
        &array,
        &DataType::Float64,
        CastOptions {
            wrapped: true,
            ..Default::default()
        },
    )
    .unwrap();
    let c = b.as_any().downcast_ref::<Float64Array>().unwrap();
    assert!((5.0 - c.value(0)).abs() < f64::EPSILON);
    assert!((6.0 - c.value(1)).abs() < f64::EPSILON);
    assert!((7.0 - c.value(2)).abs() < f64::EPSILON);
    assert!((8.0 - c.value(3)).abs() < f64::EPSILON);
    assert!((9.0 - c.value(4)).abs() < f64::EPSILON);
}

#[test]
fn u16_as_u8_overflow() {
    let array = UInt16Array::from_slice([255, 256, 257, 258, 259]);
    let b = cast(
        &array,
        &DataType::UInt8,
        CastOptions {
            wrapped: true,
            ..Default::default()
        },
    )
    .unwrap();
    let c = b.as_any().downcast_ref::<UInt8Array>().unwrap();
    let values = c.values().as_slice();

    assert_eq!(values, &[255, 0, 1, 2, 3])
}

#[test]
fn u16_as_u8_no_overflow() {
    let array = UInt16Array::from_slice([1, 2, 3, 4, 5]);
    let b = cast(
        &array,
        &DataType::UInt8,
        CastOptions {
            wrapped: true,
            ..Default::default()
        },
    )
    .unwrap();
    let c = b.as_any().downcast_ref::<UInt8Array>().unwrap();
    let values = c.values().as_slice();
    assert_eq!(values, &[1, 2, 3, 4, 5])
}

#[test]
fn f32_as_u8_overflow() {
    let array = Float32Array::from_slice([1.1, 5000.0]);
    let b = cast(&array, &DataType::UInt8, CastOptions::default()).unwrap();
    let expected = UInt8Array::from(&[Some(1), None]);
    assert_eq!(expected, b.as_ref());

    let b = cast(
        &array,
        &DataType::UInt8,
        CastOptions {
            wrapped: true,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = UInt8Array::from(&[Some(1), Some(255)]);
    assert_eq!(expected, b.as_ref());
}

#[test]
fn i32_to_u8() {
    let array = Int32Array::from_slice([-5, 6, -7, 8, 100000000]);
    let b = cast(&array, &DataType::UInt8, CastOptions::default()).unwrap();
    let expected = UInt8Array::from(&[None, Some(6), None, Some(8), None]);
    let c = b.as_any().downcast_ref::<UInt8Array>().unwrap();
    assert_eq!(c, &expected);
}

#[test]
fn i32_to_u8_sliced() {
    let array = Int32Array::from_slice([-5, 6, -7, 8, 100000000]);
    let array = array.sliced(2, 3);
    let b = cast(&array, &DataType::UInt8, CastOptions::default()).unwrap();
    let expected = UInt8Array::from(&[None, Some(8), None]);
    let c = b.as_any().downcast_ref::<UInt8Array>().unwrap();
    assert_eq!(c, &expected);
}

#[test]
fn i32_to_i32() {
    let array = Int32Array::from_slice([5, 6, 7, 8, 9]);
    let b = cast(&array, &DataType::Int32, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = &[5, 6, 7, 8, 9];
    let expected = Int32Array::from_slice(expected);
    assert_eq!(c, &expected);
}

#[test]
fn i32_to_large_list_i32() {
    let array = Int32Array::from_slice([5, 6, 7, 8, 9]);
    let b = cast(
        &array,
        &LargeList(Box::new(Field::new("item", DataType::Int32, true))),
        CastOptions::default(),
    )
    .unwrap();

    let arr = b.as_any().downcast_ref::<ListArray<i64>>().unwrap();
    assert_eq!(&[0, 1, 2, 3, 4, 5], arr.offsets().as_slice());
    let values = arr.values();
    let c = values
        .as_any()
        .downcast_ref::<PrimitiveArray<i32>>()
        .unwrap();

    let expected = Int32Array::from_slice([5, 6, 7, 8, 9]);
    assert_eq!(c, &expected);
}

#[test]
fn i32_to_list_i32() {
    let array = Int32Array::from_slice([5, 6, 7, 8, 9]);
    let b = cast(
        &array,
        &DataType::List(Box::new(Field::new("item", DataType::Int32, true))),
        CastOptions::default(),
    )
    .unwrap();

    let arr = b.as_any().downcast_ref::<ListArray<i32>>().unwrap();
    assert_eq!(&[0, 1, 2, 3, 4, 5], arr.offsets().as_slice());
    let values = arr.values();
    let c = values
        .as_any()
        .downcast_ref::<PrimitiveArray<i32>>()
        .unwrap();

    let expected = Int32Array::from_slice([5, 6, 7, 8, 9]);
    assert_eq!(c, &expected);
}

#[test]
fn i32_to_list_i32_nullable() {
    let input = [Some(5), None, Some(7), Some(8), Some(9)];

    let array = Int32Array::from(input);
    let b = cast(
        &array,
        &DataType::List(Box::new(Field::new("item", DataType::Int32, true))),
        CastOptions::default(),
    )
    .unwrap();

    let arr = b.as_any().downcast_ref::<ListArray<i32>>().unwrap();
    assert_eq!(&[0, 1, 2, 3, 4, 5], arr.offsets().as_slice());
    let values = arr.values();
    let c = values.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = &[Some(5), None, Some(7), Some(8), Some(9)];
    let expected = Int32Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn i32_to_list_f64_nullable_sliced() {
    let input = [Some(5), None, Some(7), Some(8), None, Some(10)];

    let array = Int32Array::from(input);

    let array = array.sliced(2, 4);
    let b = cast(
        &array,
        &DataType::List(Box::new(Field::new("item", DataType::Float64, true))),
        CastOptions::default(),
    )
    .unwrap();

    let arr = b.as_any().downcast_ref::<ListArray<i32>>().unwrap();
    assert_eq!(&[0, 1, 2, 3, 4], arr.offsets().as_slice());
    let values = arr.values();
    let c = values.as_any().downcast_ref::<Float64Array>().unwrap();

    let expected = &[Some(7.0), Some(8.0), None, Some(10.0)];
    let expected = Float64Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn i32_to_binary() {
    let array = Int32Array::from_slice([5, 6, 7]);
    let b = cast(&array, &DataType::Binary, CastOptions::default()).unwrap();
    let expected = BinaryArray::<i32>::from([Some(b"5"), Some(b"6"), Some(b"7")]);
    let c = b.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
    assert_eq!(c, &expected);
}

#[test]
fn binary_to_i32() {
    let array = BinaryArray::<i32>::from_slice(["5", "6", "seven", "8", "9.1"]);
    let b = cast(&array, &DataType::Int32, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();

    let expected = &[Some(5), Some(6), None, Some(8), None];
    let expected = Int32Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn binary_to_i32_partial() {
    let array = BinaryArray::<i32>::from_slice(["5", "6", "123 abseven", "aaa", "9.1"]);
    let b = cast(
        &array,
        &DataType::Int32,
        CastOptions {
            partial: true,
            ..Default::default()
        },
    )
    .unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();

    let expected = &[Some(5), Some(6), Some(123), Some(0), Some(9)];
    let expected = Int32Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn fixed_size_binary_to_binary() {
    let slice = [[0, 1], [2, 3]];
    let array = FixedSizeBinaryArray::from_slice(slice);

    // large-binary
    let b = cast(
        &array,
        &DataType::LargeBinary,
        CastOptions {
            ..Default::default()
        },
    )
    .unwrap();
    let c = b.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
    let expected = BinaryArray::<i64>::from_slice(slice);
    assert_eq!(c, &expected);

    // binary
    let b = cast(
        &array,
        &DataType::Binary,
        CastOptions {
            ..Default::default()
        },
    )
    .unwrap();
    let c = b.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();
    let expected = BinaryArray::<i32>::from_slice(slice);
    assert_eq!(c, &expected);
}

#[test]
fn utf8_to_i32() {
    let array = Utf8Array::<i32>::from_slice(["5", "6", "seven", "8", "9.1"]);
    let b = cast(&array, &DataType::Int32, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();

    let expected = &[Some(5), Some(6), None, Some(8), None];
    let expected = Int32Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn int32_to_decimal() {
    // 10 and -10 can be represented with precision 1 and scale 0
    let array = Int32Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None]);

    let b = cast(&array, &DataType::Decimal(1, 0), CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i128>>().unwrap();

    let expected = Int128Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None])
        .to(DataType::Decimal(1, 0));
    assert_eq!(c, &expected)
}

#[test]
fn float32_to_decimal() {
    let array = Float32Array::from(&[
        Some(2.4),
        Some(10.0),
        Some(1.123_456_8),
        Some(-2.0),
        Some(-10.0),
        Some(-100.01), // can't be represented in (1,0)
        None,
    ]);

    let b = cast(&array, &DataType::Decimal(10, 2), CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i128>>().unwrap();

    let expected = Int128Array::from(&[
        Some(240),
        Some(1000),
        Some(112),
        Some(-200),
        Some(-1000),
        Some(-10001),
        None,
    ])
    .to(DataType::Decimal(10, 2));
    assert_eq!(c, &expected)
}

#[test]
fn int32_to_decimal_scaled() {
    // 10 and -10 can't be represented with precision 1 and scale 1
    let array = Int32Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None]);

    let b = cast(&array, &DataType::Decimal(1, 1), CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i128>>().unwrap();

    let expected =
        Int128Array::from(&[Some(20), None, Some(-20), None, None]).to(DataType::Decimal(1, 1));
    assert_eq!(c, &expected)
}

#[test]
fn decimal_to_decimal() {
    // increase scale and precision
    let array = Int128Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None])
        .to(DataType::Decimal(1, 0));

    let b = cast(&array, &DataType::Decimal(2, 1), CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i128>>().unwrap();

    let expected = Int128Array::from(&[Some(20), Some(100), Some(-20), Some(-100), None])
        .to(DataType::Decimal(2, 1));
    assert_eq!(c, &expected)
}

#[test]
fn decimal_to_decimal_scaled() {
    // decrease precision
    // 10 and -10 can't be represented with precision 1 and scale 1
    let array = Int128Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None])
        .to(DataType::Decimal(1, 0));

    let b = cast(&array, &DataType::Decimal(1, 1), CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i128>>().unwrap();

    let expected =
        Int128Array::from(&[Some(20), None, Some(-20), None, None]).to(DataType::Decimal(1, 1));
    assert_eq!(c, &expected)
}

#[test]
fn decimal_to_decimal_fast() {
    // increase precision
    // 10 and -10 can't be represented with precision 1 and scale 1
    let array = Int128Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None])
        .to(DataType::Decimal(1, 1));

    let b = cast(&array, &DataType::Decimal(2, 1), CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i128>>().unwrap();

    let expected = Int128Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None])
        .to(DataType::Decimal(2, 1));
    assert_eq!(c, &expected)
}

#[test]
fn decimal_to_float() {
    let array = Int128Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None])
        .to(DataType::Decimal(2, 1));

    let b = cast(&array, &DataType::Float32, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<f32>>().unwrap();

    let expected = Float32Array::from(&[Some(0.2), Some(1.0), Some(-0.2), Some(-1.0), None]);
    assert_eq!(c, &expected)
}

#[test]
fn decimal_to_integer() {
    let array = Int128Array::from(&[Some(2), Some(10), Some(-2), Some(-10), None, Some(2560)])
        .to(DataType::Decimal(2, 1));

    let b = cast(&array, &DataType::Int8, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i8>>().unwrap();

    let expected = Int8Array::from(&[Some(0), Some(1), Some(0), Some(-1), None, None]);
    assert_eq!(c, &expected)
}

#[test]
fn utf8_to_i32_partial() {
    let array = Utf8Array::<i32>::from_slice(["5", "6", "seven", "8aa", "9.1aa"]);
    let b = cast(
        &array,
        &DataType::Int32,
        CastOptions {
            partial: true,
            ..Default::default()
        },
    )
    .unwrap();
    let c = b.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();

    let expected = &[Some(5), Some(6), Some(0), Some(8), Some(9)];
    let expected = Int32Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn bool_to_i32() {
    let array = BooleanArray::from(vec![Some(true), Some(false), None]);
    let b = cast(&array, &DataType::Int32, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = &[Some(1), Some(0), None];
    let expected = Int32Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn bool_to_f64() {
    let array = BooleanArray::from(vec![Some(true), Some(false), None]);
    let b = cast(&array, &DataType::Float64, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<Float64Array>().unwrap();

    let expected = &[Some(1.0), Some(0.0), None];
    let expected = Float64Array::from(expected);
    assert_eq!(c, &expected);
}

#[test]
fn bool_to_utf8() {
    let array = BooleanArray::from(vec![Some(true), Some(false), None]);
    let b = cast(&array, &DataType::Utf8, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();

    let expected = Utf8Array::<i32>::from([Some("1"), Some("0"), Some("0")]);
    assert_eq!(c, &expected);
}

#[test]
fn bool_to_binary() {
    let array = BooleanArray::from(vec![Some(true), Some(false), None]);
    let b = cast(&array, &DataType::Binary, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<BinaryArray<i32>>().unwrap();

    let expected = BinaryArray::<i32>::from([Some("1"), Some("0"), Some("0")]);
    assert_eq!(c, &expected);
}

#[test]
fn int32_to_timestamp() {
    let array = Int32Array::from(&[Some(2), Some(10), None]);
    assert!(cast(
        &array,
        &DataType::Timestamp(TimeUnit::Microsecond, None),
        CastOptions::default()
    )
    .is_err());
}

#[test]
fn consistency() {
    use DataType::*;
    let datatypes = vec![
        Null,
        Boolean,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float16,
        Float32,
        Float64,
        Timestamp(TimeUnit::Second, None),
        Timestamp(TimeUnit::Millisecond, None),
        Timestamp(TimeUnit::Millisecond, Some("+01:00".to_string())),
        Timestamp(TimeUnit::Microsecond, None),
        Timestamp(TimeUnit::Nanosecond, None),
        Time64(TimeUnit::Microsecond),
        Time64(TimeUnit::Nanosecond),
        Date32,
        Time32(TimeUnit::Second),
        Time32(TimeUnit::Millisecond),
        Decimal(1, 2),
        Decimal(2, 2),
        Date64,
        Utf8,
        LargeUtf8,
        Binary,
        LargeBinary,
        Duration(TimeUnit::Second),
        Duration(TimeUnit::Millisecond),
        Duration(TimeUnit::Microsecond),
        Duration(TimeUnit::Nanosecond),
        List(Box::new(Field::new("a", Utf8, true))),
        LargeList(Box::new(Field::new("a", Utf8, true))),
    ];
    for d1 in &datatypes {
        for d2 in &datatypes {
            let array = new_null_array(d1.clone(), 10);
            if can_cast_types(d1, d2) {
                let result = cast(array.as_ref(), d2, CastOptions::default());
                if let Ok(result) = result {
                    assert_eq!(result.data_type(), d2, "type not equal: {d1:?} {d2:?}");
                } else {
                    panic!("Cast should have not have failed {d1:?} {d2:?}: {result:?}");
                }
            } else if cast(array.as_ref(), d2, CastOptions::default()).is_ok() {
                panic!("Cast should have failed {d1:?} {d2:?}");
            }
        }
    }
}

fn test_primitive_to_primitive<I: NativeType, O: NativeType>(
    lhs: &[I],
    lhs_type: DataType,
    expected: &[O],
    expected_type: DataType,
) {
    let a = PrimitiveArray::<I>::from_slice(lhs).to(lhs_type);
    let b = cast(&a, &expected_type, CastOptions::default()).unwrap();
    let b = b.as_any().downcast_ref::<PrimitiveArray<O>>().unwrap();
    let expected = PrimitiveArray::<O>::from_slice(expected).to(expected_type);
    assert_eq!(b, &expected);
}

#[test]
fn date32_to_date64() {
    test_primitive_to_primitive(
        &[10000i32, 17890],
        DataType::Date32,
        &[864000000000i64, 1545696000000],
        DataType::Date64,
    );
}

#[test]
fn days_ms_to_months_days_ns() {
    test_primitive_to_primitive(
        &[days_ms::new(1, 1), days_ms::new(1, 2)],
        DataType::Interval(IntervalUnit::DayTime),
        &[
            months_days_ns::new(0, 1, 1000),
            months_days_ns::new(0, 1, 2000),
        ],
        DataType::Interval(IntervalUnit::MonthDayNano),
    );
}

#[test]
fn months_to_months_days_ns() {
    test_primitive_to_primitive(
        &[1, 2],
        DataType::Interval(IntervalUnit::YearMonth),
        &[months_days_ns::new(1, 0, 0), months_days_ns::new(2, 0, 0)],
        DataType::Interval(IntervalUnit::MonthDayNano),
    );
}

#[test]
fn date64_to_date32() {
    test_primitive_to_primitive(
        &[864000000005i64, 1545696000001],
        DataType::Date64,
        &[10000i32, 17890],
        DataType::Date32,
    );
}

#[test]
fn date32_to_int32() {
    test_primitive_to_primitive(
        &[10000i32, 17890],
        DataType::Date32,
        &[10000i32, 17890],
        DataType::Int32,
    );
}

#[test]
fn date64_to_int32() {
    test_primitive_to_primitive(
        &[10000i64, 17890],
        DataType::Date64,
        &[10000i32, 17890],
        DataType::Int32,
    );
}

#[test]
fn date32_to_int64() {
    test_primitive_to_primitive(
        &[10000i32, 17890],
        DataType::Date32,
        &[10000i64, 17890],
        DataType::Int64,
    );
}

#[test]
fn int32_to_date32() {
    test_primitive_to_primitive(
        &[10000i32, 17890],
        DataType::Int32,
        &[10000i32, 17890],
        DataType::Date32,
    );
}

#[test]
fn timestamp_to_date32() {
    test_primitive_to_primitive(
        &[864000000005i64, 1545696000001],
        DataType::Timestamp(TimeUnit::Millisecond, Some(String::from("UTC"))),
        &[10000i32, 17890],
        DataType::Date32,
    );
}

#[test]
fn timestamp_to_date64() {
    test_primitive_to_primitive(
        &[864000000005i64, 1545696000001],
        DataType::Timestamp(TimeUnit::Millisecond, Some(String::from("UTC"))),
        &[864000000005i64, 1545696000001i64],
        DataType::Date64,
    );
}

#[test]
fn timestamp_to_i64() {
    test_primitive_to_primitive(
        &[864000000005i64, 1545696000001],
        DataType::Timestamp(TimeUnit::Millisecond, Some(String::from("UTC"))),
        &[864000000005i64, 1545696000001i64],
        DataType::Int64,
    );
}

#[test]
fn timestamp_to_timestamp() {
    test_primitive_to_primitive(
        &[864000003005i64, 1545696002001],
        DataType::Timestamp(TimeUnit::Millisecond, None),
        &[864000003i64, 1545696002],
        DataType::Timestamp(TimeUnit::Second, None),
    );
}

#[test]
fn utf8_to_dict() {
    let array = Utf8Array::<i32>::from([Some("one"), None, Some("three"), Some("one")]);

    // Cast to a dictionary (same value type, Utf8)
    let cast_type = DataType::Dictionary(u8::KEY_TYPE, Box::new(DataType::Utf8), false);
    let result = cast(&array, &cast_type, CastOptions::default()).expect("cast failed");

    let mut expected = MutableDictionaryArray::<u8, MutableUtf8Array<i32>>::new();
    expected
        .try_extend([Some("one"), None, Some("three"), Some("one")])
        .unwrap();
    let expected: DictionaryArray<u8> = expected.into();
    assert_eq!(expected, result.as_ref());
}

#[test]
fn dict_to_utf8() {
    let mut array = MutableDictionaryArray::<u8, MutableUtf8Array<i32>>::new();
    array
        .try_extend([Some("one"), None, Some("three"), Some("one")])
        .unwrap();
    let array: DictionaryArray<u8> = array.into();

    let result = cast(&array, &DataType::Utf8, CastOptions::default()).expect("cast failed");

    let expected = Utf8Array::<i32>::from([Some("one"), None, Some("three"), Some("one")]);

    assert_eq!(expected, result.as_ref());
}

#[test]
fn i32_to_dict() {
    let array = Int32Array::from(&[Some(1), None, Some(3), Some(1)]);

    // Cast to a dictionary (same value type, Utf8)
    let cast_type = DataType::Dictionary(u8::KEY_TYPE, Box::new(DataType::Int32), false);
    let result = cast(&array, &cast_type, CastOptions::default()).expect("cast failed");

    let mut expected = MutableDictionaryArray::<u8, MutablePrimitiveArray<i32>>::new();
    expected
        .try_extend([Some(1), None, Some(3), Some(1)])
        .unwrap();
    let expected: DictionaryArray<u8> = expected.into();
    assert_eq!(expected, result.as_ref());
}

#[test]
fn list_to_list() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        None,
        Some(vec![Some(4), None, Some(6)]),
    ];

    let expected_data = data
        .iter()
        .map(|x| x.as_ref().map(|x| x.iter().map(|x| x.map(|x| x as u16))));

    let mut array = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    array.try_extend(data.clone()).unwrap();
    let array: ListArray<i32> = array.into();

    let mut expected = MutableListArray::<i32, MutablePrimitiveArray<u16>>::new();
    expected.try_extend(expected_data).unwrap();
    let expected: ListArray<i32> = expected.into();

    let result = cast(&array, expected.data_type(), CastOptions::default()).unwrap();
    assert_eq!(expected, result.as_ref());
}

#[test]
fn list_to_from_fixed_size_list() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        Some(vec![Some(4), Some(5), None]),
        Some(vec![Some(6), None, Some(7)]),
    ];

    let fixed_data = data
        .iter()
        .map(|x| x.as_ref().map(|x| x.iter().map(|x| x.map(|x| x as u16))));

    let mut list = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    list.try_extend(data.clone()).unwrap();
    let list: ListArray<i32> = list.into();

    let inner = MutablePrimitiveArray::<u16>::new();
    let mut fixed = MutableFixedSizeListArray::<MutablePrimitiveArray<u16>>::new(inner, 3);
    fixed.try_extend(fixed_data).unwrap();
    let fixed: FixedSizeListArray = fixed.into();

    let result = cast(&list, fixed.data_type(), CastOptions::default()).unwrap();
    assert_eq!(fixed, result.as_ref());

    let result = cast(&fixed, list.data_type(), CastOptions::default()).unwrap();
    assert_eq!(list, result.as_ref());
}

#[test]
fn timestamp_with_tz_to_utf8() {
    let tz = "-02:00".to_string();
    let expected =
        Utf8Array::<i32>::from_slice(["1996-12-19T16:39:57-02:00", "1996-12-19T17:39:57-02:00"]);
    let array = Int64Array::from_slice([851020797000000000, 851024397000000000])
        .to(DataType::Timestamp(TimeUnit::Nanosecond, Some(tz)));

    let result = cast(&array, expected.data_type(), CastOptions::default()).expect("cast failed");
    assert_eq!(expected, result.as_ref());
}

#[test]
fn utf8_to_timestamp_with_tz() {
    let tz = "-02:00".to_string();
    let array =
        Utf8Array::<i32>::from_slice(["1996-12-19T16:39:57-02:00", "1996-12-19T17:39:57-02:00"]);
    // the timezone is used to map the time to UTC.
    let expected = Int64Array::from_slice([851020797000000000, 851024397000000000])
        .to(DataType::Timestamp(TimeUnit::Nanosecond, Some(tz)));

    let result = cast(&array, expected.data_type(), CastOptions::default()).expect("cast failed");
    assert_eq!(expected, result.as_ref());
}

#[test]
fn utf8_to_naive_timestamp() {
    let array =
        Utf8Array::<i32>::from_slice(["1996-12-19T16:39:57-02:00", "1996-12-19T17:39:57-02:00"]);
    // the timezone is disregarded from the string and we assume UTC
    let expected = Int64Array::from_slice([851013597000000000, 851017197000000000])
        .to(DataType::Timestamp(TimeUnit::Nanosecond, None));

    let result = cast(&array, expected.data_type(), CastOptions::default()).expect("cast failed");
    assert_eq!(expected, result.as_ref());
}

#[test]
fn naive_timestamp_to_utf8() {
    let array = Int64Array::from_slice([851013597000000000, 851017197000000000])
        .to(DataType::Timestamp(TimeUnit::Nanosecond, None));

    let expected = Utf8Array::<i32>::from_slice(["1996-12-19 16:39:57", "1996-12-19 17:39:57"]);

    let result = cast(&array, expected.data_type(), CastOptions::default()).expect("cast failed");
    assert_eq!(expected, result.as_ref());
}

#[test]
fn null_array_from_and_to_others() {
    macro_rules! typed_test {
        ($ARR_TYPE:ident, $DATATYPE:ident) => {{
            {
                let array = new_null_array(DataType::Null, 6);
                let expected = $ARR_TYPE::from(vec![None; 6]);
                let cast_type = DataType::$DATATYPE;
                let result =
                    cast(array.as_ref(), &cast_type, CastOptions::default()).expect("cast failed");
                let result = result.as_any().downcast_ref::<$ARR_TYPE>().unwrap();
                assert_eq!(result.data_type(), &cast_type);
                assert_eq!(result, &expected);
            }
            {
                let array = $ARR_TYPE::from(vec![None; 4]);
                let expected = NullArray::new_null(DataType::Null, 4);
                let result =
                    cast(&array, &DataType::Null, CastOptions::default()).expect("cast failed");
                let result = result.as_any().downcast_ref::<NullArray>().unwrap();
                assert_eq!(result.data_type(), &DataType::Null);
                assert_eq!(result, &expected);
            }
        }};
    }

    typed_test!(Int16Array, Int16);
    typed_test!(Int32Array, Int32);
    typed_test!(Int64Array, Int64);

    typed_test!(UInt16Array, UInt16);
    typed_test!(UInt32Array, UInt32);
    typed_test!(UInt64Array, UInt64);

    typed_test!(Float16Array, Float16);
    typed_test!(Float32Array, Float32);
    typed_test!(Float64Array, Float64);
}

#[test]
fn utf8_to_date32() {
    let array = Utf8Array::<i32>::from_slice(["1970-01-01", "1970-01-02"]);
    let b = cast(&array, &DataType::Date32, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = Int32Array::from_slice([0, 1]).to(DataType::Date32);

    assert_eq!(&expected, c);
}

#[test]
fn utf8_to_date64() {
    let array = Utf8Array::<i32>::from_slice(["1970-01-01", "1970-01-02"]);
    let b = cast(&array, &DataType::Date64, CastOptions::default()).unwrap();
    let c = b.as_any().downcast_ref::<Int64Array>().unwrap();

    let expected = Int64Array::from_slice([0, 86400000]).to(DataType::Date64);

    assert_eq!(&expected, c);
}

#[test]
fn dict_keys() {
    let mut array = MutableDictionaryArray::<u8, MutableUtf8Array<i32>>::new();
    array
        .try_extend([Some("one"), None, Some("three"), Some("one")])
        .unwrap();
    let array: DictionaryArray<u8> = array.into();

    let result = cast(
        &array,
        &DataType::Dictionary(IntegerType::Int64, Box::new(DataType::Utf8), false),
        CastOptions::default(),
    )
    .expect("cast failed");

    let mut expected = MutableDictionaryArray::<i64, MutableUtf8Array<i32>>::new();
    expected
        .try_extend([Some("one"), None, Some("three"), Some("one")])
        .unwrap();
    let expected: DictionaryArray<i64> = expected.into();

    assert_eq!(expected, result.as_ref());
}

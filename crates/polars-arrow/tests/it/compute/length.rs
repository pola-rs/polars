use polars_arrow::array::*;
use polars_arrow::compute::length::*;
use polars_arrow::datatypes::*;
use polars_arrow::offset::Offset;

fn length_test_string<O: Offset>() {
    vec![
        (
            vec![Some("hello"), Some(" "), None],
            vec![Some(5usize), Some(1), None],
        ),
        (vec![Some("💖")], vec![Some(4)]),
    ]
    .into_iter()
    .for_each(|(input, expected)| {
        let array = Utf8Array::<O>::from(input);
        let result = length(&array).unwrap();

        let data_type = if O::IS_LARGE {
            DataType::Int64
        } else {
            DataType::Int32
        };

        let expected = expected
            .into_iter()
            .map(|x| x.map(|x| O::from_usize(x).unwrap()))
            .collect::<PrimitiveArray<O>>()
            .to(data_type);
        assert_eq!(expected, result.as_ref());
    })
}

#[test]
fn large_utf8() {
    length_test_string::<i64>()
}

#[test]
fn utf8() {
    length_test_string::<i32>()
}

#[test]
fn consistency() {
    use polars_arrow::datatypes::DataType::*;

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
        Float32,
        Float64,
        Timestamp(TimeUnit::Second, None),
        Timestamp(TimeUnit::Millisecond, None),
        Timestamp(TimeUnit::Microsecond, None),
        Timestamp(TimeUnit::Nanosecond, None),
        Time64(TimeUnit::Microsecond),
        Time64(TimeUnit::Nanosecond),
        Date32,
        Time32(TimeUnit::Second),
        Time32(TimeUnit::Millisecond),
        Date64,
        Utf8,
        LargeUtf8,
        Binary,
        LargeBinary,
        Duration(TimeUnit::Second),
        Duration(TimeUnit::Millisecond),
        Duration(TimeUnit::Microsecond),
        Duration(TimeUnit::Nanosecond),
    ];

    datatypes.into_iter().for_each(|d1| {
        let array = new_null_array(d1.clone(), 10);
        if can_length(&d1) {
            assert!(length(array.as_ref()).is_ok());
        } else {
            assert!(length(array.as_ref()).is_err());
        }
    });
}

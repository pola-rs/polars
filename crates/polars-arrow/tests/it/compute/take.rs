use polars_arrow::array::*;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::buffer::Buffer;
use polars_arrow::compute::take::{can_take, take};
use polars_arrow::datatypes::{DataType, Field, IntervalUnit};
use polars_arrow::error::Result;
use polars_arrow::types::NativeType;

fn test_take_primitive<T>(
    data: &[Option<T>],
    indices: &Int32Array,
    expected_data: &[Option<T>],
    data_type: DataType,
) -> Result<()>
where
    T: NativeType,
{
    let output = PrimitiveArray::<T>::from(data).to(data_type.clone());
    let expected = PrimitiveArray::<T>::from(expected_data).to(data_type);
    let output = take(&output, indices)?;
    assert_eq!(expected, output.as_ref());
    Ok(())
}

#[test]
fn test_take_primitive_non_null_indices() {
    let indices = Int32Array::from_slice([0, 5, 3, 1, 4, 2]);
    test_take_primitive::<i8>(
        &[None, Some(2), Some(4), Some(6), Some(8), None],
        &indices,
        &[None, None, Some(6), Some(2), Some(8), Some(4)],
        DataType::Int8,
    )
    .unwrap();

    test_take_primitive::<i8>(
        &[Some(0), Some(2), Some(4), Some(6), Some(8), Some(10)],
        &indices,
        &[Some(0), Some(10), Some(6), Some(2), Some(8), Some(4)],
        DataType::Int8,
    )
    .unwrap();
}

#[test]
fn test_take_primitive_null_values() {
    let indices = Int32Array::from(&[Some(0), None, Some(3), Some(1), Some(4), Some(2)]);
    test_take_primitive::<i8>(
        &[Some(0), Some(2), Some(4), Some(6), Some(8), Some(10)],
        &indices,
        &[Some(0), None, Some(6), Some(2), Some(8), Some(4)],
        DataType::Int8,
    )
    .unwrap();

    test_take_primitive::<i8>(
        &[None, Some(2), Some(4), Some(6), Some(8), Some(10)],
        &indices,
        &[None, None, Some(6), Some(2), Some(8), Some(4)],
        DataType::Int8,
    )
    .unwrap();
}

fn create_test_struct() -> StructArray {
    let boolean = BooleanArray::from_slice([true, false, false, true]);
    let int = Int32Array::from_slice([42, 28, 19, 31]);
    let validity = vec![true, true, false, true]
        .into_iter()
        .collect::<MutableBitmap>()
        .into();
    let fields = vec![
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Int32, true),
    ];
    StructArray::new(
        DataType::Struct(fields),
        vec![boolean.boxed(), int.boxed()],
        validity,
    )
}

#[test]
fn test_struct_with_nulls() {
    let array = create_test_struct();

    let indices = Int32Array::from(&[None, Some(3), Some(1), None, Some(0)]);

    let output = take(&array, &indices).unwrap();

    let boolean = BooleanArray::from(&[None, Some(true), Some(false), None, Some(true)]);
    let int = Int32Array::from(&[None, Some(31), Some(28), None, Some(42)]);
    let validity = vec![false, true, true, false, true]
        .into_iter()
        .collect::<MutableBitmap>()
        .into();
    let expected = StructArray::new(
        array.data_type().clone(),
        vec![boolean.boxed(), int.boxed()],
        validity,
    );
    assert_eq!(expected, output.as_ref());
}

#[test]
fn consistency() {
    use polars_arrow::array::new_null_array;
    use polars_arrow::datatypes::DataType::*;
    use polars_arrow::datatypes::TimeUnit;

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
        Interval(IntervalUnit::DayTime),
        Interval(IntervalUnit::YearMonth),
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
        let indices = Int32Array::from(&[Some(1), Some(2), None, Some(3)]);
        if can_take(&d1) {
            assert!(take(array.as_ref(), &indices).is_ok());
        } else {
            assert!(take(array.as_ref(), &indices).is_err());
        }
    });
}

#[test]
fn empty() {
    let indices = Int32Array::from_slice([]);
    let values = BooleanArray::from(vec![Some(true), Some(false)]);
    let a = take(&values, &indices).unwrap();
    assert_eq!(a.len(), 0)
}

#[test]
fn unsigned_take() {
    let indices = UInt32Array::from_slice([]);
    let values = BooleanArray::from(vec![Some(true), Some(false)]);
    let a = take(&values, &indices).unwrap();
    assert_eq!(a.len(), 0)
}

#[test]
fn list_with_no_none() {
    let values = Buffer::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let values = PrimitiveArray::<i32>::new(DataType::Int32, values, None);

    let data_type = ListArray::<i32>::default_datatype(DataType::Int32);
    let array = ListArray::<i32>::new(
        data_type,
        vec![0, 2, 2, 6, 9, 10].try_into().unwrap(),
        Box::new(values),
        None,
    );

    let indices = PrimitiveArray::from([Some(4i32), Some(1), Some(3)]);
    let result = take(&array, &indices).unwrap();

    let expected_values = Buffer::from(vec![9, 6, 7, 8]);
    let expected_values = PrimitiveArray::<i32>::new(DataType::Int32, expected_values, None);
    let expected_type = ListArray::<i32>::default_datatype(DataType::Int32);
    let expected = ListArray::<i32>::new(
        expected_type,
        vec![0, 1, 1, 4].try_into().unwrap(),
        Box::new(expected_values),
        None,
    );

    assert_eq!(expected, result.as_ref());
}

#[test]
fn list_with_none() {
    let values = Buffer::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let values = PrimitiveArray::<i32>::new(DataType::Int32, values, None);

    let validity_values = vec![true, false, true, true, true];
    let validity = Bitmap::from_trusted_len_iter(validity_values.into_iter());

    let data_type = ListArray::<i32>::default_datatype(DataType::Int32);
    let array = ListArray::<i32>::new(
        data_type,
        vec![0, 2, 2, 6, 9, 10].try_into().unwrap(),
        Box::new(values),
        Some(validity),
    );

    let indices = PrimitiveArray::from([Some(4i32), None, Some(2), Some(3)]);
    let result = take(&array, &indices).unwrap();

    let data_expected = vec![
        Some(vec![Some(9i32)]),
        None,
        Some(vec![Some(2i32), Some(3), Some(4), Some(5)]),
        Some(vec![Some(6i32), Some(7), Some(8)]),
    ];

    let mut expected = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    expected.try_extend(data_expected).unwrap();
    let expected: ListArray<i32> = expected.into();

    assert_eq!(expected, result.as_ref());
}

#[test]
fn list_both_validity() {
    let values = vec![
        Some(vec![Some(2i32), Some(3), Some(4), Some(5)]),
        None,
        Some(vec![Some(9i32)]),
        Some(vec![Some(6i32), Some(7), Some(8)]),
    ];

    let mut array = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    array.try_extend(values).unwrap();
    let array: ListArray<i32> = array.into();

    let indices = PrimitiveArray::from([Some(3i32), None, Some(1), Some(0)]);
    let result = take(&array, &indices).unwrap();

    let data_expected = vec![
        Some(vec![Some(6i32), Some(7), Some(8)]),
        None,
        None,
        Some(vec![Some(2i32), Some(3), Some(4), Some(5)]),
    ];
    let mut expected = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    expected.try_extend(data_expected).unwrap();
    let expected: ListArray<i32> = expected.into();

    assert_eq!(expected, result.as_ref());
}

#[test]
fn fixed_size_list_with_no_none() {
    let values = Buffer::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let values = PrimitiveArray::<i32>::new(DataType::Int32, values, None);

    let data_type = FixedSizeListArray::default_datatype(DataType::Int32, 2);
    let array = FixedSizeListArray::new(data_type, Box::new(values), None);

    let indices = PrimitiveArray::from([Some(4i32), Some(1), Some(3)]);
    let result = take(&array, &indices).unwrap();

    let expected_values = Buffer::from(vec![8, 9, 2, 3, 6, 7]);
    let expected_values = PrimitiveArray::<i32>::new(DataType::Int32, expected_values, None);
    let expected_type = FixedSizeListArray::default_datatype(DataType::Int32, 2);
    let expected = FixedSizeListArray::new(expected_type, Box::new(expected_values), None);

    assert_eq!(expected, result.as_ref());
}

#[test]
fn test_nested() {
    let values = Buffer::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let values = PrimitiveArray::<i32>::new(DataType::Int32, values, None);

    let data_type = ListArray::<i32>::default_datatype(DataType::Int32);
    let array = ListArray::<i32>::new(
        data_type,
        vec![0, 2, 4, 7, 7, 8, 10].try_into().unwrap(),
        Box::new(values),
        None,
    );

    let data_type = ListArray::<i32>::default_datatype(array.data_type().clone());
    let nested = ListArray::<i32>::new(
        data_type,
        vec![0, 2, 5, 6].try_into().unwrap(),
        Box::new(array),
        None,
    );

    let indices = PrimitiveArray::from([Some(0i32), Some(1)]);
    let result = take(&nested, &indices).unwrap();

    // expected data
    let expected_values = Buffer::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let expected_values = PrimitiveArray::<i32>::new(DataType::Int32, expected_values, None);

    let expected_data_type = ListArray::<i32>::default_datatype(DataType::Int32);
    let expected_array = ListArray::<i32>::new(
        expected_data_type,
        vec![0, 2, 4, 7, 7, 8].try_into().unwrap(),
        Box::new(expected_values),
        None,
    );

    let expected_data_type = ListArray::<i32>::default_datatype(expected_array.data_type().clone());
    let expected = ListArray::<i32>::new(
        expected_data_type,
        vec![0, 2, 5].try_into().unwrap(),
        Box::new(expected_array),
        None,
    );

    assert_eq!(expected, result.as_ref());
}

use polars_arrow::array::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::filter::*;

#[test]
fn array_slice() {
    let a = Int32Array::from_slice([5, 6, 7, 8, 9]).sliced(1, 4);
    let b = BooleanArray::from_slice(vec![true, true, false, false, true]).sliced(1, 4);
    let c = filter(&a, &b).unwrap();

    let expected = Int32Array::from_slice([6, 9]);

    assert_eq!(expected, c.as_ref());
}

#[test]
fn array_large_filter_chunks() {
    let len = 65usize;
    let a = Int32Array::from_iter((0..(len as i32)).map(Some));

    let init = vec![true, true, true, false, false, true];
    let remaining = len - init.len();
    let iter = init
        .into_iter()
        .chain(std::iter::repeat(false).take(remaining))
        .map(Some);
    let b = BooleanArray::from_iter(iter);

    let c = filter(&a, &b).unwrap();

    let expected = Int32Array::from_slice([0, 1, 2, 5]);

    assert_eq!(expected, c.as_ref());
}

#[test]
fn array_low_density() {
    // this test exercises the all 0's branch of the filter algorithm
    let mut data_values = (1..=65).collect::<Vec<i32>>();
    let mut filter_values = (1..=65).map(|i| matches!(i % 65, 0)).collect::<Vec<bool>>();
    // set up two more values after the batch
    data_values.extend_from_slice(&[66, 67]);
    filter_values.extend_from_slice(&[false, true]);
    let a = Int32Array::from_slice(&data_values);
    let b = BooleanArray::from_slice(filter_values);
    let c = filter(&a, &b).unwrap();

    let expected = Int32Array::from_slice([65, 67]);

    assert_eq!(expected, c.as_ref());
}

#[test]
fn array_high_density() {
    // this test exercises the all 1's branch of the filter algorithm
    let mut data_values = (1..=65).map(Some).collect::<Vec<_>>();
    let mut filter_values = (1..=65)
        .map(|i| !matches!(i % 65, 0))
        .collect::<Vec<bool>>();
    // set second data value to null
    data_values[1] = None;
    // set up two more values after the batch
    data_values.extend_from_slice(&[Some(66), None, Some(67), None]);
    filter_values.extend_from_slice(&[false, true, true, true]);
    let a = Int32Array::from(data_values);
    let b = BooleanArray::from_slice(filter_values);
    let c = filter(&a, &b).unwrap();
    let d = c.as_ref().as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(67, d.len());
    assert_eq!(3, d.null_count());
    assert_eq!(1, d.value(0));
    assert!(d.is_null(1));
    assert_eq!(64, d.value(63));
    assert!(d.is_null(64));
    assert_eq!(67, d.value(65));
}

#[test]
fn string_array_simple() {
    let a = Utf8Array::<i32>::from_slice(["hello", " ", "world", "!"]);
    let b = BooleanArray::from_slice([true, false, true, false]);
    let c = filter(&a, &b).unwrap();
    let d = c
        .as_ref()
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap();
    assert_eq!(2, d.len());
    assert_eq!("hello", d.value(0));
    assert_eq!("world", d.value(1));
}

#[test]
fn primitive_array_with_null() {
    let a = Int32Array::from(&[Some(5), None]);
    let b = BooleanArray::from_slice(vec![false, true]);
    let c = filter(&a, &b).unwrap();
    let d = c.as_ref().as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(1, d.len());
    assert!(d.is_null(0));
}

#[test]
fn string_array_with_null() {
    let a = Utf8Array::<i32>::from([Some("hello"), None, Some("world"), None]);
    let b = BooleanArray::from_slice([true, false, false, true]);
    let c = filter(&a, &b).unwrap();
    let d = c
        .as_ref()
        .as_any()
        .downcast_ref::<Utf8Array<i32>>()
        .unwrap();
    assert_eq!(2, d.len());
    assert_eq!("hello", d.value(0));
    assert!(!d.is_null(0));
    assert!(d.is_null(1));
}

#[test]
fn binary_array_with_null() {
    let data: Vec<Option<&[u8]>> = vec![Some(b"hello"), None, Some(b"world"), None];
    let a = BinaryArray::<i32>::from(data);
    let b = BooleanArray::from_slice(vec![true, false, false, true]);
    let c = filter(&a, &b).unwrap();
    let d = c
        .as_ref()
        .as_any()
        .downcast_ref::<BinaryArray<i32>>()
        .unwrap();
    assert_eq!(2, d.len());
    assert_eq!(b"hello", d.value(0));
    assert!(!d.is_null(0));
    assert!(d.is_null(1));
}

#[test]
fn masked_true_values() {
    let a = Int32Array::from_slice([1, 2, 3]);
    let b = BooleanArray::from_slice([true, false, true]);
    let validity = Bitmap::from(&[true, false, false]);

    let b = b.with_validity(Some(validity));

    let c = filter(&a, &b).unwrap();

    let expected = Int32Array::from_slice([1]);

    assert_eq!(expected, c.as_ref());
}

/*
#[test]
fn dictionary_array() {
    let values = vec![Some("hello"), None, Some("world"), Some("!")];
    let a: Int8DictionaryArray = values.iter().copied().collect();
    let b = BooleanArray::from(vec![false, true, true, false]);
    let c = filter(&a, &b).unwrap();
    let d = c
        .as_ref()
        .as_any()
        .downcast_ref::<Int8DictionaryArray>()
        .unwrap();
    let value_array = d.values();
    let values = value_array.as_any().downcast_ref::<StringArray>().unwrap();
    // values are cloned in the filtered dictionary array
    assert_eq!(3, values.len());
    // but keys are filtered
    assert_eq!(2, d.len());
    assert_eq!(true, d.is_null(0));
    assert_eq!("world", values.value(d.keys().value(1) as usize));
}

#[test]
fn list_array() {
    let value_data = ArrayData::builder(DataType::Int32)
        .len(8)
        .add_buffer(Buffer::from_slice_ref(&[0, 1, 2, 3, 4, 5, 6, 7]))
        .build();

    let value_offsets = Buffer::from_slice_ref(&[0i64, 3, 6, 8, 8]);

    let list_data_type =
        DataType::LargeList(Box::new(Field::new("item", DataType::Int32, false)));
    let list_data = ArrayData::builder(list_data_type)
        .len(4)
        .add_buffer(value_offsets)
        .add_child_data(value_data)
        .null_bit_buffer(Buffer::from([0b00000111]))
        .build();

    //  a = [[0, 1, 2], [3, 4, 5], [6, 7], null]
    let a = LargeListArray::from(list_data);
    let b = BooleanArray::from(vec![false, true, false, true]);
    let result = filter(&a, &b).unwrap();

    // expected: [[3, 4, 5], null]
    let value_data = ArrayData::builder(DataType::Int32)
        .len(3)
        .add_buffer(Buffer::from_slice_ref(&[3, 4, 5]))
        .build();

    let value_offsets = Buffer::from_slice_ref(&[0i64, 3, 3]);

    let list_data_type =
        DataType::LargeList(Box::new(Field::new("item", DataType::Int32, false)));
    let expected = ArrayData::builder(list_data_type)
        .len(2)
        .add_buffer(value_offsets)
        .add_child_data(value_data)
        .null_bit_buffer(Buffer::from([0b00000001]))
        .build();

    assert_eq!(&make_array(expected), &result);
}
*/

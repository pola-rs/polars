use arrow::array::*;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;

mod mutable;

#[test]
fn debug() {
    let values = Buffer::from(vec![1, 2, 3, 4, 5]);
    let values = PrimitiveArray::<i32>::new(ArrowDataType::Int32, values, None);

    let data_type = ListArray::<i32>::default_datatype(ArrowDataType::Int32);
    let array = ListArray::<i32>::new(
        data_type,
        vec![0, 2, 2, 3, 5].try_into().unwrap(),
        Box::new(values),
        None,
    );

    assert_eq!(format!("{array:?}"), "ListArray[[1, 2], [], [3], [4, 5]]");
}

#[test]
fn split_at() {
    let values = Buffer::from(vec![1, 2, 3, 4, 5]);
    let values = PrimitiveArray::<i32>::new(ArrowDataType::Int32, values, None);

    let data_type = ListArray::<i32>::default_datatype(ArrowDataType::Int32);
    let array = ListArray::<i32>::new(
        data_type,
        vec![0, 2, 2, 3, 5].try_into().unwrap(),
        Box::new(values),
        None,
    );

    let (lhs, rhs) = array.split_at(2);

    assert_eq!(format!("{lhs:?}"), "ListArray[[1, 2], []]");
    assert_eq!(format!("{rhs:?}"), "ListArray[[3], [4, 5]]");
}

#[test]
#[should_panic]
fn test_nested_panic() {
    let values = Buffer::from(vec![1, 2, 3, 4, 5]);
    let values = PrimitiveArray::<i32>::new(ArrowDataType::Int32, values, None);

    let data_type = ListArray::<i32>::default_datatype(ArrowDataType::Int32);
    let array = ListArray::<i32>::new(
        data_type.clone(),
        vec![0, 2, 2, 3, 5].try_into().unwrap(),
        Box::new(values),
        None,
    );

    // The datatype for the nested array has to be created considering
    // the nested structure of the child data
    let _ = ListArray::<i32>::new(
        data_type,
        vec![0, 2, 4].try_into().unwrap(),
        Box::new(array),
        None,
    );
}

#[test]
fn test_nested_display() {
    let values = Buffer::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let values = PrimitiveArray::<i32>::new(ArrowDataType::Int32, values, None);

    let data_type = ListArray::<i32>::default_datatype(ArrowDataType::Int32);
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

    let expected = "ListArray[[[1, 2], [3, 4]], [[5, 6, 7], [], [8]], [[9, 10]]]";
    assert_eq!(format!("{nested:?}"), expected);
}

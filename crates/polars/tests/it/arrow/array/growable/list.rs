use arrow::array::growable::{Growable, GrowableList};
use arrow::array::{Array, ListArray, MutableListArray, MutablePrimitiveArray, TryExtend};
use arrow::datatypes::ArrowDataType;

fn create_list_array(data: Vec<Option<Vec<Option<i32>>>>) -> ListArray<i32> {
    let mut array = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    array.try_extend(data).unwrap();
    array.into()
}

#[test]
fn extension() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        Some(vec![Some(4), Some(5)]),
        Some(vec![Some(6i32), Some(7), Some(8)]),
    ];

    let array = create_list_array(data);

    let data_type =
        ArrowDataType::Extension("ext".to_owned(), Box::new(array.data_type().clone()), None);
    let array_ext = ListArray::new(
        data_type,
        array.offsets().clone(),
        array.values().clone(),
        array.validity().cloned(),
    );

    let mut a = GrowableList::new(vec![&array_ext], false, 0);
    unsafe {
        a.extend(0, 0, 1);
    }
    assert_eq!(a.len(), 1);

    let result: ListArray<i32> = a.into();
    assert_eq!(array_ext.data_type(), result.data_type());
}

#[test]
fn basic() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        Some(vec![Some(4), Some(5)]),
        Some(vec![Some(6i32), Some(7), Some(8)]),
    ];

    let array = create_list_array(data);

    let mut a = GrowableList::new(vec![&array], false, 0);
    unsafe {
        a.extend(0, 0, 1);
    }
    assert_eq!(a.len(), 1);

    let result: ListArray<i32> = a.into();

    let expected = vec![Some(vec![Some(1i32), Some(2), Some(3)])];
    let expected = create_list_array(expected);

    assert_eq!(result, expected)
}

#[test]
fn null_offset() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        None,
        Some(vec![Some(6i32), Some(7), Some(8)]),
    ];
    let array = create_list_array(data);
    let array = array.sliced(1, 2);

    let mut a = GrowableList::new(vec![&array], false, 0);
    unsafe {
        a.extend(0, 1, 1);
    }
    assert_eq!(a.len(), 1);

    let result: ListArray<i32> = a.into();

    let expected = vec![Some(vec![Some(6i32), Some(7), Some(8)])];
    let expected = create_list_array(expected);

    assert_eq!(result, expected)
}

#[test]
fn null_offsets() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        None,
        Some(vec![Some(6i32), None, Some(8)]),
    ];
    let array = create_list_array(data);
    let array = array.sliced(1, 2);

    let mut a = GrowableList::new(vec![&array], false, 0);
    unsafe {
        a.extend(0, 1, 1);
    }
    assert_eq!(a.len(), 1);

    let result: ListArray<i32> = a.into();

    let expected = vec![Some(vec![Some(6i32), None, Some(8)])];
    let expected = create_list_array(expected);

    assert_eq!(result, expected)
}

#[test]
fn test_from_two_lists() {
    let data_1 = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        None,
        Some(vec![Some(6i32), None, Some(8)]),
    ];
    let array_1 = create_list_array(data_1);

    let data_2 = vec![
        Some(vec![Some(8i32), Some(7), Some(6)]),
        Some(vec![Some(5i32), None, Some(4)]),
        Some(vec![Some(2i32), Some(1), Some(0)]),
    ];
    let array_2 = create_list_array(data_2);

    let mut a = GrowableList::new(vec![&array_1, &array_2], false, 6);
    unsafe {
        a.extend(0, 0, 2);
    }
    unsafe {
        a.extend(1, 1, 1);
    }
    assert_eq!(a.len(), 3);

    let result: ListArray<i32> = a.into();

    let expected = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        None,
        Some(vec![Some(5i32), None, Some(4)]),
    ];
    let expected = create_list_array(expected);

    assert_eq!(result, expected);
}

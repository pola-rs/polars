use arrow::array::{Int32Array, ListArray, MutableListArray, MutablePrimitiveArray, TryExtend};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

use super::test_equal;

fn create_list_array<U: AsRef<[i32]>, T: AsRef<[Option<U>]>>(data: T) -> ListArray<i32> {
    let iter = data.as_ref().iter().map(|x| {
        x.as_ref()
            .map(|x| x.as_ref().iter().map(|x| Some(*x)).collect::<Vec<_>>())
    });
    let mut array = MutableListArray::<i32, MutablePrimitiveArray<i32>>::new();
    array.try_extend(iter).unwrap();
    array.into()
}

#[test]
fn test_list_equal() {
    let a = create_list_array([Some(&[1, 2, 3]), Some(&[4, 5, 6])]);
    let b = create_list_array([Some(&[1, 2, 3]), Some(&[4, 5, 6])]);
    test_equal(&a, &b, true);

    let b = create_list_array([Some(&[1, 2, 3]), Some(&[4, 5, 7])]);
    test_equal(&a, &b, false);
}

// Test the case where null_count > 0
#[test]
fn test_list_null() {
    let a = create_list_array([Some(&[1, 2]), None, None, Some(&[3, 4]), None, None]);
    let b = create_list_array([Some(&[1, 2]), None, None, Some(&[3, 4]), None, None]);
    test_equal(&a, &b, true);

    let b = create_list_array([
        Some(&[1, 2]),
        None,
        Some(&[5, 6]),
        Some(&[3, 4]),
        None,
        None,
    ]);
    test_equal(&a, &b, false);

    let b = create_list_array([Some(&[1, 2]), None, None, Some(&[3, 5]), None, None]);
    test_equal(&a, &b, false);
}

// Test the case where offset != 0
#[test]
fn test_list_offsets() {
    let a = create_list_array([Some(&[1, 2]), None, None, Some(&[3, 4]), None, None]);
    let b = create_list_array([Some(&[1, 2]), None, None, Some(&[3, 5]), None, None]);

    let a_slice = a.clone().sliced(0, 3);
    let b_slice = b.clone().sliced(0, 3);
    test_equal(&a_slice, &b_slice, true);

    let a_slice = a.clone().sliced(0, 5);
    let b_slice = b.clone().sliced(0, 5);
    test_equal(&a_slice, &b_slice, false);

    let a_slice = a.sliced(4, 1);
    let b_slice = b.sliced(4, 1);
    test_equal(&a_slice, &b_slice, true);
}

#[test]
fn test_bla() {
    let offsets = vec![0, 3, 3, 6].try_into().unwrap();
    let data_type = ListArray::<i32>::default_datatype(ArrowDataType::Int32);
    let values = Box::new(Int32Array::from([
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        None,
        Some(6),
    ]));
    let validity = Bitmap::from([true, false, true]);
    let lhs = ListArray::<i32>::new(data_type, offsets, values, Some(validity));
    let lhs = lhs.sliced(1, 2);

    let offsets = vec![0, 0, 3].try_into().unwrap();
    let data_type = ListArray::<i32>::default_datatype(ArrowDataType::Int32);
    let values = Box::new(Int32Array::from([Some(4), None, Some(6)]));
    let validity = Bitmap::from([false, true]);
    let rhs = ListArray::<i32>::new(data_type, offsets, values, Some(validity));

    assert_eq!(lhs, rhs);
}

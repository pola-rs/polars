use arrow::array::{
    FixedSizeListArray, MutableFixedSizeListArray, MutablePrimitiveArray, TryExtend,
};

use super::test_equal;

/// Create a fixed size list of 2 value lengths
fn create_fixed_size_list_array<U: AsRef<[i32]>, T: AsRef<[Option<U>]>>(
    data: T,
) -> FixedSizeListArray {
    let data = data.as_ref().iter().map(|x| {
        Some(match x {
            Some(x) => x.as_ref().iter().map(|x| Some(*x)).collect::<Vec<_>>(),
            None => std::iter::repeat(None).take(3).collect::<Vec<_>>(),
        })
    });

    let mut list = MutableFixedSizeListArray::new(MutablePrimitiveArray::<i32>::new(), 3);
    list.try_extend(data).unwrap();
    list.into()
}

#[test]
fn test_fixed_size_list_equal() {
    let a = create_fixed_size_list_array([Some(&[1, 2, 3]), Some(&[4, 5, 6])]);
    let b = create_fixed_size_list_array([Some(&[1, 2, 3]), Some(&[4, 5, 6])]);
    test_equal(&a, &b, true);

    let b = create_fixed_size_list_array([Some(&[1, 2, 3]), Some(&[4, 5, 7])]);
    test_equal(&a, &b, false);
}

// Test the case where null_count > 0
#[test]
fn test_fixed_list_null() {
    let a =
        create_fixed_size_list_array([Some(&[1, 2, 3]), None, None, Some(&[4, 5, 6]), None, None]);
    /*
    let b = create_fixed_size_list_array(&[
        Some(&[1, 2, 3]),
        None,
        None,
        Some(&[4, 5, 6]),
        None,
        None,
    ]);
    test_equal(&a, &b, true);

    let b = create_fixed_size_list_array(&[
        Some(&[1, 2, 3]),
        None,
        Some(&[7, 8, 9]),
        Some(&[4, 5, 6]),
        None,
        None,
    ]);
    test_equal(&a, &b, false);
        */

    let b =
        create_fixed_size_list_array([Some(&[1, 2, 3]), None, None, Some(&[3, 6, 9]), None, None]);
    test_equal(&a, &b, false);
}

#[test]
fn test_fixed_list_offsets() {
    // Test the case where offset != 0
    let a =
        create_fixed_size_list_array([Some(&[1, 2, 3]), None, None, Some(&[4, 5, 6]), None, None]);
    let b =
        create_fixed_size_list_array([Some(&[1, 2, 3]), None, None, Some(&[3, 6, 9]), None, None]);

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

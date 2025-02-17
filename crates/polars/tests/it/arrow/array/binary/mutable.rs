use std::ops::Deref;

use arrow::array::{BinaryArray, MutableArray, MutableBinaryArray, TryExtendFromSelf};
use arrow::bitmap::Bitmap;
use polars_error::PolarsError;

#[test]
fn new() {
    assert_eq!(MutableBinaryArray::<i32>::new().len(), 0);

    let a = MutableBinaryArray::<i32>::with_capacity(2);
    assert_eq!(a.len(), 0);
    assert!(a.offsets().capacity() >= 2);
    assert_eq!(a.values().capacity(), 0);

    let a = MutableBinaryArray::<i32>::with_capacities(2, 60);
    assert_eq!(a.len(), 0);
    assert!(a.offsets().capacity() >= 2);
    assert!(a.values().capacity() >= 60);
}

#[test]
fn from_iter() {
    let iter = (0..3u8).map(|x| Some(vec![x; x as usize]));
    let a: MutableBinaryArray<i32> = iter.clone().collect();
    assert_eq!(a.values().deref(), &[1u8, 2, 2]);
    assert_eq!(a.offsets().as_slice(), &[0, 0, 1, 3]);
    assert_eq!(a.validity(), None);

    let a = unsafe { MutableBinaryArray::<i32>::from_trusted_len_iter_unchecked(iter) };
    assert_eq!(a.values().deref(), &[1u8, 2, 2]);
    assert_eq!(a.offsets().as_slice(), &[0, 0, 1, 3]);
    assert_eq!(a.validity(), None);
}

#[test]
fn from_trusted_len_iter() {
    let data = [vec![0; 0], vec![1; 1], vec![2; 2]];
    let a: MutableBinaryArray<i32> = data.iter().cloned().map(Some).collect();
    assert_eq!(a.values().deref(), &[1u8, 2, 2]);
    assert_eq!(a.offsets().as_slice(), &[0, 0, 1, 3]);
    assert_eq!(a.validity(), None);

    let a = MutableBinaryArray::<i32>::from_trusted_len_iter(data.iter().cloned().map(Some));
    assert_eq!(a.values().deref(), &[1u8, 2, 2]);
    assert_eq!(a.offsets().as_slice(), &[0, 0, 1, 3]);
    assert_eq!(a.validity(), None);

    let a = MutableBinaryArray::<i32>::try_from_trusted_len_iter::<PolarsError, _, _>(
        data.iter().cloned().map(Some).map(Ok),
    )
    .unwrap();
    assert_eq!(a.values().deref(), &[1u8, 2, 2]);
    assert_eq!(a.offsets().as_slice(), &[0, 0, 1, 3]);
    assert_eq!(a.validity(), None);

    let a = MutableBinaryArray::<i32>::from_trusted_len_values_iter(data.iter().cloned());
    assert_eq!(a.values().deref(), &[1u8, 2, 2]);
    assert_eq!(a.offsets().as_slice(), &[0, 0, 1, 3]);
    assert_eq!(a.validity(), None);
}

#[test]
fn push_null() {
    let mut array = MutableBinaryArray::<i32>::new();
    array.push::<&str>(None);

    let array: BinaryArray<i32> = array.into();
    assert_eq!(array.validity(), Some(&Bitmap::from([false])));
}

#[test]
fn pop() {
    let mut a = MutableBinaryArray::<i32>::new();
    a.push(Some(b"first"));
    a.push(Some(b"second"));
    a.push::<Vec<u8>>(None);
    a.push_null();

    assert_eq!(a.pop(), None);
    assert_eq!(a.len(), 3);
    assert_eq!(a.pop(), None);
    assert_eq!(a.len(), 2);
    assert_eq!(a.pop(), Some(b"second".to_vec()));
    assert_eq!(a.len(), 1);
    assert_eq!(a.pop(), Some(b"first".to_vec()));
    assert_eq!(a.len(), 0);
    assert_eq!(a.pop(), None);
    assert_eq!(a.len(), 0);
}

#[test]
fn pop_all_some() {
    let mut a = MutableBinaryArray::<i32>::new();
    a.push(Some(b"first"));
    a.push(Some(b"second"));
    a.push(Some(b"third"));
    a.push(Some(b"fourth"));

    for _ in 0..4 {
        a.push(Some(b"aaaa"));
    }

    a.push(Some(b"bbbb"));

    assert_eq!(a.pop(), Some(b"bbbb".to_vec()));
    assert_eq!(a.pop(), Some(b"aaaa".to_vec()));
    assert_eq!(a.pop(), Some(b"aaaa".to_vec()));
    assert_eq!(a.pop(), Some(b"aaaa".to_vec()));
    assert_eq!(a.len(), 5);
    assert_eq!(a.pop(), Some(b"aaaa".to_vec()));
    assert_eq!(a.pop(), Some(b"fourth".to_vec()));
    assert_eq!(a.pop(), Some(b"third".to_vec()));
    assert_eq!(a.pop(), Some(b"second".to_vec()));
    assert_eq!(a.pop(), Some(b"first".to_vec()));
    assert!(a.is_empty());
    assert_eq!(a.pop(), None);
}

#[test]
fn extend_trusted_len_values() {
    let mut array = MutableBinaryArray::<i32>::new();

    array.extend_trusted_len_values(vec![b"first".to_vec(), b"second".to_vec()].into_iter());
    array.extend_trusted_len_values(vec![b"third".to_vec()].into_iter());
    array.extend_trusted_len(vec![None, Some(b"fourth".to_vec())].into_iter());

    let array: BinaryArray<i32> = array.into();

    assert_eq!(array.values().as_slice(), b"firstsecondthirdfourth");
    assert_eq!(array.offsets().as_slice(), &[0, 5, 11, 16, 16, 22]);
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00010111], 5))
    );
}

#[test]
fn extend_trusted_len() {
    let mut array = MutableBinaryArray::<i32>::new();

    array.extend_trusted_len(vec![Some(b"first".to_vec()), Some(b"second".to_vec())].into_iter());
    array.extend_trusted_len(vec![None, Some(b"third".to_vec())].into_iter());

    let array: BinaryArray<i32> = array.into();

    assert_eq!(array.values().as_slice(), b"firstsecondthird");
    assert_eq!(array.offsets().as_slice(), &[0, 5, 11, 11, 16]);
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00001011], 4))
    );
}

#[test]
fn extend_from_self() {
    let mut a = MutableBinaryArray::<i32>::from([Some(b"aa"), None]);

    a.try_extend_from_self(&a.clone()).unwrap();

    assert_eq!(
        a,
        MutableBinaryArray::<i32>::from([Some(b"aa"), None, Some(b"aa"), None])
    );
}

#[test]
fn test_set_validity() {
    let mut array = MutableBinaryArray::<i32>::new();
    array.push(Some(b"first"));
    array.push(Some(b"second"));
    array.push(Some(b"third"));
    array.set_validity(Some([false, false, true].into()));

    assert!(!array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(array.is_valid(2));
}

#[test]
fn test_apply_validity() {
    let mut array = MutableBinaryArray::<i32>::new();
    array.push(Some(b"first"));
    array.push(Some(b"second"));
    array.push(Some(b"third"));
    array.set_validity(Some([true, true, true].into()));

    array.apply_validity(|mut mut_bitmap| {
        mut_bitmap.set(1, false);
        mut_bitmap.set(2, false);
        mut_bitmap
    });

    assert!(array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(!array.is_valid(2));
}

#[test]
fn test_apply_validity_with_no_validity_inited() {
    let mut array = MutableBinaryArray::<i32>::new();
    array.push(Some(b"first"));
    array.push(Some(b"second"));
    array.push(Some(b"third"));

    array.apply_validity(|mut mut_bitmap| {
        mut_bitmap.set(1, false);
        mut_bitmap.set(2, false);
        mut_bitmap
    });

    assert!(array.is_valid(0));
    assert!(array.is_valid(1));
    assert!(array.is_valid(2));
}

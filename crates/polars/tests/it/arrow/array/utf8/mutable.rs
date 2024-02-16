use arrow::array::{MutableArray, MutableUtf8Array, TryExtendFromSelf, Utf8Array};
use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowDataType;

#[test]
fn capacities() {
    let b = MutableUtf8Array::<i32>::with_capacities(1, 10);

    assert!(b.values().capacity() >= 10);
    assert!(b.offsets().capacity() >= 1);
}

#[test]
fn push_null() {
    let mut array = MutableUtf8Array::<i32>::new();
    array.push::<&str>(None);

    let array: Utf8Array<i32> = array.into();
    assert_eq!(array.validity(), Some(&Bitmap::from([false])));
}

#[test]
fn pop() {
    let mut a = MutableUtf8Array::<i32>::new();
    a.push(Some("first"));
    a.push(Some("second"));
    a.push(Some("third"));
    a.push::<&str>(None);

    assert_eq!(a.pop(), None);
    assert_eq!(a.len(), 3);
    assert_eq!(a.pop(), Some("third".to_owned()));
    assert_eq!(a.len(), 2);
    assert_eq!(a.pop(), Some("second".to_string()));
    assert_eq!(a.len(), 1);
    assert_eq!(a.pop(), Some("first".to_string()));
    assert!(a.is_empty());
    assert_eq!(a.pop(), None);
    assert!(a.is_empty());
}

#[test]
fn pop_all_some() {
    let mut a = MutableUtf8Array::<i32>::new();
    a.push(Some("first"));
    a.push(Some("second"));
    a.push(Some("third"));
    a.push(Some("fourth"));
    for _ in 0..4 {
        a.push(Some("aaaa"));
    }
    a.push(Some("こんにちは"));

    assert_eq!(a.pop(), Some("こんにちは".to_string()));
    assert_eq!(a.pop(), Some("aaaa".to_string()));
    assert_eq!(a.pop(), Some("aaaa".to_string()));
    assert_eq!(a.pop(), Some("aaaa".to_string()));
    assert_eq!(a.len(), 5);
    assert_eq!(a.pop(), Some("aaaa".to_string()));
    assert_eq!(a.pop(), Some("fourth".to_string()));
    assert_eq!(a.pop(), Some("third".to_string()));
    assert_eq!(a.pop(), Some("second".to_string()));
    assert_eq!(a.pop(), Some("first".to_string()));
    assert!(a.is_empty());
    assert_eq!(a.pop(), None);
}

/// Safety guarantee
#[test]
fn not_utf8() {
    let offsets = vec![0, 4].try_into().unwrap();
    let values = vec![0, 159, 146, 150]; // invalid utf8
    assert!(MutableUtf8Array::<i32>::try_new(ArrowDataType::Utf8, offsets, values, None).is_err());
}

#[test]
fn wrong_data_type() {
    let offsets = vec![0, 4].try_into().unwrap();
    let values = vec![1, 2, 3, 4];
    assert!(MutableUtf8Array::<i32>::try_new(ArrowDataType::Int8, offsets, values, None).is_err());
}

#[test]
fn test_extend_trusted_len_values() {
    let mut array = MutableUtf8Array::<i32>::new();

    array.extend_trusted_len_values(["hi", "there"].iter());
    array.extend_trusted_len_values(["hello"].iter());
    array.extend_trusted_len(vec![Some("again"), None].into_iter());

    let array: Utf8Array<i32> = array.into();

    assert_eq!(array.values().as_slice(), b"hitherehelloagain");
    assert_eq!(array.offsets().as_slice(), &[0, 2, 7, 12, 17, 17]);
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00001111], 5))
    );
}

#[test]
fn test_extend_trusted_len() {
    let mut array = MutableUtf8Array::<i32>::new();

    array.extend_trusted_len(vec![Some("hi"), Some("there")].into_iter());
    array.extend_trusted_len(vec![None, Some("hello")].into_iter());
    array.extend_trusted_len_values(["again"].iter());

    let array: Utf8Array<i32> = array.into();

    assert_eq!(array.values().as_slice(), b"hitherehelloagain");
    assert_eq!(array.offsets().as_slice(), &[0, 2, 7, 7, 12, 17]);
    assert_eq!(
        array.validity(),
        Some(&Bitmap::from_u8_slice([0b00011011], 5))
    );
}

#[test]
fn test_extend_values() {
    let mut array = MutableUtf8Array::<i32>::new();

    array.extend_values([Some("hi"), None, Some("there"), None].iter().flatten());
    array.extend_values([Some("hello"), None].iter().flatten());
    array.extend_values(vec![Some("again"), None].into_iter().flatten());

    let array: Utf8Array<i32> = array.into();

    assert_eq!(array.values().as_slice(), b"hitherehelloagain");
    assert_eq!(array.offsets().as_slice(), &[0, 2, 7, 12, 17]);
    assert_eq!(array.validity(), None,);
}

#[test]
fn test_extend() {
    let mut array = MutableUtf8Array::<i32>::new();

    array.extend([Some("hi"), None, Some("there"), None]);

    let array: Utf8Array<i32> = array.into();

    assert_eq!(
        array,
        Utf8Array::<i32>::from([Some("hi"), None, Some("there"), None])
    );
}

#[test]
fn as_arc() {
    let mut array = MutableUtf8Array::<i32>::new();

    array.extend([Some("hi"), None, Some("there"), None]);

    assert_eq!(
        Utf8Array::<i32>::from([Some("hi"), None, Some("there"), None]),
        array.as_arc().as_ref()
    );
}

#[test]
fn test_iter() {
    let mut array = MutableUtf8Array::<i32>::new();

    array.extend_trusted_len(vec![Some("hi"), Some("there")].into_iter());
    array.extend_trusted_len(vec![None, Some("hello")].into_iter());
    array.extend_trusted_len_values(["again"].iter());

    let result = array.iter().collect::<Vec<_>>();
    assert_eq!(
        result,
        vec![
            Some("hi"),
            Some("there"),
            None,
            Some("hello"),
            Some("again"),
        ]
    );
}

#[test]
fn as_box_twice() {
    let mut a = MutableUtf8Array::<i32>::new();
    let _ = a.as_box();
    let _ = a.as_box();
    let mut a = MutableUtf8Array::<i32>::new();
    let _ = a.as_arc();
    let _ = a.as_arc();
}

#[test]
fn extend_from_self() {
    let mut a = MutableUtf8Array::<i32>::from([Some("aa"), None]);

    a.try_extend_from_self(&a.clone()).unwrap();

    assert_eq!(
        a,
        MutableUtf8Array::<i32>::from([Some("aa"), None, Some("aa"), None])
    );
}

#[test]
fn test_set_validity() {
    let mut array = MutableUtf8Array::<i32>::from([Some("Red"), Some("Green"), Some("Blue")]);
    array.set_validity(Some([false, false, true].into()));

    assert!(!array.is_valid(0));
    assert!(!array.is_valid(1));
    assert!(array.is_valid(2));
}

#[test]
fn test_apply_validity() {
    let mut array = MutableUtf8Array::<i32>::from([Some("Red"), Some("Green"), Some("Blue")]);
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
    let mut array = MutableUtf8Array::<i32>::from([Some("Red"), Some("Green"), Some("Blue")]);

    array.apply_validity(|mut mut_bitmap| {
        mut_bitmap.set(1, false);
        mut_bitmap.set(2, false);
        mut_bitmap
    });

    assert!(array.is_valid(0));
    assert!(array.is_valid(1));
    assert!(array.is_valid(2));
}

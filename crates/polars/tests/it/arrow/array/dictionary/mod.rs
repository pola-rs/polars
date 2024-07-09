mod mutable;

use arrow::array::*;
use arrow::datatypes::ArrowDataType;

#[test]
fn try_new_ok() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let data_type =
        ArrowDataType::Dictionary(i32::KEY_TYPE, Box::new(values.data_type().clone()), false);
    let array = DictionaryArray::try_new(
        data_type,
        PrimitiveArray::from_vec(vec![1, 0]),
        values.boxed(),
    )
    .unwrap();

    assert_eq!(array.keys(), &PrimitiveArray::from_vec(vec![1i32, 0]));
    assert_eq!(
        &Utf8Array::<i32>::from_slice(["a", "aa"]) as &dyn Array,
        array.values().as_ref(),
    );
    assert!(!array.is_ordered());

    assert_eq!(format!("{array:?}"), "DictionaryArray[aa, a]");
}

#[test]
fn split_at() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let data_type =
        ArrowDataType::Dictionary(i32::KEY_TYPE, Box::new(values.data_type().clone()), false);
    let array = DictionaryArray::try_new(
        data_type,
        PrimitiveArray::from_vec(vec![1, 0]),
        values.boxed(),
    )
    .unwrap();

    let (lhs, rhs) = array.split_at(1);

    assert_eq!(format!("{lhs:?}"), "DictionaryArray[aa]");
    assert_eq!(format!("{rhs:?}"), "DictionaryArray[a]");
}

#[test]
fn try_new_incorrect_key() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let data_type =
        ArrowDataType::Dictionary(i16::KEY_TYPE, Box::new(values.data_type().clone()), false);

    let r = DictionaryArray::try_new(
        data_type,
        PrimitiveArray::from_vec(vec![1, 0]),
        values.boxed(),
    )
    .is_err();

    assert!(r);
}

#[test]
fn try_new_nulls() {
    let key: Option<u32> = None;
    let keys = PrimitiveArray::from_iter([key]);
    let value: &[&str] = &[];
    let values = Utf8Array::<i32>::from_slice(value);

    let data_type =
        ArrowDataType::Dictionary(u32::KEY_TYPE, Box::new(values.data_type().clone()), false);
    let r = DictionaryArray::try_new(data_type, keys, values.boxed()).is_ok();

    assert!(r);
}

#[test]
fn try_new_incorrect_dt() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let data_type = ArrowDataType::Int32;

    let r = DictionaryArray::try_new(
        data_type,
        PrimitiveArray::from_vec(vec![1, 0]),
        values.boxed(),
    )
    .is_err();

    assert!(r);
}

#[test]
fn try_new_incorrect_values_dt() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let data_type =
        ArrowDataType::Dictionary(i32::KEY_TYPE, Box::new(ArrowDataType::LargeUtf8), false);

    let r = DictionaryArray::try_new(
        data_type,
        PrimitiveArray::from_vec(vec![1, 0]),
        values.boxed(),
    )
    .is_err();

    assert!(r);
}

#[test]
fn try_new_out_of_bounds() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);

    let r = DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![2, 0]), values.boxed())
        .is_err();

    assert!(r);
}

#[test]
fn try_new_out_of_bounds_neg() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);

    let r = DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![-1, 0]), values.boxed())
        .is_err();

    assert!(r);
}

#[test]
fn new_null() {
    let dt = ArrowDataType::Dictionary(i16::KEY_TYPE, Box::new(ArrowDataType::Int32), false);
    let array = DictionaryArray::<i16>::new_null(dt, 2);

    assert_eq!(format!("{array:?}"), "DictionaryArray[None, None]");
}

#[test]
fn new_empty() {
    let dt = ArrowDataType::Dictionary(i16::KEY_TYPE, Box::new(ArrowDataType::Int32), false);
    let array = DictionaryArray::<i16>::new_empty(dt);

    assert_eq!(format!("{array:?}"), "DictionaryArray[]");
}

#[test]
fn with_validity() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let array =
        DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![1, 0]), values.boxed())
            .unwrap();

    let array = array.with_validity(Some([true, false].into()));

    assert_eq!(format!("{array:?}"), "DictionaryArray[aa, None]");
}

#[test]
fn rev_iter() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let array =
        DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![1, 0]), values.boxed())
            .unwrap();

    let mut iter = array.into_iter();
    assert_eq!(iter.by_ref().rev().count(), 2);
    assert_eq!(iter.size_hint(), (0, Some(0)));
}

#[test]
fn iter_values() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let array =
        DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![1, 0]), values.boxed())
            .unwrap();

    let mut iter = array.values_iter();
    assert_eq!(iter.by_ref().count(), 2);
    assert_eq!(iter.size_hint(), (0, Some(0)));
}

#[test]
fn keys_values_iter() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let array =
        DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![1, 0]), values.boxed())
            .unwrap();

    assert_eq!(array.keys_values_iter().collect::<Vec<_>>(), vec![1, 0]);
}

#[test]
fn iter_values_typed() {
    let values = Utf8Array::<i32>::from_slice(["a", "aa"]);
    let array =
        DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![1, 0, 0]), values.boxed())
            .unwrap();

    let iter = array.values_iter_typed::<Utf8Array<i32>>().unwrap();
    assert_eq!(iter.size_hint(), (3, Some(3)));
    assert_eq!(iter.collect::<Vec<_>>(), vec!["aa", "a", "a"]);

    let iter = array.iter_typed::<Utf8Array<i32>>().unwrap();
    assert_eq!(iter.size_hint(), (3, Some(3)));
    assert_eq!(
        iter.collect::<Vec<_>>(),
        vec![Some("aa"), Some("a"), Some("a")]
    );
}

#[test]
#[should_panic]
fn iter_values_typed_panic() {
    let values = Utf8Array::<i32>::from_iter([Some("a"), Some("aa"), None]);
    let array =
        DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![1, 0, 0]), values.boxed())
            .unwrap();

    // should not be iterating values
    let iter = array.values_iter_typed::<Utf8Array<i32>>().unwrap();
    let _ = iter.collect::<Vec<_>>();
}

#[test]
#[should_panic]
fn iter_values_typed_panic_2() {
    let values = Utf8Array::<i32>::from_iter([Some("a"), Some("aa"), None]);
    let array =
        DictionaryArray::try_from_keys(PrimitiveArray::from_vec(vec![1, 0, 0]), values.boxed())
            .unwrap();

    // should not be iterating values
    let iter = array.iter_typed::<Utf8Array<i32>>().unwrap();
    let _ = iter.collect::<Vec<_>>();
}

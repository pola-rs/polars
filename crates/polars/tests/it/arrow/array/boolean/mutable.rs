use arrow::array::{MutableArray, MutableBooleanArray, TryExtendFromSelf};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

#[test]
fn set() {
    let mut a = MutableBooleanArray::from(&[Some(false), Some(true), Some(false)]);

    a.set(1, None);
    a.set(0, Some(true));
    assert_eq!(
        a,
        MutableBooleanArray::from([Some(true), None, Some(false)])
    );
    assert_eq!(a.values(), &MutableBitmap::from([true, false, false]));
}

#[test]
fn push() {
    let mut a = MutableBooleanArray::new();
    a.push_value(true);
    a.push_value(false);
    a.push(None);
    a.push_null();
    assert_eq!(
        a,
        MutableBooleanArray::from([Some(true), Some(false), None, None])
    );
}

#[test]
fn pop() {
    let mut a = MutableBooleanArray::new();
    a.push(Some(true));
    a.push(Some(false));
    a.push(None);
    a.push_null();

    assert_eq!(a.pop(), None);
    assert_eq!(a.len(), 3);
    assert_eq!(a.pop(), None);
    assert_eq!(a.len(), 2);
    assert_eq!(a.pop(), Some(false));
    assert_eq!(a.len(), 1);
    assert_eq!(a.pop(), Some(true));
    assert_eq!(a.len(), 0);
    assert_eq!(a.pop(), None);
    assert_eq!(a.len(), 0);
}

#[test]
fn pop_all_some() {
    let mut a = MutableBooleanArray::new();
    for _ in 0..4 {
        a.push(Some(true));
    }

    for _ in 0..4 {
        a.push(Some(false));
    }

    a.push(Some(true));

    assert_eq!(a.pop(), Some(true));
    assert_eq!(a.pop(), Some(false));
    assert_eq!(a.pop(), Some(false));
    assert_eq!(a.pop(), Some(false));
    assert_eq!(a.len(), 5);

    assert_eq!(
        a,
        MutableBooleanArray::from([Some(true), Some(true), Some(true), Some(true), Some(false)])
    );
}

#[test]
fn from_trusted_len_iter() {
    let iter = std::iter::repeat(true).take(2).map(Some);
    let a = MutableBooleanArray::from_trusted_len_iter(iter);
    assert_eq!(a, MutableBooleanArray::from([Some(true), Some(true)]));
}

#[test]
fn from_iter() {
    let iter = std::iter::repeat(true).take(2).map(Some);
    let a: MutableBooleanArray = iter.collect();
    assert_eq!(a, MutableBooleanArray::from([Some(true), Some(true)]));
}

#[test]
fn try_from_trusted_len_iter() {
    let iter = vec![Some(true), Some(true), None]
        .into_iter()
        .map(PolarsResult::Ok);
    let a = MutableBooleanArray::try_from_trusted_len_iter(iter).unwrap();
    assert_eq!(a, MutableBooleanArray::from([Some(true), Some(true), None]));
}

#[test]
fn reserve() {
    let mut a = MutableBooleanArray::try_new(
        ArrowDataType::Boolean,
        MutableBitmap::new(),
        Some(MutableBitmap::new()),
    )
    .unwrap();

    a.reserve(10);
    assert!(a.validity().unwrap().capacity() > 0);
    assert!(a.values().capacity() > 0)
}

#[test]
fn extend_trusted_len() {
    let mut a = MutableBooleanArray::new();

    a.extend_trusted_len(vec![Some(true), Some(false)].into_iter());
    assert_eq!(a.validity(), None);

    a.extend_trusted_len(vec![None, Some(true)].into_iter());
    assert_eq!(
        a.validity(),
        Some(&MutableBitmap::from([true, true, false, true]))
    );
    assert_eq!(a.values(), &MutableBitmap::from([true, false, false, true]));
}

#[test]
fn extend_trusted_len_values() {
    let mut a = MutableBooleanArray::new();

    a.extend_trusted_len_values(vec![true, true, false].into_iter());
    assert_eq!(a.validity(), None);
    assert_eq!(a.values(), &MutableBitmap::from([true, true, false]));

    let mut a = MutableBooleanArray::new();
    a.push(None);
    a.extend_trusted_len_values(vec![true, false].into_iter());
    assert_eq!(
        a.validity(),
        Some(&MutableBitmap::from([false, true, true]))
    );
    assert_eq!(a.values(), &MutableBitmap::from([false, true, false]));
}

#[test]
fn into_iter() {
    let ve = MutableBitmap::from([true, false])
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(ve, vec![true, false]);
    let ve = MutableBitmap::from([true, false])
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(ve, vec![true, false]);
}

#[test]
fn shrink_to_fit() {
    let mut a = MutableBitmap::with_capacity(100);
    a.push(true);
    a.shrink_to_fit();
    assert_eq!(a.capacity(), 8);
}

#[test]
fn extend_from_self() {
    let mut a = MutableBooleanArray::from([Some(true), None]);

    a.try_extend_from_self(&a.clone()).unwrap();

    assert_eq!(
        a,
        MutableBooleanArray::from([Some(true), None, Some(true), None])
    );
}

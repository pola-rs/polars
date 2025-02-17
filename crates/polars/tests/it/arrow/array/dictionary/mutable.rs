use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;

use arrow::array::indexable::{AsIndexed, Indexable};
use arrow::array::*;
use polars_error::PolarsResult;
use polars_utils::aliases::{InitHashMaps, PlHashSet};

#[test]
fn primitive() -> PolarsResult<()> {
    let data = vec![Some(1), Some(2), Some(1)];

    let mut a = MutableDictionaryArray::<i32, MutablePrimitiveArray<i32>>::new();
    a.try_extend(data)?;
    assert_eq!(a.len(), 3);
    assert_eq!(a.values().len(), 2);
    Ok(())
}

#[test]
fn utf8_natural() -> PolarsResult<()> {
    let data = vec![Some("a"), Some("b"), Some("a")];

    let mut a = MutableDictionaryArray::<i32, MutableUtf8Array<i32>>::new();
    a.try_extend(data)?;

    assert_eq!(a.len(), 3);
    assert_eq!(a.values().len(), 2);
    Ok(())
}

#[test]
fn binary_natural() -> PolarsResult<()> {
    let data = vec![
        Some("a".as_bytes()),
        Some("b".as_bytes()),
        Some("a".as_bytes()),
    ];

    let mut a = MutableDictionaryArray::<i32, MutableBinaryArray<i32>>::new();
    a.try_extend(data)?;
    assert_eq!(a.len(), 3);
    assert_eq!(a.values().len(), 2);
    Ok(())
}

#[test]
fn push_utf8() {
    let mut new: MutableDictionaryArray<i32, MutableUtf8Array<i32>> = MutableDictionaryArray::new();

    for value in [Some("A"), Some("B"), None, Some("C"), Some("A"), Some("B")] {
        new.try_push(value).unwrap();
    }

    assert_eq!(
        new.values().values(),
        MutableUtf8Array::<i32>::from_iter_values(["A", "B", "C"].into_iter()).values()
    );

    let mut expected_keys = MutablePrimitiveArray::<i32>::from_slice([0, 1]);
    expected_keys.push(None);
    expected_keys.push(Some(2));
    expected_keys.push(Some(0));
    expected_keys.push(Some(1));
    assert_eq!(*new.keys(), expected_keys);
}

#[test]
fn into_empty() {
    let mut new: MutableDictionaryArray<i32, MutableUtf8Array<i32>> = MutableDictionaryArray::new();
    for value in [Some("A"), Some("B"), None, Some("C"), Some("A"), Some("B")] {
        new.try_push(value).unwrap();
    }
    let values = new.values().clone();
    let empty = new.into_empty();
    assert_eq!(empty.values(), &values);
    assert!(empty.is_empty());
}

#[test]
fn from_values() {
    let mut new: MutableDictionaryArray<i32, MutableUtf8Array<i32>> = MutableDictionaryArray::new();
    for value in [Some("A"), Some("B"), None, Some("C"), Some("A"), Some("B")] {
        new.try_push(value).unwrap();
    }
    let mut values = new.values().clone();
    let empty = MutableDictionaryArray::<i32, _>::from_values(values.clone()).unwrap();
    assert_eq!(empty.values(), &values);
    assert!(empty.is_empty());
    values.push(Some("A"));
    assert!(MutableDictionaryArray::<i32, _>::from_values(values).is_err());
}

#[test]
fn try_empty() {
    let mut values = MutableUtf8Array::<i32>::new();
    MutableDictionaryArray::<i32, _>::try_empty(values.clone()).unwrap();
    values.push(Some("A"));
    assert!(MutableDictionaryArray::<i32, _>::try_empty(values.clone()).is_err());
}

fn test_push_ex<M, T>(values: Vec<T>, gen: impl Fn(usize) -> T)
where
    M: MutableArray + Indexable + TryPush<Option<T>> + TryExtend<Option<T>> + Default + 'static,
    M::Type: Eq + Hash + Debug,
    T: AsIndexed<M> + Default + Clone + Eq + Hash,
{
    for is_extend in [false, true] {
        let mut set = PlHashSet::new();
        let mut arr = MutableDictionaryArray::<u8, M>::new();
        macro_rules! push {
            ($v:expr) => {
                if is_extend {
                    arr.try_extend(std::iter::once($v))
                } else {
                    arr.try_push($v)
                }
            };
        }
        arr.push_null();
        push!(None).unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.values().len(), 0);
        for (i, v) in values.iter().cloned().enumerate() {
            push!(Some(v.clone())).unwrap();
            let is_dup = !set.insert(v.clone());
            if !is_dup {
                assert_eq!(arr.values().value_at(i).borrow(), v.as_indexed());
                assert_eq!(arr.keys().value_at(arr.keys().len() - 1), i as u8);
            }
            assert_eq!(arr.values().len(), set.len());
            assert_eq!(arr.len(), 3 + i);
        }
        for i in 0..256 - set.len() {
            push!(Some(gen(i))).unwrap();
        }
        assert!(push!(Some(gen(256))).is_err());
    }
}

#[test]
fn test_push_utf8_ex() {
    test_push_ex::<MutableUtf8Array<i32>, _>(vec!["a".into(), "b".into(), "a".into()], |i| {
        i.to_string()
    })
}

#[test]
fn test_push_i64_ex() {
    test_push_ex::<MutablePrimitiveArray<i64>, _>(vec![10, 20, 30, 20], |i| 1000 + i as i64);
}

#[test]
fn test_big_dict() {
    let n = 10;
    let strings = (0..10).map(|i| i.to_string()).collect::<Vec<_>>();
    let mut arr = MutableDictionaryArray::<u8, MutableUtf8Array<i32>>::new();
    for s in &strings {
        arr.try_push(Some(s)).unwrap();
    }
    assert_eq!(arr.values().len(), n);
    for _ in 0..10_000 {
        for s in &strings {
            arr.try_push(Some(s)).unwrap();
        }
    }
    assert_eq!(arr.values().len(), n);
}

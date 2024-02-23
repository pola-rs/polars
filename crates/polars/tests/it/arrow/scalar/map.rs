use arrow::array::{BooleanArray, StructArray, Utf8Array};
use arrow::datatypes::{ArrowDataType, Field};
use arrow::scalar::{MapScalar, Scalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let kv_dt = ArrowDataType::Struct(vec![
        Field::new("key", ArrowDataType::Utf8, false),
        Field::new("value", ArrowDataType::Boolean, true),
    ]);
    let kv_array1 = StructArray::try_new(
        kv_dt.clone(),
        vec![
            Utf8Array::<i32>::from([Some("k1"), Some("k2")]).boxed(),
            BooleanArray::from_slice([true, false]).boxed(),
        ],
        None,
    )
    .unwrap();
    let kv_array2 = StructArray::try_new(
        kv_dt.clone(),
        vec![
            Utf8Array::<i32>::from([Some("k1"), Some("k3")]).boxed(),
            BooleanArray::from_slice([true, true]).boxed(),
        ],
        None,
    )
    .unwrap();

    let dt = ArrowDataType::Map(Box::new(Field::new("entries", kv_dt, true)), false);
    let a = MapScalar::new(dt.clone(), Some(Box::new(kv_array1)));
    let b = MapScalar::new(dt.clone(), None);
    assert_eq!(a, a);
    assert_eq!(b, b);
    assert!(a != b);
    let b = MapScalar::new(dt, Some(Box::new(kv_array2)));
    assert!(a != b);
    assert_eq!(b, b);
}

#[test]
fn basics() {
    let kv_dt = ArrowDataType::Struct(vec![
        Field::new("key", ArrowDataType::Utf8, false),
        Field::new("value", ArrowDataType::Boolean, true),
    ]);
    let kv_array = StructArray::try_new(
        kv_dt.clone(),
        vec![
            Utf8Array::<i32>::from([Some("k1"), Some("k2")]).boxed(),
            BooleanArray::from_slice([true, false]).boxed(),
        ],
        None,
    )
    .unwrap();

    let dt = ArrowDataType::Map(Box::new(Field::new("entries", kv_dt, true)), false);
    let a = MapScalar::new(dt.clone(), Some(Box::new(kv_array.clone())));

    assert_eq!(kv_array, a.values().as_ref());
    assert_eq!(a.data_type(), &dt);
    assert!(a.is_valid());

    let _: &dyn std::any::Any = a.as_any();
}

use arrow::array::BooleanArray;
use arrow::datatypes::{ArrowDataType, Field};
use arrow::scalar::{FixedSizeListScalar, Scalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let dt =
        ArrowDataType::FixedSizeList(Box::new(Field::new("a", ArrowDataType::Boolean, true)), 2);
    let a = FixedSizeListScalar::new(
        dt.clone(),
        Some(BooleanArray::from_slice([true, false]).boxed()),
    );

    let b = FixedSizeListScalar::new(dt.clone(), None);

    assert_eq!(a, a);
    assert_eq!(b, b);
    assert!(a != b);

    let b = FixedSizeListScalar::new(dt, Some(BooleanArray::from_slice([true, true]).boxed()));
    assert!(a != b);
    assert_eq!(b, b);
}

#[test]
fn basics() {
    let dt =
        ArrowDataType::FixedSizeList(Box::new(Field::new("a", ArrowDataType::Boolean, true)), 2);
    let a = FixedSizeListScalar::new(
        dt.clone(),
        Some(BooleanArray::from_slice([true, false]).boxed()),
    );

    assert_eq!(
        BooleanArray::from_slice([true, false]),
        a.values().unwrap().as_ref()
    );
    assert_eq!(a.data_type(), &dt);
    assert!(a.is_valid());

    let _: &dyn std::any::Any = a.as_any();
}

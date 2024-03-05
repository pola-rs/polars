use arrow::array::BooleanArray;
use arrow::datatypes::{ArrowDataType, Field};
use arrow::scalar::{ListScalar, Scalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let dt = ArrowDataType::List(Box::new(Field::new("a", ArrowDataType::Boolean, true)));
    let a = ListScalar::<i32>::new(
        dt.clone(),
        Some(BooleanArray::from_slice([true, false]).boxed()),
    );
    let b = ListScalar::<i32>::new(dt.clone(), None);
    assert_eq!(a, a);
    assert_eq!(b, b);
    assert!(a != b);
    let b = ListScalar::<i32>::new(dt, Some(BooleanArray::from_slice([true, true]).boxed()));
    assert!(a != b);
    assert_eq!(b, b);
}

#[test]
fn basics() {
    let dt = ArrowDataType::List(Box::new(Field::new("a", ArrowDataType::Boolean, true)));
    let a = ListScalar::<i32>::new(
        dt.clone(),
        Some(BooleanArray::from_slice([true, false]).boxed()),
    );

    assert_eq!(BooleanArray::from_slice([true, false]), a.values().as_ref());
    assert_eq!(a.data_type(), &dt);
    assert!(a.is_valid());

    let _: &dyn std::any::Any = a.as_any();
}

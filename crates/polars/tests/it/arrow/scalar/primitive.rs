use arrow::datatypes::ArrowDataType;
use arrow::scalar::{PrimitiveScalar, Scalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let a = PrimitiveScalar::from(Some(2i32));
    let b = PrimitiveScalar::<i32>::from(None);
    assert_eq!(a, a);
    assert_eq!(b, b);
    assert!(a != b);
    let b = PrimitiveScalar::<i32>::from(Some(1i32));
    assert!(a != b);
    assert_eq!(b, b);
}

#[test]
fn basics() {
    let a = PrimitiveScalar::from(Some(2i32));

    assert_eq!(a.value(), &Some(2i32));
    assert_eq!(a.data_type(), &ArrowDataType::Int32);

    let a = a.to(ArrowDataType::Date32);
    assert_eq!(a.data_type(), &ArrowDataType::Date32);

    let a = PrimitiveScalar::<i32>::from(None);

    assert_eq!(a.data_type(), &ArrowDataType::Int32);
    assert!(!a.is_valid());

    let a = a.to(ArrowDataType::Date32);
    assert_eq!(a.data_type(), &ArrowDataType::Date32);

    let _: &dyn std::any::Any = a.as_any();
}

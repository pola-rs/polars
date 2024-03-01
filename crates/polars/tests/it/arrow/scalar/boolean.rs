use arrow::datatypes::ArrowDataType;
use arrow::scalar::{BooleanScalar, Scalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let a = BooleanScalar::from(Some(true));
    let b = BooleanScalar::from(None);
    assert_eq!(a, a);
    assert_eq!(b, b);
    assert!(a != b);
    let b = BooleanScalar::from(Some(false));
    assert!(a != b);
    assert_eq!(b, b);
}

#[test]
fn basics() {
    let a = BooleanScalar::new(Some(true));

    assert_eq!(a.value(), Some(true));
    assert_eq!(a.data_type(), &ArrowDataType::Boolean);
    assert!(a.is_valid());

    let _: &dyn std::any::Any = a.as_any();
}

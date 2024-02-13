use arrow::datatypes::ArrowDataType;
use arrow::scalar::{NullScalar, Scalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let a = NullScalar::new();
    assert_eq!(a, a);
}

#[test]
fn basics() {
    let a = NullScalar::default();

    assert_eq!(a.data_type(), &ArrowDataType::Null);
    assert!(!a.is_valid());

    let _: &dyn std::any::Any = a.as_any();
}

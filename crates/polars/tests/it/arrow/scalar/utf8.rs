use arrow::datatypes::ArrowDataType;
use arrow::scalar::{Scalar, Utf8Scalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let a = Utf8Scalar::<i32>::from(Some("a"));
    let b = Utf8Scalar::<i32>::from(None::<&str>);
    assert_eq!(a, a);
    assert_eq!(b, b);
    assert!(a != b);
    let b = Utf8Scalar::<i32>::from(Some("b"));
    assert!(a != b);
    assert_eq!(b, b);
}

#[test]
fn basics() {
    let a = Utf8Scalar::<i32>::from(Some("a"));

    assert_eq!(a.value(), Some("a"));
    assert_eq!(a.data_type(), &ArrowDataType::Utf8);
    assert!(a.is_valid());

    let a = Utf8Scalar::<i64>::from(None::<&str>);

    assert_eq!(a.data_type(), &ArrowDataType::LargeUtf8);
    assert!(!a.is_valid());

    let _: &dyn std::any::Any = a.as_any();
}

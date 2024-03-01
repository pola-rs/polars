use arrow::datatypes::{ArrowDataType, Field};
use arrow::scalar::{BooleanScalar, Scalar, StructScalar};

#[allow(clippy::eq_op)]
#[test]
fn equal() {
    let dt = ArrowDataType::Struct(vec![Field::new("a", ArrowDataType::Boolean, true)]);
    let a = StructScalar::new(
        dt.clone(),
        Some(vec![
            Box::new(BooleanScalar::from(Some(true))) as Box<dyn Scalar>
        ]),
    );
    let b = StructScalar::new(dt.clone(), None);
    assert_eq!(a, a);
    assert_eq!(b, b);
    assert!(a != b);
    let b = StructScalar::new(
        dt,
        Some(vec![
            Box::new(BooleanScalar::from(Some(false))) as Box<dyn Scalar>
        ]),
    );
    assert!(a != b);
    assert_eq!(b, b);
}

#[test]
fn basics() {
    let dt = ArrowDataType::Struct(vec![Field::new("a", ArrowDataType::Boolean, true)]);

    let values = vec![Box::new(BooleanScalar::from(Some(true))) as Box<dyn Scalar>];

    let a = StructScalar::new(dt.clone(), Some(values.clone()));

    assert_eq!(a.values(), &values);
    assert_eq!(a.data_type(), &dt);
    assert!(a.is_valid());

    let _: &dyn std::any::Any = a.as_any();
}

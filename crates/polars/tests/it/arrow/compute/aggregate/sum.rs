use arrow::array::*;
use arrow::compute::aggregate::{sum, sum_primitive};
use arrow::datatypes::ArrowDataType;
use arrow::scalar::{PrimitiveScalar, Scalar};

#[test]
fn test_primitive_array_sum() {
    let a = Int32Array::from_slice([1, 2, 3, 4, 5]);
    assert_eq!(
        &PrimitiveScalar::<i32>::from(Some(15)) as &dyn Scalar,
        sum(&a).unwrap().as_ref()
    );

    let a = a.to(ArrowDataType::Date32);
    assert_eq!(
        &PrimitiveScalar::<i32>::from(Some(15)).to(ArrowDataType::Date32) as &dyn Scalar,
        sum(&a).unwrap().as_ref()
    );
}

#[test]
fn test_primitive_array_float_sum() {
    let a = Float64Array::from_slice([1.1f64, 2.2, 3.3, 4.4, 5.5]);
    assert!((16.5 - sum_primitive(&a).unwrap()).abs() < f64::EPSILON);
}

#[test]
fn test_primitive_array_sum_with_nulls() {
    let a = Int32Array::from(&[None, Some(2), Some(3), None, Some(5)]);
    assert_eq!(10, sum_primitive(&a).unwrap());
}

#[test]
fn test_primitive_array_sum_all_nulls() {
    let a = Int32Array::from(&[None, None, None]);
    assert_eq!(None, sum_primitive(&a));
}

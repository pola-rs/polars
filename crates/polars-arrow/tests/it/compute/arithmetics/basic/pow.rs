use polars_arrow::array::*;
use polars_arrow::compute::arithmetics::basic::*;

#[test]
fn test_raise_power_scalar() {
    let a = Float32Array::from(&[Some(2f32), None]);
    let actual = powf_scalar(&a, 2.0);
    let expected = Float32Array::from(&[Some(4f32), None]);
    assert_eq!(expected, actual);
}

#[test]
fn test_raise_power_scalar_checked() {
    let a = Int8Array::from(&[Some(1i8), None, Some(7i8)]);
    let actual = checked_powf_scalar(&a, 8usize);
    let expected = Int8Array::from(&[Some(1i8), None, None]);
    assert_eq!(expected, actual);
}

use polars_arrow::array::{new_null_array, Int32Array};
use polars_arrow::compute::window::*;
use polars_arrow::datatypes::ArrowDataType;

#[test]
fn shift_pos() {
    let array = Int32Array::from(&[Some(1), None, Some(3)]);
    let result = shift(&array, 1).unwrap();

    let expected = Int32Array::from(&[None, Some(1), None]);

    assert_eq!(expected, result.as_ref());
}

#[test]
fn shift_many() {
    let array = Int32Array::from(&[Some(1), None, Some(3)]).to(ArrowDataType::Date32);
    assert!(shift(&array, 10).is_err());
}

#[test]
fn shift_max() {
    let array = Int32Array::from(&[Some(1), None, Some(3)]).to(ArrowDataType::Date32);
    let result = shift(&array, 3).unwrap();

    let expected = new_null_array(ArrowDataType::Date32, 3);

    assert_eq!(expected.as_ref(), result.as_ref());
}

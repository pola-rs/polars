use arrow::array::*;
use arrow::datatypes::Field;
use arrow::ffi;
use polars_error::PolarsResult;

fn _test_round_trip(array: Box<dyn Array>, expected: Box<dyn Array>) -> PolarsResult<()> {
    let field = Field::new("a", array.data_type().clone(), true);

    // export array and corresponding data_type
    let array_ffi = ffi::export_array_to_c(array);
    let schema_ffi = ffi::export_field_to_c(&field);

    // import references
    let result_field = unsafe { ffi::import_field_from_c(&schema_ffi)? };
    let result_array =
        unsafe { ffi::import_array_from_c(array_ffi, result_field.data_type.clone())? };

    assert_eq!(&result_array, &expected);
    assert_eq!(result_field, field);
    Ok(())
}

fn test_round_trip(expected: impl Array + Clone + 'static) -> PolarsResult<()> {
    let array: Box<dyn Array> = Box::new(expected.clone());
    let expected = Box::new(expected) as Box<dyn Array>;
    _test_round_trip(array.clone(), clone(expected.as_ref()))?;

    // sliced
    _test_round_trip(array.sliced(1, 2), expected.sliced(1, 2))
}

#[test]
fn bool_nullable() -> PolarsResult<()> {
    let data = BooleanArray::from(&[Some(true), None, Some(false), None]);
    test_round_trip(data)
}

#[test]
fn binview_nullable_inlined() -> PolarsResult<()> {
    let data = Utf8ViewArray::from_slice([Some("foo"), None, Some("barbar"), None]);
    test_round_trip(data)
}

#[test]
fn binview_nullable_buffered() -> PolarsResult<()> {
    let data = Utf8ViewArray::from_slice([
        Some("foobaroiwalksdfjoiei"),
        None,
        Some("barbar"),
        None,
        Some("aoisejiofjfoiewjjwfoiwejfo"),
    ]);
    test_round_trip(data)
}

use arrow::array::*;
use arrow::datatypes::{ArrowDataType, Field, UnionMode, UnionType};
use arrow::ffi;
use polars_error::PolarsResult;

fn _test_round_trip(array: Box<dyn Array>, expected: Box<dyn Array>) -> PolarsResult<()> {
    let field = Field::new("a".into(), array.dtype().clone(), true);

    // export array and corresponding dtype
    let array_ffi = ffi::export_array_to_c(array);
    let schema_ffi = ffi::export_field_to_c(&field);

    // import references
    let result_field = unsafe { ffi::import_field_from_c(&schema_ffi)? };
    let result_array = unsafe { ffi::import_array_from_c(array_ffi, result_field.dtype.clone())? };

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

/// Explicit `ids`: the C format string always carries type ids, so `None` would not round-trip.
fn union(mode: UnionMode) -> UnionArray {
    let fields = vec![
        Field::new("i".into(), ArrowDataType::Int32, true),
        Field::new("s".into(), ArrowDataType::LargeUtf8, true),
    ];
    let dtype = ArrowDataType::Union(Box::new(UnionType {
        fields,
        ids: Some(vec![0, 1]),
        mode,
    }));
    let children: Vec<Box<dyn Array>> = vec![
        Box::new(PrimitiveArray::<i32>::from_slice([1, 2, 3, 4])),
        Box::new(Utf8Array::<i64>::from_slice(["a", "bb", "ccc", "dddd"])),
    ];
    let offsets = matches!(mode, UnionMode::Dense).then(|| vec![0i32, 1, 2, 3].into());
    UnionArray::new(dtype, vec![0i8, 1, 0, 1].into(), children, offsets)
}

#[test]
fn union_sparse() -> PolarsResult<()> {
    test_round_trip(union(UnionMode::Sparse))
}

#[test]
fn union_dense() -> PolarsResult<()> {
    test_round_trip(union(UnionMode::Dense))
}

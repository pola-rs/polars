use polars::prelude::*;
use polars::series::*;

#[test]
fn test_series_arithmetic() -> PolarsResult<()> {
    let a = &Series::new("a".into(), &[1, 100, 6, 40]);
    let b = &Series::new("b".into(), &[-1, 2, 3, 4]);
    assert_eq!((a + b)?, Series::new("a".into(), &[0, 102, 9, 44]));
    assert_eq!((a - b)?, Series::new("a".into(), &[2, 98, 3, 36]));
    assert_eq!((a * b)?, Series::new("a".into(), &[-1, 200, 18, 160]));
    assert_eq!((a / b)?, Series::new("a".into(), &[-1, 50, 2, 10]));

    Ok(())
}

#[test]
fn test_min_max_sorted_asc() {
    let a = &mut Series::new("a".into(), &[1, 2, 3, 4]);
    a.set_sorted_flag(IsSorted::Ascending);
    assert_eq!(a.max().unwrap(), Some(4));
    assert_eq!(a.min().unwrap(), Some(1));
}

#[test]
fn test_min_max_sorted_desc() {
    let a = &mut Series::new("a".into(), &[4, 3, 2, 1]);
    a.set_sorted_flag(IsSorted::Descending);
    assert_eq!(a.max().unwrap(), Some(4));
    assert_eq!(a.min().unwrap(), Some(1));
}

#[test]
fn test_construct_list_of_null_series() {
    let s = Series::new(
        "a".into(),
        [
            Series::new_null("a1".into(), 1),
            Series::new_null("a1".into(), 1),
        ],
    );
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.field().name(), "a");
}

#[test]
#[cfg(feature = "dtype-struct")]
fn test_borrowed_struct_values_matched_to_fields() -> PolarsResult<()> {
    let to_series = |df: DataFrame| df.into_struct("s".into()).into_series();
    let ordered = to_series(df!("a" => [1i64, 2], "b" => ["x", "y"])?);
    let reordered = to_series(df!("b" => ["p", "q"], "a" => [3i64, 4])?);
    let missing = to_series(df!("a" => [5i64])?);
    let values = [
        ordered.get(0)?,
        reordered.get(0)?,
        reordered.get(1)?,
        missing.get(0)?,
        ordered.get(1)?,
        AnyValue::Null,
    ];
    assert!(matches!(values[0], AnyValue::Struct(..)));

    // Match each value's fields by position if it has the same field names ("ordered")
    // or by name ("reordered"); fields that don't match should be null ("missing").
    let s = Series::from_any_values_and_dtype("s".into(), &values, ordered.dtype(), true)?;
    let expected = df!(
        "a" => [Some(1i64), Some(3), Some(4), Some(5), Some(2), None],
        "b" => [Some("x"), Some("p"), Some("q"), None, Some("y"), None],
    )?;
    assert!(s.struct_()?.clone().unnest().equals_missing(&expected));
    assert_eq!(s.null_count(), 1);
    Ok(())
}

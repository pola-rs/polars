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
fn test_fill_null_median() -> PolarsResult<()> {
    let s = Series::new("a".into(), &[Some(1.0), None, Some(3.0), None, Some(5.0)]);
    let filled = s.fill_null(FillNullStrategy::Median)?;
    let expected = Series::new("a".into(), &[1.0, 3.0, 3.0, 3.0, 5.0]);
    assert_eq!(filled, expected);
    Ok(())
}

#[test]
fn test_fill_null_median_even_count() -> PolarsResult<()> {
    let s = Series::new(
        "a".into(),
        &[Some(1.0), None, Some(2.0), None, Some(4.0), Some(10.0)],
    );
    let filled = s.fill_null(FillNullStrategy::Median)?;
    let expected = Series::new("a".into(), &[1.0, 3.0, 2.0, 3.0, 4.0, 10.0]);
    assert_eq!(filled, expected);
    Ok(())
}

#[test]
fn test_fill_null_median_integer() -> PolarsResult<()> {
    let s = Series::new("a".into(), &[Some(1), None, Some(2), None, Some(10)]);
    let filled = s.fill_null(FillNullStrategy::Median)?;
    let expected = Series::new("a".into(), &[Some(1), Some(2), Some(2), Some(2), Some(10)]);
    assert_eq!(filled, expected);
    Ok(())
}

#[test]
fn test_fill_null_median_all_null() -> PolarsResult<()> {
    let s = Series::new("a".into(), &[None::<f64>, None, None]);
    let filled = s.fill_null(FillNullStrategy::Median)?;
    assert_eq!(filled.null_count(), 3);
    Ok(())
}

use polars::prelude::*;
use polars::series::*;

#[test]
fn test_series_arithmetic() -> PolarsResult<()> {
    let a = &Series::new("a", &[1, 100, 6, 40]);
    let b = &Series::new("b", &[-1, 2, 3, 4]);
    assert_eq!((a + b)?, Series::new("a", &[0, 102, 9, 44]));
    assert_eq!((a - b)?, Series::new("a", &[2, 98, 3, 36]));
    assert_eq!((a * b)?, Series::new("a", &[-1, 200, 18, 160]));
    assert_eq!((a / b)?, Series::new("a", &[-1, 50, 2, 10]));

    Ok(())
}

#[test]
fn test_min_max_sorted_asc() {
    let a = &mut Series::new("a", &[1, 2, 3, 4]);
    a.set_sorted_flag(IsSorted::Ascending);
    assert_eq!(a.max().unwrap(), Some(4));
    assert_eq!(a.min().unwrap(), Some(1));
}

#[test]
fn test_min_max_sorted_desc() {
    let a = &mut Series::new("a", &[4, 3, 2, 1]);
    a.set_sorted_flag(IsSorted::Descending);
    assert_eq!(a.max().unwrap(), Some(4));
    assert_eq!(a.min().unwrap(), Some(1));
}

#[test]
fn test_construct_list_of_null_series() {
    let s = Series::new("a", [Series::new_null("a1", 1), Series::new_null("a1", 1)]);
    assert_eq!(s.null_count(), 0);
    assert_eq!(s.field().name(), "a");
}

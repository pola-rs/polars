use polars::prelude::*;
use polars::series::*;

#[test]
fn test_series_arithmetic() {
    let a = &Series::new("a", &[1, 100, 6, 40]);
    let b = &Series::new("b", &[-1, 2, 3, 4]);
    assert_eq!(a + b, Series::new("a", &[0, 102, 9, 44]));
    assert_eq!(a - b, Series::new("a", &[2, 98, 3, 36]));
    assert_eq!(a * b, Series::new("a", &[-1, 200, 18, 160]));
    assert_eq!(a / b, Series::new("a", &[-1, 50, 2, 10]));
}

#[test]
fn test_aggregates() {
    let a = &Series::new("a", &[9, 11, 10]);
    assert_eq!(a.max(), Some(11));
    assert_eq!(a.min(), Some(9));
    assert_eq!(a.sum(), Some(30));
    assert_eq!(a.mean(), Some(10.0));
    assert_eq!(a.median(), Some(10.0));
}

#[test]
fn test_min_max_sorted_asc() {
    let a = &mut Series::new("a", &[1, 2, 3, 4]);
    a.set_sorted_flag(IsSorted::Ascending);
    assert_eq!(a.max(), Some(4));
    assert_eq!(a.min(), Some(1));
}

#[test]
fn test_min_max_sorted_desc() {
    let mut a = &mut Series::new("a", &[4, 3, 2, 1]);
    a.set_sorted_flag(IsSorted::Descending);
    assert_eq!(a.max(), Some(4));
    assert_eq!(a.min(), Some(1));
}

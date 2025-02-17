use polars_core::series::IsSorted;
use super::*;

#[test]
fn test_sample_sorted()  {
    let s = Series::new("a", [1, 2, 3]).sort(false);
    matches!(s.is_sorted_flag(),IsSorted::Ascending);
    let out = s.sample_frac(1.5, true, false, None).unwrap();
    matches!(s.is_sorted_flag(),IsSorted::Not);
}

#[test]
fn test_sample() {
    let df = df![
            "foo" => &[1, 2, 3, 4, 5]
        ]
        .unwrap();

    // default samples are random and don't require seeds
    assert!(df.sample_n(3, false, false, None).is_ok());
    assert!(df.sample_frac(0.4, false, false, None).is_ok());
    // with seeding
    assert!(df.sample_n(3, false, false, Some(0)).is_ok());
    assert!(df.sample_frac(0.4, false, false, Some(0)).is_ok());
    // without replacement can not sample more than 100%
    assert!(df.sample_frac(2.0, false, false, Some(0)).is_err());
    assert!(df.sample_n(3, true, false, Some(0)).is_ok());
    assert!(df.sample_frac(0.4, true, false, Some(0)).is_ok());
    // with replacement can sample more than 100%
    assert!(df.sample_frac(2.0, true, false, Some(0)).is_ok());
}

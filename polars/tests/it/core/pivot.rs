use polars::export::chrono::{NaiveDate, NaiveDateTime};
use polars::prelude::*;
use polars_ops::pivot::{pivot, pivot_stable, PivotAgg};

#[test]
#[cfg(feature = "dtype-date")]
fn test_pivot_date() -> PolarsResult<()> {
    let mut df = df![
        "A" => [1, 1, 1, 1, 1, 1, 1, 1],
        "B" => [8, 2, 3, 6, 3, 6, 2, 2],
        "C" => [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ]?;
    df.try_apply("C", |s| s.cast(&DataType::Date))?;

    let out = pivot(&df, ["A"], ["B"], ["C"], PivotAgg::Count, true, None)?;

    let first = 1 as IdxSize;
    let expected = df![
        "B" => [8i32, 2, 3, 6],
        "1972-09-27" => [first, 3, 2, 2]
    ]?;
    assert!(out.frame_equal_missing(&expected));

    let mut out = pivot_stable(&df, ["C"], ["B"], ["A"], PivotAgg::First, true, None)?;
    out.try_apply("1", |s| {
        let ca = s.date()?;
        Ok(ca.strftime("%Y-%d-%m"))
    })?;

    let expected = df![
        "B" => [8i32, 2, 3, 6],
        "1" => ["1972-27-09", "1972-27-09", "1972-27-09", "1972-27-09"]
    ]?;
    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

#[test]
fn test_pivot_old() {
    let s0 = Series::new("foo", ["A", "A", "B", "B", "C"].as_ref());
    let s1 = Series::new("N", [1, 2, 2, 4, 2].as_ref());
    let s2 = Series::new("bar", ["k", "l", "m", "m", "l"].as_ref());
    let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

    let pvt = pivot(&df, ["N"], ["foo"], ["bar"], PivotAgg::Sum, false, None).unwrap();
    assert_eq!(pvt.get_column_names(), &["foo", "k", "l", "m"]);
    assert_eq!(
        Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
        &[None, None, Some(6)]
    );
    let pvt = pivot(&df, ["N"], ["foo"], ["bar"], PivotAgg::Min, false, None).unwrap();
    assert_eq!(
        Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
        &[None, None, Some(2)]
    );
    let pvt = pivot(&df, ["N"], ["foo"], ["bar"], PivotAgg::Max, false, None).unwrap();
    assert_eq!(
        Vec::from(&pvt.column("m").unwrap().i32().unwrap().sort(false)),
        &[None, None, Some(4)]
    );
    let pvt = pivot(&df, ["N"], ["foo"], ["bar"], PivotAgg::Mean, false, None).unwrap();
    assert_eq!(
        Vec::from(&pvt.column("m").unwrap().f64().unwrap().sort(false)),
        &[None, None, Some(3.0)]
    );
    let pvt = pivot(&df, ["N"], ["foo"], ["bar"], PivotAgg::Count, false, None).unwrap();
    assert_eq!(
        Vec::from(&pvt.column("m").unwrap().idx().unwrap().sort(false)),
        &[None, None, Some(2)]
    );
}

#[test]
#[cfg(feature = "dtype-categorical")]
fn test_pivot_categorical() -> PolarsResult<()> {
    let mut df = df![
        "A" => [1, 1, 1, 1, 1, 1, 1, 1],
        "B" => [8, 2, 3, 6, 3, 6, 2, 2],
        "C" => ["a", "b", "c", "a", "b", "c", "a", "b"]
    ]?;
    df.try_apply("C", |s| s.cast(&DataType::Categorical(None)))?;

    let out = pivot(&df, ["A"], ["B"], ["C"], PivotAgg::Count, true, None)?;
    assert_eq!(out.get_column_names(), &["B", "a", "b", "c"]);

    Ok(())
}

#[test]
fn test_pivot_new() -> PolarsResult<()> {
    let df = df!["A"=> ["foo", "foo", "foo", "foo", "foo",
        "bar", "bar", "bar", "bar"],
        "B"=> ["one", "one", "one", "two", "two",
        "one", "one", "two", "two"],
        "C"=> ["small", "large", "large", "small",
        "small", "large", "small", "small", "large"],
        "breaky"=> ["jam", "egg", "egg", "egg",
         "jam", "jam", "potato", "jam", "jam"],
        "D"=> [1, 2, 2, 3, 3, 4, 5, 6, 7],
        "E"=> [2, 4, 5, 5, 6, 6, 8, 9, 9]
    ]?;

    let out = (pivot_stable(&df, ["D"], ["A", "B"], ["C"], PivotAgg::Sum, true, None))?;
    let expected = df![
        "A" => ["foo", "foo", "bar", "bar"],
        "B" => ["one", "two", "one", "two"],
        "large" => [Some(4), None, Some(4), Some(7)],
        "small" => [1, 6, 5, 6],
    ]?;
    assert!(out.frame_equal_missing(&expected));

    let out = pivot_stable(
        &df,
        ["D"],
        ["A", "B"],
        ["C", "breaky"],
        PivotAgg::Sum,
        true,
        None,
    )?;
    let expected = df![
        "A" => ["foo", "foo", "bar", "bar"],
        "B" => ["one", "two", "one", "two"],
        "large" => [Some(4), None, Some(4), Some(7)],
        "small" => [1, 6, 5, 6],
        "egg" => [Some(4), Some(3), None, None],
        "jam" => [1, 3, 4, 13],
        "potato" => [None, None, Some(5), None]
    ]?;
    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

#[test]
fn test_pivot_2() -> PolarsResult<()> {
    let df = df![
        "name"=> ["avg", "avg", "act", "test", "test"],
        "err" => [Some("name1"), Some("name2"), None, Some("name1"), Some("name2")],
        "wght"=> [0.0, 0.1, 1.0, 0.4, 0.2]
    ]?;

    let out = pivot_stable(
        &df,
        ["wght"],
        ["err"],
        ["name"],
        PivotAgg::First,
        false,
        None,
    )?;
    let expected = df![
        "err" => [Some("name1"), Some("name2"), None],
        "avg" => [Some(0.0), Some(0.1), None],
        "act" => [None, None, Some(1.)],
        "test" => [Some(0.4), Some(0.2), None],
    ]?;
    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "dtype-datetime")]
fn test_pivot_datetime() -> PolarsResult<()> {
    let dt = NaiveDate::from_ymd_opt(2021, 1, 1)
        .unwrap()
        .and_hms_opt(12, 15, 0)
        .unwrap();
    let df = df![
        "dt" => [dt, dt, dt, dt],
        "key" => ["x", "x", "y", "y"],
        "val" => [100, 50, 500, -80]
    ]?;

    let out = pivot(&df, ["val"], ["dt"], ["key"], PivotAgg::Sum, false, None)?;
    let expected = df![
        "dt" => [dt],
        "x" => [150],
        "y" => [420]
    ]?;
    assert!(out.frame_equal(&expected));

    Ok(())
}

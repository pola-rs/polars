use polars::prelude::*;

#[test]
#[cfg(feature = "dtype-date")]
fn test_pivot_date() -> Result<()> {
    let mut df = df![
        "A" => [1, 1, 1, 1, 1, 1, 1, 1],
        "B" => [8, 2, 3, 6, 3, 6, 2, 2],
        "C" => [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ]?;
    df.try_apply("C", |s| s.cast(&DataType::Date))?;

    let out = df.groupby_stable(["B"])?.pivot(["C"], ["A"]).count()?;
    let expected = df![
        "B" => [8i32, 2, 3, 6],
        "1972-09-27" => [1u32, 3, 2, 2]
    ]?;
    assert!(out.frame_equal_missing(&expected));

    let mut out = df.groupby_stable(["B"])?.pivot(["A"], ["C"]).first()?;
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

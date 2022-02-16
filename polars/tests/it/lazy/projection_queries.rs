use polars::prelude::*;

#[test]
fn test_sum_after_filter() -> Result<()> {
    let df = df![
        "ids" => 0..10,
        "values" => 10..20,
    ]?
    .lazy()
    .filter(not(col("ids").eq(lit(5))))
    .select([col("values").sum()])
    .collect()?;

    assert_eq!(df.column("values")?.get(0), AnyValue::Int32(130));
    Ok(())
}

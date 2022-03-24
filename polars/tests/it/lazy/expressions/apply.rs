use super::*;

#[test]
#[cfg(feature = "arange")]
fn test_arange_agg() -> Result<()> {
    let df = df![
        "x" => [5, 5, 4, 4, 2, 2]
    ]?;

    let out = df
        .lazy()
        .with_columns([arange(lit(0i32), count(), 1).over([col("x")])])
        .collect()?;
    assert_eq!(
        Vec::from_iter(out.column("literal")?.i64()?.into_no_null_iter()),
        &[0, 1, 0, 1, 0, 1]
    );

    Ok(())
}

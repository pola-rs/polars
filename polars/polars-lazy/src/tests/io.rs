use super::*;

#[test]
fn test_parquet_exec() -> Result<()> {
    // filter
    for par in [true, false] {
        let out = scan_foods_parquet(par)
            .filter(col("category").eq(lit("seafood")))
            .collect()?;
        assert_eq!(out.shape(), (8, 4));
    }

    // project
    for par in [true, false] {
        let out = scan_foods_parquet(par)
            .select([col("category"), col("sugars_g")])
            .collect()?;
        assert_eq!(out.shape(), (27, 2));
    }

    // project + filter
    for par in [true, false] {
        let out = scan_foods_parquet(par)
            .select([col("category"), col("sugars_g")])
            .filter(col("category").eq(lit("seafood")))
            .collect()?;
        assert_eq!(out.shape(), (8, 2));
    }

    Ok(())
}

#[test]
#[cfg(target_os = "unix")]
fn test_parquet_globbing() -> Result<()> {
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.parquet";
    let df = LazyFrame::scan_parquet(glob.into(), None, false, true, false)?.collect()?;
    assert_eq!(df.shape(), (54, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0), AnyValue::Int64(45));
    assert_eq!(cal.get(53), AnyValue::Int64(194));

    Ok(())
}

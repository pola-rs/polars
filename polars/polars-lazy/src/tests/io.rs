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
#[cfg(not(target_os = "windows"))]
fn test_parquet_globbing() -> Result<()> {
    // for side effects
    init_files();
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.parquet";
    let df = LazyFrame::scan_parquet(
        glob.into(),
        ScanArgsParquet {
            n_rows: None,
            cache: true,
            parallel: true,
            rechunk: false,
        },
    )?
    .collect()?;
    assert_eq!(df.shape(), (54, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0), AnyValue::Int64(45));
    assert_eq!(cal.get(53), AnyValue::Int64(194));

    Ok(())
}

#[test]
#[cfg(not(target_os = "windows"))]
fn test_ipc_globbing() -> Result<()> {
    // for side effects
    init_files();
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.ipc";
    let df = LazyFrame::scan_ipc(
        glob.into(),
        ScanArgsIpc {
            n_rows: None,
            cache: true,
            rechunk: false,
        },
    )?
    .collect()?;
    assert_eq!(df.shape(), (54, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0), AnyValue::Int64(45));
    assert_eq!(cal.get(53), AnyValue::Int64(194));

    Ok(())
}

#[test]
#[cfg(not(target_os = "windows"))]
fn test_csv_globbing() -> Result<()> {
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.csv";
    let df = LazyCsvReader::new(glob.into()).finish()?.collect()?;

    // all 5 files * 27 rows
    assert_eq!(df.shape(), (135, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0), AnyValue::Int64(45));
    assert_eq!(cal.get(53), AnyValue::Int64(194));

    Ok(())
}

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

fn slice_at_union(lp_arena: &Arena<ALogicalPlan>, lp: Node) -> bool {
    (&lp_arena).iter(lp).all(|(_, lp)| {
        if let ALogicalPlan::Union { options, .. } = lp {
            options.slice
        } else {
            true
        }
    })
}

#[test]
#[cfg(not(target_os = "windows"))]
fn test_csv_globbing() -> Result<()> {
    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.csv";
    let full_df = LazyCsvReader::new(glob.into()).finish()?.collect()?;

    // all 5 files * 27 rows
    assert_eq!(full_df.shape(), (135, 4));
    let cal = full_df.column("calories")?;
    assert_eq!(cal.get(0), AnyValue::Int64(45));
    assert_eq!(cal.get(53), AnyValue::Int64(194));

    let glob = "../../examples/aggregate_multiple_files_in_chunks/datasets/*.csv";
    let lf = LazyCsvReader::new(glob.into()).finish()?.slice(0, 100);

    let df = lf.clone().collect()?;
    assert_eq!(df.shape(), (100, 4));
    let df = LazyCsvReader::new(glob.into())
        .finish()?
        .slice(20, 60)
        .collect()?;
    assert!(full_df.slice(20, 60).frame_equal(&df));

    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);
    let node = lf.clone().optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_union(&mut lp_arena, node));

    let lf = LazyCsvReader::new(glob.into())
        .finish()?
        .filter(col("sugars_g").lt(lit(1i32)))
        .slice(0, 100);
    let node = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_union(&mut lp_arena, node));

    Ok(())
}

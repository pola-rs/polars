use super::*;
use polars_io::RowCount;

#[test]
fn test_parquet_exec() -> Result<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
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
fn test_parquet_statistics_no_skip() {
    let _guard = SINGLE_LOCK.lock().unwrap();
    init_files();
    let par = true;
    let out = scan_foods_parquet(par)
        .filter(col("calories").gt(lit(0i32)))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (27, 4));

    let out = scan_foods_parquet(par)
        .filter(col("calories").lt(lit(1000i32)))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (27, 4));

    let out = scan_foods_parquet(par)
        .filter(lit(0i32).lt(col("calories")))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (27, 4));

    let out = scan_foods_parquet(par)
        .filter(lit(1000i32).gt(col("calories")))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (27, 4));

    // Or operation
    let out = scan_foods_parquet(par)
        .filter(
            col("sugars_g")
                .lt(lit(0i32))
                .or(col("fats_g").lt(lit(1000.0))),
        )
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (27, 4));
}

#[test]
fn test_parquet_statistics() -> Result<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    init_files();
    std::env::set_var("POLARS_PANIC_IF_PARQUET_PARSED", "1");
    let par = true;

    // Test single predicates
    let out = scan_foods_parquet(par)
        .filter(col("calories").lt(lit(0i32)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    let out = scan_foods_parquet(par)
        .filter(col("calories").gt(lit(1000)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    let out = scan_foods_parquet(par)
        .filter(lit(0i32).gt(col("calories")))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    let out = scan_foods_parquet(par)
        .filter(lit(1000i32).lt(col("calories")))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // Test multiple predicates

    // And operation
    let out = scan_foods_parquet(par)
        .filter(col("calories").lt(lit(0i32)))
        .filter(col("calories").gt(lit(1000)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    let out = scan_foods_parquet(par)
        .filter(col("calories").lt(lit(0i32)))
        .filter(col("calories").gt(lit(1000)))
        .filter(col("calories").lt(lit(50i32)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    let out = scan_foods_parquet(par)
        .filter(
            col("calories")
                .lt(lit(0i32))
                .and(col("fats_g").lt(lit(0.0))),
        )
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // Or operation
    let out = scan_foods_parquet(par)
        .filter(
            col("sugars_g")
                .lt(lit(0i32))
                .or(col("fats_g").gt(lit(1000.0))),
        )
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    std::env::remove_var("POLARS_PANIC_IF_PARQUET_PARSED");

    Ok(())
}

#[test]
#[cfg(not(target_os = "windows"))]
fn test_parquet_globbing() -> Result<()> {
    // for side effects
    init_files();
    let _guard = SINGLE_LOCK.lock().unwrap();
    let glob = "../../examples/datasets/*.parquet";
    let df = LazyFrame::scan_parquet(
        glob.into(),
        ScanArgsParquet {
            n_rows: None,
            cache: true,
            parallel: true,
            rechunk: false,
            row_count: None,
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
    let glob = "../../examples/datasets/*.ipc";
    let df = LazyFrame::scan_ipc(
        glob.into(),
        ScanArgsIpc {
            n_rows: None,
            cache: true,
            rechunk: false,
            row_count: None,
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
    let glob = "../../examples/datasets/*.csv";
    let full_df = LazyCsvReader::new(glob.into()).finish()?.collect()?;

    // all 5 files * 27 rows
    assert_eq!(full_df.shape(), (135, 4));
    let cal = full_df.column("calories")?;
    assert_eq!(cal.get(0), AnyValue::Int64(45));
    assert_eq!(cal.get(53), AnyValue::Int64(194));

    let glob = "../../examples/datasets/*.csv";
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

#[test]
pub fn test_simple_slice() -> Result<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let out = scan_foods_parquet(false).limit(3).collect()?;
    assert_eq!(out.height(), 3);

    Ok(())
}
#[test]
fn test_union_and_agg_projections() -> Result<()> {
    init_files();
    let _guard = SINGLE_LOCK.lock().unwrap();
    // a union vstacks columns and aggscan optimization determines columns to aggregate in a
    // hashmap, if that doesn't set them sorted the vstack will panic.
    let lf1 = LazyFrame::scan_parquet(GLOB_PARQUET.into(), Default::default())?;
    let lf2 = LazyFrame::scan_ipc(GLOB_IPC.into(), Default::default())?;
    let lf3 = LazyCsvReader::new(GLOB_CSV.into()).finish()?;

    for lf in [lf1, lf2, lf3] {
        let lf = lf.filter(col("category").eq(lit("vegetables"))).select([
            col("fats_g").sum().alias("sum"),
            col("fats_g").cast(DataType::Float64).mean().alias("mean"),
            col("fats_g").min().alias("min"),
        ]);

        let out = lf.collect()?;
        assert_eq!(out.shape(), (1, 3));
    }

    Ok(())
}

#[test]
#[cfg(all(feature = "ipc", feature = "csv-file"))]
fn test_slice_filter() -> Result<()> {
    init_files();
    let _guard = SINGLE_LOCK.lock().unwrap();

    // make sure that the slices are not applied before the predicates.
    let len = 5;
    let offset = 3;

    let df1 = scan_foods_csv()
        .filter(col("category").eq(lit("fruit")))
        .slice(offset, len)
        .collect()?;
    let df2 = scan_foods_parquet(false)
        .filter(col("category").eq(lit("fruit")))
        .slice(offset, len)
        .collect()?;
    let df3 = scan_foods_ipc()
        .filter(col("category").eq(lit("fruit")))
        .slice(offset, len)
        .collect()?;

    let df1_ = scan_foods_csv()
        .collect()?
        .lazy()
        .filter(col("category").eq(lit("fruit")))
        .slice(offset, len)
        .collect()?;
    let df2_ = scan_foods_parquet(false)
        .collect()?
        .lazy()
        .filter(col("category").eq(lit("fruit")))
        .slice(offset, len)
        .collect()?;
    let df3_ = scan_foods_ipc()
        .collect()?
        .lazy()
        .filter(col("category").eq(lit("fruit")))
        .slice(offset, len)
        .collect()?;

    assert_eq!(df1.shape(), df1_.shape());
    assert_eq!(df2.shape(), df2_.shape());
    assert_eq!(df3.shape(), df3_.shape());

    Ok(())
}

#[test]
fn skip_rows_and_slice() -> Result<()> {
    let out = LazyCsvReader::new(FOODS_CSV.to_string())
        .with_skip_rows(4)
        .finish()?
        .limit(1)
        .collect()?;
    assert_eq!(out.column("fruit")?.get(0), AnyValue::Utf8("seafood"));
    assert_eq!(out.shape(), (1, 4));
    Ok(())
}

#[test]
fn test_row_count() -> Result<()> {
    for offset in [0u32, 10] {
        let lf = LazyCsvReader::new(FOODS_CSV.to_string())
            .with_row_count(Some(RowCount {
                name: "rc".into(),
                offset,
            }))
            .finish()?;

        assert!(row_count_at_scan(lf.clone()));
        let df = lf.collect()?;
        let rc = df.column("rc")?;
        assert_eq!(
            rc.u32()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let lf = LazyFrame::scan_parquet(
            FOODS_PARQUET.to_string(),
            ScanArgsParquet {
                row_count: Some(RowCount {
                    name: "rc".into(),
                    offset,
                }),
                ..Default::default()
            },
        )?;
        assert!(row_count_at_scan(lf.clone()));
        let df = lf.collect()?;
        let rc = df.column("rc")?;
        assert_eq!(
            rc.u32()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let lf = LazyFrame::scan_ipc(
            FOODS_IPC.to_string(),
            ScanArgsIpc {
                row_count: Some(RowCount {
                    name: "rc".into(),
                    offset,
                }),
                ..Default::default()
            },
        )?;

        assert!(row_count_at_scan(lf.clone()));
        let df = lf.collect()?;
        let rc = df.column("rc")?;
        assert_eq!(
            rc.u32()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );
    }

    Ok(())
}

#[test]
fn scan_predicate_on_set_null_values() -> Result<()> {
    let df = LazyCsvReader::new(FOODS_CSV.into())
        .with_null_values(Some(NullValues::Named(vec![("fats_g".into(), "0".into())])))
        .with_infer_schema_length(Some(0))
        .finish()?
        .select([col("category"), col("fats_g")])
        .filter(col("fats_g").is_null())
        .collect()?;

    assert_eq!(df.shape(), (12, 2));
    Ok(())
}

use polars_io::RowCount;

use super::*;

#[test]
fn test_parquet_exec() -> PolarsResult<()> {
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
fn test_parquet_statistics() -> PolarsResult<()> {
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
fn test_parquet_globbing() -> PolarsResult<()> {
    // for side effects
    init_files();
    let _guard = SINGLE_LOCK.lock().unwrap();
    let glob = "../../examples/datasets/*.parquet";
    let df = LazyFrame::scan_parquet(
        glob,
        ScanArgsParquet {
            n_rows: None,
            cache: true,
            parallel: Default::default(),
            ..Default::default()
        },
    )?
    .collect()?;
    assert_eq!(df.shape(), (54, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0)?, AnyValue::Int64(45));
    assert_eq!(cal.get(53)?, AnyValue::Int64(194));

    Ok(())
}

#[test]
#[cfg(not(target_os = "windows"))]
fn test_ipc_globbing() -> PolarsResult<()> {
    // for side effects
    init_files();
    let glob = "../../examples/datasets/*.ipc";
    let df = LazyFrame::scan_ipc(
        glob,
        ScanArgsIpc {
            n_rows: None,
            cache: true,
            rechunk: false,
            row_count: None,
            memmap: true,
        },
    )?
    .collect()?;
    assert_eq!(df.shape(), (54, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0)?, AnyValue::Int64(45));
    assert_eq!(cal.get(53)?, AnyValue::Int64(194));

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
fn test_csv_globbing() -> PolarsResult<()> {
    let glob = "../../examples/datasets/*.csv";
    let full_df = LazyCsvReader::new(glob).finish()?.collect()?;

    // all 5 files * 27 rows
    assert_eq!(full_df.shape(), (135, 4));
    let cal = full_df.column("calories")?;
    assert_eq!(cal.get(0)?, AnyValue::Int64(45));
    assert_eq!(cal.get(53)?, AnyValue::Int64(194));

    let glob = "../../examples/datasets/*.csv";
    let lf = LazyCsvReader::new(glob).finish()?.slice(0, 100);

    let df = lf.clone().collect()?;
    assert_eq!(df.shape(), (100, 4));
    let df = LazyCsvReader::new(glob).finish()?.slice(20, 60).collect()?;
    assert!(full_df.slice(20, 60).frame_equal(&df));

    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);
    let node = lf.clone().optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_union(&mut lp_arena, node));

    let lf = LazyCsvReader::new(glob)
        .finish()?
        .filter(col("sugars_g").lt(lit(1i32)))
        .slice(0, 100);
    let node = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_union(&mut lp_arena, node));

    Ok(())
}

#[test]
pub fn test_simple_slice() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let out = scan_foods_parquet(false).limit(3).collect()?;
    assert_eq!(out.height(), 3);

    Ok(())
}
#[test]
fn test_union_and_agg_projections() -> PolarsResult<()> {
    init_files();
    let _guard = SINGLE_LOCK.lock().unwrap();
    // a union vstacks columns and aggscan optimization determines columns to aggregate in a
    // hashmap, if that doesn't set them sorted the vstack will panic.
    let lf1 = LazyFrame::scan_parquet(GLOB_PARQUET, Default::default())?;
    let lf2 = LazyFrame::scan_ipc(GLOB_IPC, Default::default())?;
    let lf3 = LazyCsvReader::new(GLOB_CSV).finish()?;

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
fn test_slice_filter() -> PolarsResult<()> {
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
fn skip_rows_and_slice() -> PolarsResult<()> {
    let out = LazyCsvReader::new(FOODS_CSV)
        .with_skip_rows(4)
        .finish()?
        .limit(1)
        .collect()?;
    assert_eq!(out.column("fruit")?.get(0)?, AnyValue::Utf8("seafood"));
    assert_eq!(out.shape(), (1, 4));
    Ok(())
}

#[test]
fn test_row_count_on_files() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    for offset in [0 as IdxSize, 10] {
        let lf = LazyCsvReader::new(FOODS_CSV)
            .with_row_count(Some(RowCount {
                name: "rc".into(),
                offset,
            }))
            .finish()?;

        assert!(row_count_at_scan(lf.clone()));
        let df = lf.collect()?;
        let rc = df.column("rc")?;
        assert_eq!(
            rc.idx()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let lf = LazyFrame::scan_parquet(FOODS_PARQUET, Default::default())?
            .with_row_count("rc", Some(offset));
        assert!(row_count_at_scan(lf.clone()));
        let df = lf.collect()?;
        let rc = df.column("rc")?;
        assert_eq!(
            rc.idx()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let lf =
            LazyFrame::scan_ipc(FOODS_IPC, Default::default())?.with_row_count("rc", Some(offset));

        assert!(row_count_at_scan(lf.clone()));
        let df = lf.clone().collect()?;
        let rc = df.column("rc")?;
        assert_eq!(
            rc.idx()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let out = lf
            .filter(col("rc").gt(lit(-1)))
            .select([col("calories")])
            .collect()?;
        assert!(out.column("calories").is_ok());
        assert_eq!(out.shape(), (27, 1));
    }

    Ok(())
}

#[test]
fn scan_predicate_on_set_null_values() -> PolarsResult<()> {
    let df = LazyCsvReader::new(FOODS_CSV)
        .with_null_values(Some(NullValues::Named(vec![("fats_g".into(), "0".into())])))
        .with_infer_schema_length(Some(0))
        .finish()?
        .select([col("category"), col("fats_g")])
        .filter(col("fats_g").is_null())
        .collect()?;

    assert_eq!(df.shape(), (12, 2));
    Ok(())
}

#[test]
fn scan_anonymous_fn() -> PolarsResult<()> {
    let function = Arc::new(|_scan_opts: AnonymousScanOptions| Ok(fruits_cars()));

    let args = ScanArgsAnonymous {
        schema: Some(fruits_cars().schema()),
        ..ScanArgsAnonymous::default()
    };

    let df = LazyFrame::anonymous_scan(function, args)?.collect()?;

    assert_eq!(df.shape(), (5, 4));
    Ok(())
}

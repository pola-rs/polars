use polars_io::RowIndex;
#[cfg(feature = "is_between")]
use polars_ops::prelude::ClosedInterval;

use super::*;

#[test]
#[cfg(feature = "parquet")]
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
#[cfg(all(feature = "parquet", feature = "is_between"))]
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

    // statistics and `is_between`
    // normal case
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(40, 300, ClosedInterval::Both))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (19, 4));
    // normal case
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(10, 50, ClosedInterval::Both))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (11, 4));
    // edge case: 20 = min(calories) but the right end is closed
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(5, 20, ClosedInterval::Right))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (1, 4));
    // edge case: 200 = max(calories) but the left end is closed
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(200, 250, ClosedInterval::Left))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (3, 4));
    // edge case: left == right but both ends are closed
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(200, 200, ClosedInterval::Both))
        .collect()
        .unwrap();
    assert_eq!(out.shape(), (3, 4));

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
#[cfg(all(feature = "parquet", feature = "is_between"))]
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

    // issue: 13427
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_in(lit(Series::new("", [0, 500]))))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // statistics and `is_between`
    // 15 < min(calories)=20
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(5, 15, ClosedInterval::Both))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // 300 > max(calories)=200
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(300, 500, ClosedInterval::Both))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // 20 == min(calories) but right end is open
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(5, 20, ClosedInterval::Left))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // 20 == min(calories) but both  ends are open
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(5, 20, ClosedInterval::None))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // 200 == max(calories) but left end is open
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(200, 250, ClosedInterval::Right))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // 200 == max(calories) but both ends are open
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(200, 250, ClosedInterval::None))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // between(100, 40) is impossible
    let out = scan_foods_parquet(par)
        .filter(col("calories").is_between(100, 40, ClosedInterval::Both))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // with strings
    let out = scan_foods_parquet(par)
        .filter(col("category").is_between(lit("yams"), lit("zest"), ClosedInterval::Both))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // with strings
    let out = scan_foods_parquet(par)
        .filter(col("category").is_between(lit("dairy"), lit("eggs"), ClosedInterval::Both))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    let out = scan_foods_parquet(par)
        .filter(lit(1000i32).lt(col("calories")))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(a > b) => a <= b
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").gt(5)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(a >= b) => a < b
    // note that min(calories)=20
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").gt_eq(20)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(a < b) => a >= b
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").lt(250)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(a <= b) => a > b
    // note that max(calories)=200
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").lt_eq(200)))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(a == b) => a != b
    // note that proteins_g=10 for all rows
    let out = scan_nutri_score_null_column_parquet(par)
        .filter(not(col("proteins_g").eq(10)))
        .collect()?;
    assert_eq!(out.shape(), (0, 6));

    // not(a != b) => a == b
    // note that proteins_g=10 for all rows
    let out = scan_nutri_score_null_column_parquet(par)
        .filter(not(col("proteins_g").neq(5)))
        .collect()?;
    assert_eq!(out.shape(), (0, 6));

    // not(col(c) is between [a, b]) => col(c) < a or col(c) > b
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").is_between(
            20,
            200,
            ClosedInterval::Both,
        )))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(col(c) is between [a, b[) => col(c) < a or col(c) >= b
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").is_between(
            20,
            201,
            ClosedInterval::Left,
        )))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(col(c) is between ]a, b]) => col(c) <= a or col(c) > b
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").is_between(
            19,
            200,
            ClosedInterval::Right,
        )))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(col(c) is between ]a, b]) => col(c) <= a or col(c) > b
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").is_between(
            19,
            200,
            ClosedInterval::Right,
        )))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not(col(c) is between ]a, b[) => col(c) <= a or col(c) >= b
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").is_between(
            19,
            201,
            ClosedInterval::None,
        )))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not (a or b) => not(a) and not(b)
    // note that not(fats_g <= 9) is possible; not(calories > 5) should allow us skip the rg
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").gt(5).or(col("fats_g").lt_eq(9))))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // not (a and b) => not(a) or not(b)
    let out = scan_foods_parquet(par)
        .filter(not(col("calories").gt(5).and(col("fats_g").lt_eq(12))))
        .collect()?;
    assert_eq!(out.shape(), (0, 4));

    // is_not_null
    let out = scan_nutri_score_null_column_parquet(par)
        .filter(col("nutri_score").is_not_null())
        .collect()?;
    assert_eq!(out.shape(), (0, 6));

    // not(is_null) (~pl.col('nutri_score').is_null())
    let out = scan_nutri_score_null_column_parquet(par)
        .filter(not(col("nutri_score").is_null()))
        .collect()?;
    assert_eq!(out.shape(), (0, 6));

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
    let glob = "../../examples/datasets/foods*.parquet";
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
fn test_scan_parquet_limit_9001() {
    init_files();
    let path = GLOB_PARQUET;
    let args = ScanArgsParquet {
        n_rows: Some(10000),
        cache: false,
        rechunk: true,
        ..Default::default()
    };
    let q = LazyFrame::scan_parquet(path, args).unwrap().limit(3);
    let IRPlan {
        lp_top, lp_arena, ..
    } = q.to_alp_optimized().unwrap();
    (&lp_arena).iter(lp_top).all(|(_, lp)| match lp {
        IR::Union { options, .. } => {
            let sliced = options.slice.unwrap();
            sliced.1 == 3
        },
        IR::Scan { file_options, .. } => file_options.slice == Some((0, 3)),
        _ => true,
    });
}

#[test]
#[cfg(not(target_os = "windows"))]
fn test_ipc_globbing() -> PolarsResult<()> {
    // for side effects
    init_files();
    let glob = "../../examples/datasets/foods*.ipc";
    let df = LazyFrame::scan_ipc(
        glob,
        ScanArgsIpc {
            n_rows: None,
            cache: true,
            rechunk: false,
            row_index: None,
            memory_map: true,
            cloud_options: None,
            hive_options: Default::default(),
            include_file_paths: None,
        },
    )?
    .collect()?;
    assert_eq!(df.shape(), (54, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0)?, AnyValue::Int64(45));
    assert_eq!(cal.get(53)?, AnyValue::Int64(194));

    Ok(())
}

fn slice_at_union(lp_arena: &Arena<IR>, lp: Node) -> bool {
    (&lp_arena).iter(lp).all(|(_, lp)| {
        if let IR::Union { options, .. } = lp {
            options.slice.is_some()
        } else {
            true
        }
    })
}

#[test]
fn test_csv_globbing() -> PolarsResult<()> {
    let glob = "../../examples/datasets/foods*.csv";
    let full_df = LazyCsvReader::new(glob).finish()?.collect()?;

    // all 5 files * 27 rows
    assert_eq!(full_df.shape(), (135, 4));
    let cal = full_df.column("calories")?;
    assert_eq!(cal.get(0)?, AnyValue::Int64(45));
    assert_eq!(cal.get(53)?, AnyValue::Int64(194));

    let glob = "../../examples/datasets/foods*.csv";
    let lf = LazyCsvReader::new(glob).finish()?.slice(0, 100);

    let df = lf.clone().collect()?;
    assert_eq!(df, full_df.slice(0, 100));
    let df = LazyCsvReader::new(glob).finish()?.slice(20, 60).collect()?;
    assert_eq!(df, full_df.slice(20, 60));

    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);
    let node = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_union(&lp_arena, node));

    let lf = LazyCsvReader::new(glob)
        .finish()?
        .filter(col("sugars_g").lt(lit(1i32)))
        .slice(0, 100);
    let node = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_union(&lp_arena, node));

    Ok(())
}

#[test]
#[cfg(feature = "json")]
fn test_ndjson_globbing() -> PolarsResult<()> {
    // for side effects
    init_files();
    let glob = "../../examples/datasets/foods*.ndjson";
    let df = LazyJsonLineReader::new(glob).finish()?.collect()?;
    assert_eq!(df.shape(), (54, 4));
    let cal = df.column("calories")?;
    assert_eq!(cal.get(0)?, AnyValue::Int64(45));
    assert_eq!(cal.get(53)?, AnyValue::Int64(194));

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
#[cfg(all(feature = "ipc", feature = "csv"))]
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
    assert_eq!(out.column("fruit")?.get(0)?, AnyValue::String("seafood"));
    assert_eq!(out.shape(), (1, 4));
    Ok(())
}

#[test]
fn test_row_index_on_files() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    for offset in [0 as IdxSize, 10] {
        let lf = LazyCsvReader::new(FOODS_CSV)
            .with_row_index(Some(RowIndex {
                name: Arc::from("index"),
                offset,
            }))
            .finish()?;

        assert!(row_index_at_scan(lf.clone()));
        let df = lf.collect()?;
        let idx = df.column("index")?;
        assert_eq!(
            idx.idx()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let lf = LazyFrame::scan_parquet(FOODS_PARQUET, Default::default())?
            .with_row_index("index", Some(offset));
        assert!(row_index_at_scan(lf.clone()));
        let df = lf.collect()?;
        let idx = df.column("index")?;
        assert_eq!(
            idx.idx()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let lf = LazyFrame::scan_ipc(FOODS_IPC, Default::default())?
            .with_row_index("index", Some(offset));

        assert!(row_index_at_scan(lf.clone()));
        let df = lf.clone().collect()?;
        let idx = df.column("index")?;
        assert_eq!(
            idx.idx()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let out = lf
            .filter(col("index").gt(lit(-1)))
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
fn scan_anonymous_fn_with_options() -> PolarsResult<()> {
    struct MyScan {}

    impl AnonymousScan for MyScan {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn allows_projection_pushdown(&self) -> bool {
            true
        }

        fn scan(&self, scan_opts: AnonymousScanArgs) -> PolarsResult<DataFrame> {
            assert_eq!(scan_opts.with_columns.clone().unwrap().len(), 2);
            assert_eq!(scan_opts.n_rows, Some(3));
            let out = fruits_cars().select(scan_opts.with_columns.unwrap().as_ref())?;
            Ok(out.slice(0, scan_opts.n_rows.unwrap()))
        }
    }

    let function = Arc::new(MyScan {});

    let args = ScanArgsAnonymous {
        schema: Some(Arc::new(fruits_cars().schema())),
        ..ScanArgsAnonymous::default()
    };

    let q = LazyFrame::anonymous_scan(function, args)?
        .with_column((col("A") * lit(2)).alias("A2"))
        .select([col("A2"), col("fruits")])
        .limit(3);

    let df = q.collect()?;

    assert_eq!(df.shape(), (3, 2));
    Ok(())
}

#[test]
#[cfg(feature = "dtype-full")]
fn scan_small_dtypes() -> PolarsResult<()> {
    let small_dt = vec![
        DataType::Int8,
        DataType::UInt8,
        DataType::Int16,
        DataType::UInt16,
    ];
    for dt in small_dt {
        let df = LazyCsvReader::new(FOODS_CSV)
            .with_has_header(true)
            .with_dtype_overwrite(Some(Arc::new(Schema::from_iter([Field::new(
                "sugars_g",
                dt.clone(),
            )]))))
            .finish()?
            .select(&[col("sugars_g")])
            .collect()?;

        assert_eq!(df.dtypes(), &[dt]);
    }
    Ok(())
}

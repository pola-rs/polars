#[cfg(feature = "is_between")]
use polars_defs::expr::ClosedInterval;
use polars_defs::join::JoinCoalesce;
use polars_io::RowIndex;
use polars_utils::pl_path::PlRefPath;
use polars_utils::slice_enum::Slice;

use super::*;
use crate::dsl;

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
    unsafe { std::env::set_var("POLARS_PANIC_IF_PARQUET_PARSED", "1") };
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
        .filter(col("calories").is_in(lit(Series::new("".into(), [0, 500])), false))
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

    unsafe { std::env::remove_var("POLARS_PANIC_IF_PARQUET_PARSED") };

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
        PlRefPath::new(glob),
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
    let q = LazyFrame::scan_parquet(PlRefPath::new(path), args)
        .unwrap()
        .limit(3);
    let IRPlan {
        lp_top, lp_arena, ..
    } = q.to_alp_optimized().unwrap();
    lp_arena.iter(lp_top).all(|(_, lp)| match lp {
        IR::Union { options, .. } => {
            let sliced = options.slice.unwrap();
            sliced.1 == 3
        },
        IR::Scan {
            unified_scan_args, ..
        } => unified_scan_args.pre_slice == Some(Slice::Positive { offset: 0, len: 3 }),
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
        PlRefPath::new(glob),
        Default::default(),
        UnifiedScanArgs {
            cache: true,
            glob: true,
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

fn slice_at_union(lp_arena: &Arena<IR>, lp: Node) -> bool {
    lp_arena.iter(lp).all(|(_, lp)| {
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
    let full_df = LazyCsvReader::new(PlRefPath::new(glob))
        .finish()?
        .collect()?;

    // all 5 files * 27 rows
    assert_eq!(full_df.shape(), (135, 4));
    let cal = full_df.column("calories")?;
    assert_eq!(cal.get(0)?, AnyValue::Int64(45));
    assert_eq!(cal.get(53)?, AnyValue::Int64(194));

    let glob = "../../examples/datasets/foods*.csv";
    let lf = LazyCsvReader::new(PlRefPath::new(glob))
        .finish()?
        .slice(0, 100);

    let df = lf.clone().collect()?;
    assert_eq!(df, full_df.slice(0, 100));
    let df = LazyCsvReader::new(PlRefPath::new(glob))
        .finish()?
        .slice(20, 60)
        .collect()?;
    assert_eq!(df, full_df.slice(20, 60));

    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);
    let node = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_union(&lp_arena, node));

    let lf = LazyCsvReader::new(PlRefPath::new(glob))
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
    let df = LazyJsonLineReader::new(PlRefPath::new(glob))
        .finish()?
        .collect()?;
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
    let lf1: LazyFrame = DslBuilder::scan_parquet(
        ScanSources::Paths(FromIterator::from_iter([PlRefPath::new(GLOB_PARQUET)])),
        ParquetOptions::default(),
        UnifiedScanArgs {
            extra_columns_policy: ExtraColumnsPolicy::Ignore,
            ..Default::default()
        },
    )
    .unwrap()
    .build()
    .into();

    let lf2: LazyFrame = DslBuilder::scan_ipc(
        ScanSources::Paths(FromIterator::from_iter([PlRefPath::new(GLOB_IPC)])),
        IpcScanOptions {
            ..Default::default()
        },
        UnifiedScanArgs {
            extra_columns_policy: ExtraColumnsPolicy::Ignore,
            ..Default::default()
        },
    )
    .unwrap()
    .build()
    .into();

    let lf3: LazyFrame = DslBuilder::scan_csv(
        ScanSources::Paths(FromIterator::from_iter([PlRefPath::new(GLOB_CSV)])),
        CsvReadOptions::default(),
        UnifiedScanArgs {
            extra_columns_policy: ExtraColumnsPolicy::Ignore,
            ..Default::default()
        },
    )
    .unwrap()
    .build()
    .into();

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
    let out = LazyCsvReader::new(PlRefPath::new(FOODS_CSV))
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
        let lf = LazyCsvReader::new(PlRefPath::new(FOODS_CSV))
            .with_row_index(Some(RowIndex {
                name: PlSmallStr::from_static("index"),
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

        let lf = LazyFrame::scan_parquet(PlRefPath::new(FOODS_PARQUET), Default::default())?
            .with_row_index("index", Some(offset));
        assert!(row_index_at_scan(lf.clone()));
        let df = lf.collect()?;
        let idx = df.column("index")?;
        assert_eq!(
            idx.idx()?.into_no_null_iter().collect::<Vec<_>>(),
            (offset..27 + offset).collect::<Vec<_>>()
        );

        let lf = LazyFrame::scan_ipc(
            PlRefPath::new(FOODS_IPC),
            Default::default(),
            Default::default(),
        )?
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
    let df = LazyCsvReader::new(PlRefPath::new(FOODS_CSV))
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
            let out = fruits_cars().select(scan_opts.with_columns.unwrap().iter().cloned())?;
            Ok(out.slice(0, scan_opts.n_rows.unwrap()))
        }
    }

    let function = Arc::new(MyScan {});

    let args = ScanArgsAnonymous {
        schema: Some(fruits_cars().schema().clone()),
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
fn scan_anonymous_fn_count() -> PolarsResult<()> {
    struct MyScan {}

    impl AnonymousScan for MyScan {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn allows_projection_pushdown(&self) -> bool {
            true
        }

        fn scan(&self, scan_opts: AnonymousScanArgs) -> PolarsResult<DataFrame> {
            assert_eq!(scan_opts.with_columns.as_deref(), Some(&["A".into()][..]));

            Ok(fruits_cars()
                .select(scan_opts.with_columns.unwrap().iter().cloned())
                .unwrap())
        }
    }

    let function = Arc::new(MyScan {});

    let args = ScanArgsAnonymous {
        schema: Some(fruits_cars().schema().clone()),
        ..ScanArgsAnonymous::default()
    };

    let df = LazyFrame::anonymous_scan(function, args)?
        .select(&[dsl::len()])
        .collect()
        .unwrap();

    assert_eq!(df.columns().len(), 1);
    assert_eq!(df.columns()[0].len(), 1);
    assert_eq!(
        df.columns()[0]
            .cast(&DataType::UInt32)
            .unwrap()
            .as_materialized_series()
            .first(),
        Scalar::new(DataType::UInt32, AnyValue::UInt32(5))
    );

    Ok(())
}

/// Regression test helpers for #29441.
///
/// The row index of a scan belongs to the scan *node*, not to the schema of the
/// file it reads. That schema is resolved once per source and shared between all
/// scan nodes over those sources, and it is cached across collections, so a
/// node's logical columns must never end up in it.
#[cfg(any(
    feature = "parquet",
    feature = "csv",
    feature = "ipc",
    feature = "json"
))]
fn assert_scan_file_schema(q: LazyFrame, file_columns: &[&str]) {
    let plan = q.to_alp().unwrap();
    let mut n_scans = 0;

    for (_, ir) in plan.lp_arena.iter(plan.lp_top) {
        let IR::Scan {
            file_info,
            unified_scan_args,
            ..
        } = ir
        else {
            continue;
        };
        n_scans += 1;

        let names: Vec<_> = file_info.schema.iter_names().cloned().collect();
        let names = match &unified_scan_args.row_index {
            // The scan node prepends its own row index.
            Some(ri) => {
                assert_eq!(names[0], ri.name);
                &names[1..]
            },
            None => &names[..],
        };
        assert_eq!(
            names,
            file_columns
                .iter()
                .map(|s| (*s).into())
                .collect::<Vec<PlSmallStr>>()
        );
    }

    assert!(n_scans > 0, "expected a scan node in the plan");
}

/// Row order of a join is not part of its contract.
#[cfg(any(
    feature = "parquet",
    feature = "csv",
    feature = "ipc",
    feature = "json"
))]
fn sort_all(df: DataFrame) -> DataFrame {
    df.sort(df.get_column_names(), SortMultipleOptions::default())
        .unwrap()
}

/// Asserts that `make_lf` produces a `LazyFrame` that stays reusable: a query
/// built from a `LazyFrame` that has already been collected in another query
/// must give the same result as the same query built from a fresh scan, and
/// collecting a query must not leave state behind in the plan of a later query.
#[cfg(any(
    feature = "parquet",
    feature = "csv",
    feature = "ipc",
    feature = "json"
))]
fn assert_scan_stays_reusable(
    make_lf: &dyn Fn(OptFlags) -> PolarsResult<LazyFrame>,
    file_columns: &[&str],
) -> PolarsResult<()> {
    type Builder = dyn Fn(LazyFrame) -> PolarsResult<LazyFrame>;

    /// Joins `lf` with its own `with_row_index` version: both scan the same
    /// sources, only one of them carries a row index.
    fn join_own_row_index(lf: LazyFrame) -> PolarsResult<LazyFrame> {
        let idx = lf
            .clone()
            .select([col("a")])
            .with_row_index("i", None)
            .select([col("i")]);
        idx.join(
            lf.with_row_index("i", None),
            [col("i")],
            [col("i")],
            JoinArgs::new(JoinType::Full).with_coalesce(JoinCoalesce::CoalesceColumns),
        )
    }

    let builders: Vec<(&str, Box<Builder>)> = vec![
        ("collect", Box::new(Ok)),
        (
            "collect_after_drop",
            Box::new(|lf| Ok(lf.drop(cols(["b"])))),
        ),
        (
            "collect_with_row_index",
            Box::new(|lf| Ok(lf.with_row_index("i", None))),
        ),
        (
            "collect_with_offset_row_index",
            Box::new(|lf| Ok(lf.with_row_index("i", Some(10)))),
        ),
        (
            "collect_self_join",
            Box::new(|lf| {
                lf.clone()
                    .join(lf, [col("a")], [col("a")], JoinArgs::new(JoinType::Inner))
            }),
        ),
        ("join_with_own_row_index", Box::new(join_own_row_index)),
        (
            "dropped_join_with_own_row_index",
            Box::new(|lf| join_own_row_index(lf.drop(cols(["b"])))),
        ),
    ];

    let mut opt_flag_sets = vec![
        OptFlags::default(),
        OptFlags::empty(),
        OptFlags::default().difference(OptFlags::PROJECTION_PUSHDOWN),
    ];
    if cfg!(feature = "cse") {
        opt_flag_sets.push(OptFlags::default().difference(OptFlags::COMM_SUBPLAN_ELIM));
    }

    for flags in opt_flag_sets {
        for (name, build) in &builders {
            // A freshly created scan is the reference the reused one must match.
            let expected = sort_all(build(make_lf(flags)?)?.collect()?);

            let reused = make_lf(flags)?;
            assert_scan_file_schema(reused.clone(), file_columns);

            // Querying the `LazyFrame` repeatedly, and querying it after it has
            // already been used in another query, must give that same result.
            for _ in 0..2 {
                let q = build(reused.clone())?;
                assert_scan_file_schema(q.clone(), file_columns);
                assert_eq!(
                    sort_all(q.collect()?),
                    expected,
                    "query '{name}' changed after the LazyFrame was reused"
                );
            }

            assert_scan_file_schema(reused, file_columns);
        }
    }

    Ok(())
}

#[cfg(feature = "parquet")]
#[test]
fn test_scan_parquet_reuse_29441() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();

    let path = std::env::temp_dir().join(format!("polars_29441_{}.parquet", std::process::id()));
    {
        let mut df = df!("a" => [1i64, 2, 3], "b" => [4i64, 5, 6])?;
        let f = std::fs::File::create(&path)?;
        ParquetWriter::new(f).finish(&mut df)?;
    }

    let make_lf = |flags: OptFlags| {
        LazyFrame::scan_parquet(
            PlRefPath::new(path.to_str().unwrap()),
            ScanArgsParquet::default(),
        )
        .map(|lf| lf.with_optimizations(flags))
    };

    let mut out = assert_scan_stays_reusable(&make_lf, &["a", "b"]);

    // `allow_missing_columns` turns the leaked row index into a silently
    // all-null column instead of an error (see the issue), so check that path
    // as well.
    if out.is_ok() {
        let make_lf = |flags: OptFlags| {
            LazyFrame::scan_parquet(
                PlRefPath::new(path.to_str().unwrap()),
                ScanArgsParquet {
                    allow_missing_columns: true,
                    ..Default::default()
                },
            )
            .map(|lf| lf.with_optimizations(flags))
        };
        out = assert_scan_stays_reusable(&make_lf, &["a", "b"]);
    }

    std::fs::remove_file(&path)?;
    out
}

#[cfg(feature = "csv")]
#[test]
fn test_scan_csv_reuse_29441() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();

    let path = std::env::temp_dir().join(format!("polars_29441_{}.csv", std::process::id()));
    std::fs::write(&path, "a,b\n1,4\n2,5\n3,6\n")?;

    let make_lf = |flags: OptFlags| {
        LazyCsvReader::new(PlRefPath::new(path.to_str().unwrap()))
            .finish()
            .map(|lf| lf.with_optimizations(flags))
    };

    let out = assert_scan_stays_reusable(&make_lf, &["a", "b"]);

    std::fs::remove_file(&path)?;
    out
}

#[cfg(feature = "ipc")]
#[test]
fn test_scan_ipc_reuse_29441() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();

    let path = std::env::temp_dir().join(format!("polars_29441_{}.ipc", std::process::id()));
    {
        let mut df = df!("a" => [1i64, 2, 3], "b" => [4i64, 5, 6])?;
        let f = std::fs::File::create(&path)?;
        IpcWriter::new(f).finish(&mut df)?;
    }
    let path = PlRefPath::new(path.to_str().unwrap());

    let make_lf = |flags: OptFlags| {
        LazyFrame::scan_ipc(path.clone(), Default::default(), Default::default())
            .map(|lf| lf.with_optimizations(flags))
    };

    assert_scan_stays_reusable(&make_lf, &["a", "b"])
}

#[cfg(feature = "json")]
#[test]
fn test_scan_ndjson_reuse_29441() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();

    let path = std::env::temp_dir().join(format!("polars_29441_{}.ndjson", std::process::id()));
    {
        let mut df = df!("a" => [1i64, 2, 3], "b" => [4i64, 5, 6])?;
        let f = std::fs::File::create(&path)?;
        JsonWriter::new(f).finish(&mut df)?;
    }

    let make_lf = |flags: OptFlags| {
        LazyJsonLineReader::new(PlRefPath::new(path.to_str().unwrap()))
            .finish()
            .map(|lf| lf.with_optimizations(flags))
    };

    assert_scan_stays_reusable(&make_lf, &["a", "b"])
}

/// A row count that a scan node knows upfront (an Iceberg snapshot reports
/// physical and deleted rows) belongs to that node. Another scan node over the
/// same sources must not be served it through the shared, cached file info.
#[cfg(feature = "parquet")]
#[test]
fn test_scan_row_count_is_not_shared() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();

    let path =
        std::env::temp_dir().join(format!("polars_row_count_{}.parquet", std::process::id()));
    {
        let mut df = df!("a" => [1i64, 2, 3])?;
        let f = std::fs::File::create(&path)?;
        ParquetWriter::new(f).finish(&mut df)?;
    }
    let path = PlRefPath::new(path.to_str().unwrap());

    fn with_row_count(mut lf: LazyFrame, row_count: (u64, u64)) -> LazyFrame {
        match &mut lf.logical_plan {
            DslPlan::Scan {
                unified_scan_args,
                cached_ir,
                ..
            } => {
                assert_eq!(unified_scan_args.row_count, None);
                unified_scan_args.row_count = Some(row_count);
                *cached_ir.lock().unwrap() = None;
            },
            other => panic!("expected a scan, got {other:?}"),
        }
        lf
    }

    let known = with_row_count(
        LazyFrame::scan_parquet(path.clone(), ScanArgsParquet::default())?,
        (3, 0),
    );
    let other = with_row_count(
        LazyFrame::scan_parquet(path, ScanArgsParquet::default())?,
        (100, 50),
    );

    // Both nodes are resolved within a single conversion.
    let q = known.join(
        other,
        [col("a")],
        [col("a")],
        JoinArgs::new(JoinType::Inner),
    )?;

    let plan = q.clone().to_alp().unwrap();
    let mut rows: Vec<Option<u64>> = plan
        .lp_arena
        .iter(plan.lp_top)
        .filter_map(|(_, ir)| match ir {
            IR::Scan { file_info, .. } => Some(match file_info.stats.rows {
                polars_plan::plans::Card::Exact(v) => Some(v),
                _ => None,
            }),
            _ => None,
        })
        .collect();
    rows.sort();
    assert_eq!(rows, [Some(3), Some(50)]);

    // The overlay only replaces the row estimate: the column statistics that the
    // footers provided must survive on every node.
    for (_, ir) in plan.lp_arena.iter(plan.lp_top) {
        if let IR::Scan { file_info, .. } = ir {
            assert!(file_info.stats.column("a").is_some());
        }
    }

    // And the query itself is unaffected by the made-up counts.
    assert_eq!(
        sort_all(q.collect()?)
            .column("a")?
            .i64()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );

    Ok(())
}

/// A row index whose name collides with a column of the file must be an error:
/// the row index belongs to the scan node, so it may not silently replace the
/// column of the same name that the file provides.
#[cfg(all(feature = "parquet", feature = "csv"))]
#[test]
fn test_row_index_name_in_file_errors() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();

    let parquet =
        std::env::temp_dir().join(format!("polars_row_index_{}.parquet", std::process::id()));
    {
        let mut df = df!("a" => [1i64, 2, 3], "b" => [4i64, 5, 6])?;
        let f = std::fs::File::create(&parquet)?;
        ParquetWriter::new(f).finish(&mut df)?;
    }
    let csv = std::env::temp_dir().join(format!("polars_row_index_{}.csv", std::process::id()));
    std::fs::write(&csv, "a,b\n1,4\n2,5\n3,6\n")?;

    let paths: Vec<(&str, PlRefPath)> = vec![
        ("parquet", PlRefPath::new(parquet.to_str().unwrap())),
        ("csv", PlRefPath::new(csv.to_str().unwrap())),
    ];

    for (name, path) in paths {
        let lf = if name == "parquet" {
            LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
        } else {
            LazyCsvReader::new(path).finish()?
        };

        let err = lf.clone().with_row_index("a", None).collect().unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot add row_index with name 'a': column already exists in file"),
            "{name}: {err}"
        );

        // The file's own `a` column is unaffected.
        let out = lf.select([col("a")]).collect()?;
        assert_eq!(
            out.column("a")?
                .i64()?
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
    }

    std::fs::remove_file(&parquet)?;
    std::fs::remove_file(&csv)?;
    Ok(())
}

/// Every part of a scan node that is resolved from its sources must belong to
/// that node: two scans of the same file with different options must each keep
/// their own schema, in the same query and across queries.
#[cfg(feature = "csv")]
#[test]
fn test_scan_csv_options_are_not_shared() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();

    let path = std::env::temp_dir().join(format!("polars_scan_opts_{}.csv", std::process::id()));
    std::fs::write(&path, "1,4\n4,5\n3,6\n")?;
    let path = PlRefPath::new(path.to_str().unwrap());

    // `has_header` changes the inferred schema: the first line is read as
    // column names or as data.
    let with_header = LazyCsvReader::new(path.clone())
        .with_has_header(true)
        .finish()?;
    let no_header = LazyCsvReader::new(path.clone())
        .with_has_header(false)
        .finish()?;

    // Both scans are resolved within a single conversion.
    let q = with_header
        .clone()
        .join(
            no_header.clone(),
            [col("1")],
            [col("column_1")],
            JoinArgs::new(JoinType::Inner),
        )?
        .select([col("1"), col("column_0")]);

    let plan = q.clone().to_alp().unwrap();
    let mut schemas: Vec<Vec<PlSmallStr>> = plan
        .lp_arena
        .iter(plan.lp_top)
        .filter_map(|(_, ir)| match ir {
            IR::Scan { file_info, .. } => Some(file_info.schema.iter_names().cloned().collect()),
            _ => None,
        })
        .collect();
    schemas.sort();
    assert_eq!(schemas, [["1", "4"], ["column_0", "column_1"]]);

    let out = q.collect()?;
    assert_eq!(out.get_column_names(), &["1", "column_0"]);
    assert_eq!(
        out.column("1")?
            .i64()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        [4]
    );
    assert_eq!(
        out.column("column_0")?
            .i64()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        [1]
    );

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
        let df = LazyCsvReader::new(PlRefPath::new(FOODS_CSV))
            .with_has_header(true)
            .with_dtype_overwrite(Some(Arc::new(Schema::from_iter([Field::new(
                "sugars_g".into(),
                dt.clone(),
            )]))))
            .finish()?
            .select(&[col("sugars_g")])
            .collect()?;

        assert_eq!(df.dtypes(), &[dt]);
    }
    Ok(())
}

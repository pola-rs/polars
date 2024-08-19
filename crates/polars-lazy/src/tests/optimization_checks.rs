use super::*;

#[cfg(feature = "parquet")]
pub(crate) fn row_index_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    (&lp_arena).iter(lp).any(|(_, lp)| {
        use IR::*;
        matches!(
            lp,
            Scan {
                file_options: FileScanOptions {
                    row_index: Some(_),
                    ..
                },
                ..
            }
        )
    })
}

pub(crate) fn predicate_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    (&lp_arena).iter(lp).any(|(_, lp)| {
        use IR::*;
        matches!(
            lp,
            DataFrameScan {
                filter: Some(_),
                ..
            } | Scan {
                predicate: Some(_),
                ..
            }
        )
    })
}

pub(crate) fn predicate_at_all_scans(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    (&lp_arena).iter(lp).all(|(_, lp)| {
        use IR::*;
        matches!(
            lp,
            DataFrameScan {
                filter: Some(_),
                ..
            } | Scan {
                predicate: Some(_),
                ..
            }
        )
    })
}

#[cfg(feature = "streaming")]
pub(crate) fn is_pipeline(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    matches!(
        lp_arena.get(lp),
        IR::MapFunction {
            function: FunctionIR::Pipeline { .. },
            ..
        }
    )
}

#[cfg(feature = "streaming")]
pub(crate) fn has_pipeline(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    (&lp_arena).iter(lp).any(|(_, lp)| {
        matches!(
            lp,
            IR::MapFunction {
                function: FunctionIR::Pipeline { .. },
                ..
            }
        )
    })
}

#[cfg(any(feature = "parquet", feature = "csv"))]
fn slice_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    (&lp_arena).iter(lp).any(|(_, lp)| {
        use IR::*;
        match lp {
            Scan { file_options, .. } => file_options.slice.is_some(),
            _ => false,
        }
    })
}

#[test]
fn test_pred_pd_1() -> PolarsResult<()> {
    let df = fruits_cars();

    let q = df
        .clone()
        .lazy()
        .select([col("A"), col("B")])
        .filter(col("A").gt(lit(1)));

    assert!(predicate_at_scan(q));

    // Check if we understand that we can unwrap the alias.
    let q = df
        .clone()
        .lazy()
        .select([col("A").alias("C"), col("B")])
        .filter(col("C").gt(lit(1)));

    assert!(predicate_at_scan(q));

    // Check if we pass hstack.
    let q = df
        .clone()
        .lazy()
        .with_columns([col("A").alias("C"), col("B")])
        .filter(col("B").gt(lit(1)));

    assert!(predicate_at_scan(q));

    // check if we do not pass slice
    let q = df.lazy().limit(10).filter(col("B").gt(lit(1)));

    assert!(!predicate_at_scan(q));

    Ok(())
}

#[test]
fn test_no_left_join_pass() -> PolarsResult<()> {
    let df1 = df![
        "foo" => ["abc", "def", "ghi"],
        "idx1" => [0, 0, 1],
    ]?;
    let df2 = df![
        "bar" => [5, 6],
        "idx2" => [0, 1],
    ]?;

    let out = df1
        .lazy()
        .join(
            df2.lazy(),
            [col("idx1")],
            [col("idx2")],
            JoinType::Left.into(),
        )
        .filter(col("bar").eq(lit(5i32)))
        .collect()?;

    let expected = df![
        "foo" => ["abc", "def"],
        "idx1" => [0, 0],
        "bar" => [5, 5],
    ]?;

    assert!(out.equals(&expected));
    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
pub fn test_simple_slice() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(3);

    assert!(slice_at_scan(q.clone()));
    let out = q.collect()?;
    assert_eq!(out.height(), 3);

    let q = scan_foods_parquet(false)
        .select([col("category"), col("calories").alias("bar")])
        .limit(3);
    assert!(slice_at_scan(q.clone()));
    let out = q.collect()?;
    assert_eq!(out.height(), 3);

    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
#[cfg(feature = "cse")]
pub fn test_slice_pushdown_join() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q1 = scan_foods_parquet(false).limit(3);
    let q2 = scan_foods_parquet(false);

    let q = q1
        .join(
            q2,
            [col("category")],
            [col("category")],
            JoinType::Left.into(),
        )
        .slice(1, 3)
        // this inserts a cache and blocks slice pushdown
        .with_comm_subplan_elim(false);
    // test if optimization continued beyond the join node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use IR::*;
        match lp {
            Join { options, .. } => options.args.slice == Some((1, 3)),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 7));

    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
pub fn test_slice_pushdown_group_by() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(100);

    let q = q
        .group_by([col("category")])
        .agg([col("calories").sum()])
        .slice(1, 3);

    // test if optimization continued beyond the group_by node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use IR::*;
        match lp {
            GroupBy { options, .. } => options.slice == Some((1, 3)),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 2));

    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
pub fn test_slice_pushdown_sort() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(100);

    let q = q
        .sort(["category"], SortMultipleOptions::default())
        .slice(1, 3);

    // test if optimization continued beyond the sort node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use IR::*;
        match lp {
            Sort { slice, .. } => *slice == Some((1, 3)),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 4));

    Ok(())
}

#[test]
#[cfg(feature = "dtype-i16")]
pub fn test_predicate_block_cast() -> PolarsResult<()> {
    let df = df![
        "value" => [10, 20, 30, 40]
    ]?;

    let lf1 = df
        .clone()
        .lazy()
        .with_column(col("value").cast(DataType::Int16) * lit(0.1).cast(DataType::Float32))
        .filter(col("value").lt(lit(2.5f32)));

    let lf2 = df
        .lazy()
        .select([col("value").cast(DataType::Int16) * lit(0.1).cast(DataType::Float32)])
        .filter(col("value").lt(lit(2.5f32)));

    for lf in [lf1, lf2] {
        assert!(!predicate_at_scan(lf.clone()));

        let out = lf.collect()?;
        let s = out.column("value").unwrap();
        assert_eq!(s, &Series::new("value", [1.0f32, 2.0]));
    }

    Ok(())
}

#[test]
fn test_lazy_filter_and_rename() {
    let df = load_df();
    let lf = df
        .clone()
        .lazy()
        .rename(["a"], ["x"])
        .filter(col("x").map(
            |s: Series| Ok(Some(s.gt(3)?.into_series())),
            GetOutput::from_type(DataType::Boolean),
        ))
        .select([col("x")]);

    let correct = df! {
        "x" => &[4, 5]
    }
    .unwrap();
    assert!(lf.collect().unwrap().equals(&correct));

    // now we check if the column is rename or added when we don't select
    let lf = df.lazy().rename(["a"], ["x"]).filter(col("x").map(
        |s: Series| Ok(Some(s.gt(3)?.into_series())),
        GetOutput::from_type(DataType::Boolean),
    ));
    // the rename function should not interfere with the predicate pushdown
    assert!(predicate_at_scan(lf.clone()));

    assert_eq!(lf.collect().unwrap().get_column_names(), &["x", "b", "c"]);
}

#[test]
fn test_with_row_index_opts() -> PolarsResult<()> {
    let df = df![
        "a" => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]?;

    let out = df
        .clone()
        .lazy()
        .with_row_index("index", None)
        .tail(5)
        .collect()?;
    let expected = df![
        "index" => [5 as IdxSize, 6, 7, 8, 9],
        "a" => [5, 6, 7, 8, 9],
    ]?;

    assert!(out.equals(&expected));
    let out = df
        .clone()
        .lazy()
        .with_row_index("index", None)
        .slice(1, 2)
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[1, 2]
    );

    let out = df
        .clone()
        .lazy()
        .with_row_index("index", None)
        .filter(col("a").eq(lit(3i32)))
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[3]
    );

    let out = df
        .clone()
        .lazy()
        .slice(1, 2)
        .with_row_index("index", None)
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[0, 1]
    );

    let out = df
        .lazy()
        .filter(col("a").eq(lit(3i32)))
        .with_row_index("index", None)
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[0]
    );

    Ok(())
}

#[cfg(all(feature = "concat_str", feature = "strings"))]
#[test]
fn test_string_addition_to_concat_str() -> PolarsResult<()> {
    let df = df![
        "a"=> ["a"],
        "b"=> ["b"],
    ]?;

    let q = df
        .lazy()
        .select([lit("foo") + col("a") + col("b") + lit("bar")]);

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let root = q.clone().optimize(&mut lp_arena, &mut expr_arena)?;
    let lp = lp_arena.get(root);
    let mut exprs = lp.get_exprs();
    let e = exprs.pop().unwrap();
    if let AExpr::Function { input, .. } = expr_arena.get(e.node()) {
        // the concat_str has the 4 expressions as input
        assert_eq!(input.len(), 4);
    } else {
        panic!()
    }

    let out = q.collect()?;
    let s = out.column("literal")?;
    assert_eq!(s.get(0)?, AnyValue::String("fooabbar"));

    Ok(())
}
#[test]
fn test_with_column_prune() -> PolarsResult<()> {
    // don't
    let df = df![
        "c0" => [0],
        "c1" => [0],
        "c2" => [0],
    ]?;
    let (mut expr_arena, mut lp_arena) = get_arenas();

    // only a single expression pruned and only one column selection
    let q = df
        .clone()
        .lazy()
        .with_columns([col("c0"), col("c1").alias("c4")])
        .select([col("c1"), col("c4")]);
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    (&lp_arena).iter(lp).for_each(|(_, lp)| {
        use IR::*;
        match lp {
            DataFrameScan { output_schema, .. } => {
                let projection = output_schema.as_ref().unwrap();
                assert_eq!(projection.len(), 1);
                let name = projection.get_at_index(0).unwrap().0;
                assert_eq!(name, "c1");
            },
            HStack { exprs, .. } => {
                assert_eq!(exprs.len(), 1);
            },
            _ => {},
        };
    });

    // whole `with_columns` pruned
    let mut q = df.lazy().with_column(col("c0")).select([col("c1")]);

    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();

    // check if with_column is pruned
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use IR::*;

        matches!(lp, SimpleProjection { .. } | DataFrameScan { .. })
    }));
    assert_eq!(
        q.collect_schema().unwrap().as_ref(),
        &Schema::from_iter([Field::new("c1", DataType::Int32)])
    );
    Ok(())
}

#[test]
#[cfg(feature = "csv")]
fn test_slice_at_scan_group_by() -> PolarsResult<()> {
    let ldf = scan_foods_csv();

    // this tests if slice pushdown restarts aggregation nodes (it did not)
    let q = ldf
        .slice(0, 5)
        .filter(col("calories").lt(lit(10)))
        .group_by([col("calories")])
        .agg([col("fats_g").first()])
        .select([col("fats_g")]);

    assert!(slice_at_scan(q));
    Ok(())
}

#[test]
fn test_flatten_unions() -> PolarsResult<()> {
    let (mut expr_arena, mut lp_arena) = get_arenas();

    let lf = df! {
        "a" => [1,2,3,4,5],
    }
    .unwrap()
    .lazy();

    let args = UnionArgs {
        rechunk: false,
        parallel: true,
        ..Default::default()
    };
    let lf2 = concat(&[lf.clone(), lf.clone()], args).unwrap();
    let lf3 = concat(&[lf.clone(), lf.clone(), lf], args).unwrap();
    let lf4 = concat(&[lf2, lf3], args).unwrap();
    let root = lf4.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    let lp = lp_arena.get(root);
    match lp {
        IR::Union { inputs, .. } => {
            // we make sure that the nested unions are flattened into a single union
            assert_eq!(inputs.len(), 5);
        },
        _ => panic!(),
    }
    Ok(())
}

fn num_occurrences(s: &str, needle: &str) -> usize {
    let mut i = 0;
    let mut num = 0;

    while let Some(n) = s[i..].find(needle) {
        i += n + 1;
        num += 1;
    }

    num
}

#[test]
fn test_cluster_with_columns() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo") * lit(2.0)])
        .with_columns([col("bar") / lit(1.5)]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.clone().to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert_eq!(num_occurrences(&unoptimized, "WITH_COLUMNS"), 2);
    assert_eq!(num_occurrences(&optimized, "WITH_COLUMNS"), 1);

    Ok(())
}

#[test]
fn test_cluster_with_columns_dependency() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo").alias("buzz")])
        .with_columns([col("buzz")]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.clone().to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert_eq!(num_occurrences(&unoptimized, "WITH_COLUMNS"), 2);
    assert_eq!(num_occurrences(&optimized, "WITH_COLUMNS"), 2);

    Ok(())
}

#[test]
fn test_cluster_with_columns_partial() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo").alias("buzz")])
        .with_columns([col("buzz"), col("foo") * lit(2.0)]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.clone().to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert!(unoptimized.contains(r#"[col("buzz"), [(col("foo")) * (2.0)]]"#));
    assert!(unoptimized.contains(r#"[col("foo").alias("buzz")]"#));
    assert!(optimized.contains(r#"[col("buzz")]"#));
    assert!(optimized.contains(r#"[col("foo").alias("buzz"), [(col("foo")) * (2.0)]]"#));

    Ok(())
}

#[test]
fn test_cluster_with_columns_chain() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo").alias("foo1")])
        .with_columns([col("foo").alias("foo2")])
        .with_columns([col("foo").alias("foo3")])
        .with_columns([col("foo").alias("foo4")]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.clone().to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert_eq!(num_occurrences(&unoptimized, "WITH_COLUMNS"), 4);
    assert_eq!(num_occurrences(&optimized, "WITH_COLUMNS"), 1);

    Ok(())
}

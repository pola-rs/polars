use super::*;

pub(crate) fn row_count_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    (&lp_arena).iter(lp).any(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            CsvScan {
                options:
                    CsvParserOptions {
                        row_count: Some(_), ..
                    },
                ..
            }
            | ParquetScan {
                options:
                    ParquetOptions {
                        row_count: Some(_), ..
                    },
                ..
            }
            | IpcScan {
                options:
                    IpcScanOptionsInner {
                        row_count: Some(_), ..
                    },
                ..
            } => true,
            _ => false,
        }
    })
}

pub(crate) fn predicate_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    (&lp_arena).iter(lp).any(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            DataFrameScan {
                selection: Some(_), ..
            }
            | CsvScan {
                predicate: Some(_), ..
            }
            | ParquetScan {
                predicate: Some(_), ..
            }
            | IpcScan {
                predicate: Some(_), ..
            } => true,
            _ => false,
        }
    })
}

fn slice_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    (&lp_arena).iter(lp).any(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            CsvScan { options, .. } => options.n_rows.is_some(),
            ParquetScan { options, .. } => options.n_rows.is_some(),
            IpcScan { options, .. } => options.n_rows.is_some(),
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

    // check if we understand that we can unwrap the alias
    let q = df
        .clone()
        .lazy()
        .select([col("A").alias("C"), col("B")])
        .filter(col("C").gt(lit(1)));

    assert!(predicate_at_scan(q));

    // check if we pass hstack
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
        .join(df2.lazy(), [col("idx1")], [col("idx2")], JoinType::Left)
        .filter(col("bar").eq(lit(5i32)))
        .collect()?;

    let expected = df![
        "foo" => ["abc", "def"],
        "idx1" => [0, 0],
        "bar" => [5, 5],
    ]?;

    assert!(out.frame_equal(&expected));
    Ok(())
}

#[test]
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
#[cfg(feature = "cse")]
pub fn test_slice_pushdown_join() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q1 = scan_foods_parquet(false).limit(3);
    let q2 = scan_foods_parquet(false);

    let q = q1
        .join(q2, [col("category")], [col("category")], JoinType::Left)
        .slice(1, 3)
        // this inserts a cache and blocks slice pushdown
        .with_common_subplan_elimination(false);
    // test if optimization continued beyond the join node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            Join { options, .. } => options.slice == Some((1, 3)),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 7));

    Ok(())
}

#[test]
pub fn test_slice_pushdown_groupby() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(100);

    let q = q
        .groupby([col("category")])
        .agg([col("calories").sum()])
        .slice(1, 3);

    // test if optimization continued beyond the groupby node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            Aggregate { options, .. } => options.slice == Some((1, 3)),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 2));

    Ok(())
}

#[test]
pub fn test_slice_pushdown_sort() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(100);

    let q = q.sort("category", SortOptions::default()).slice(1, 3);

    // test if optimization continued beyond the sort node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            Sort { args, .. } => args.slice == Some((1, 3)),
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
        .with_column(col("value").cast(DataType::Int16) * lit(0.1f32))
        .filter(col("value").lt(lit(2.5f32)));

    let lf2 = df
        .lazy()
        .select([col("value").cast(DataType::Int16) * lit(0.1f32)])
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
            |s: Series| Ok(s.gt(3)?.into_series()),
            GetOutput::from_type(DataType::Boolean),
        ))
        .select([col("x")]);

    let correct = df! {
        "x" => &[4, 5]
    }
    .unwrap();
    assert!(lf.collect().unwrap().frame_equal(&correct));

    // now we check if the column is rename or added when we don't select
    let lf = df.lazy().rename(["a"], ["x"]).filter(col("x").map(
        |s: Series| Ok(s.gt(3)?.into_series()),
        GetOutput::from_type(DataType::Boolean),
    ));
    // the rename function should not interfere with the predicate pushdown
    assert!(predicate_at_scan(lf.clone()));

    assert_eq!(lf.collect().unwrap().get_column_names(), &["x", "b", "c"]);
}

#[test]
fn test_with_row_count_opts() -> PolarsResult<()> {
    let df = df![
        "a" => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]?;

    let out = df
        .clone()
        .lazy()
        .with_row_count("row_nr", None)
        .tail(5)
        .collect()?;
    let expected = df![
        "row_nr" => [5 as IdxSize, 6, 7, 8, 9],
        "a" => [5, 6, 7, 8, 9],
    ]?;

    assert!(out.frame_equal(&expected));
    let out = df
        .clone()
        .lazy()
        .with_row_count("row_nr", None)
        .slice(1, 2)
        .collect()?;
    assert_eq!(
        out.column("row_nr")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[1, 2]
    );

    let out = df
        .clone()
        .lazy()
        .with_row_count("row_nr", None)
        .filter(col("a").eq(lit(3i32)))
        .collect()?;
    assert_eq!(
        out.column("row_nr")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[3]
    );

    let out = df
        .clone()
        .lazy()
        .slice(1, 2)
        .with_row_count("row_nr", None)
        .collect()?;
    assert_eq!(
        out.column("row_nr")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[0, 1]
    );

    let out = df
        .lazy()
        .filter(col("a").eq(lit(3i32)))
        .with_row_count("row_nr", None)
        .collect()?;
    assert_eq!(
        out.column("row_nr")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[0]
    );

    Ok(())
}

#[test]
fn test_groupby_ternary_literal_predicate() -> PolarsResult<()> {
    let df = df![
        "a" => [1, 2, 3],
        "b" => [1, 2, 3]
    ]?;

    for predicate in [true, false] {
        let q = df
            .clone()
            .lazy()
            .groupby(["a"])
            .agg([when(lit(predicate))
                .then(col("b").sum())
                .otherwise(NULL.lit())])
            .sort("a", Default::default());

        let (mut expr_arena, mut lp_arena) = get_arenas();
        let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();

        (&lp_arena).iter(lp).any(|(_, lp)| {
            use ALogicalPlan::*;
            match lp {
                Aggregate { aggs, .. } => {
                    for node in aggs {
                        // we should not have a ternary expression anymore
                        assert!(!matches!(expr_arena.get(*node), AExpr::Ternary { .. }));
                    }
                    false
                }
                _ => false,
            }
        });

        let out = q.collect()?;
        let b = out.column("b")?;
        let b = b.i32()?;
        if predicate {
            assert_eq!(Vec::from(b), &[Some(1), Some(2), Some(3)]);
        } else {
            assert_eq!(b.null_count(), 3);
        };
    }

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
    let expr_node = exprs.pop().unwrap();
    if let AExpr::Function { input, .. } = expr_arena.get(expr_node) {
        // the concat_str has the 4 expressions as input
        assert_eq!(input.len(), 4);
    } else {
        panic!()
    }

    let out = q.collect()?;
    let s = out.column("literal")?;
    assert_eq!(s.get(0), AnyValue::Utf8("fooabbar"));

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
        use ALogicalPlan::*;
        match lp {
            DataFrameScan { projection, .. } => {
                let projection = projection.as_ref().unwrap();
                let projection = projection.as_slice();
                assert_eq!(projection.len(), 1);
                let name = &projection[0];
                assert_eq!(name, "c1");
            }
            HStack { exprs, .. } => {
                assert_eq!(exprs.len(), 1);
            }
            _ => {}
        };
    });

    // whole `with_columns` pruned
    let q = df.lazy().with_column(col("c0")).select([col("c1")]);

    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();

    // check if with_column is pruned
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            ALogicalPlan::MapFunction {
                function: FunctionNode::FastProjection { .. },
                ..
            }
            | DataFrameScan { .. } => true,
            _ => false,
        }
    }));
    assert_eq!(
        q.schema().unwrap().as_ref(),
        &Schema::from([Field::new("c1", DataType::Int32)].into_iter())
    );
    Ok(())
}

#[test]
fn test_slice_at_scan_groupby() -> PolarsResult<()> {
    let ldf = scan_foods_csv();

    // this tests if slice pushdown restarts aggregation nodes (it did not)
    let q = ldf
        .slice(0, 5)
        .filter(col("calories").lt(lit(10)))
        .groupby([col("calories")])
        .agg([col("fats_g").first()])
        .select([col("fats_g")]);

    assert!(slice_at_scan(q));
    Ok(())
}

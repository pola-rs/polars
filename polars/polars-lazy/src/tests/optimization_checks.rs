use super::*;

fn get_arenas() -> (Arena<AExpr>, Arena<ALogicalPlan>) {
    let expr_arena = Arena::with_capacity(16);
    let lp_arena = Arena::with_capacity(8);
    (expr_arena, lp_arena)
}

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
                    IpcScanOptions {
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

fn slice_at_scan(lp_arena: &Arena<ALogicalPlan>, lp: Node) -> bool {
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
fn test_pred_pd_1() -> Result<()> {
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
fn test_no_left_join_pass() -> Result<()> {
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
pub fn test_simple_slice() -> Result<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let q = scan_foods_parquet(false).limit(3);

    let root = q.clone().optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(slice_at_scan(&lp_arena, root));
    let out = q.collect()?;
    assert_eq!(out.height(), 3);

    let q = scan_foods_parquet(false)
        .select([col("category"), col("calories").alias("bar")])
        .limit(3);
    assert!(slice_at_scan(&lp_arena, root));
    let out = q.collect()?;
    assert_eq!(out.height(), 3);

    Ok(())
}

#[test]
pub fn test_predicate_block_cast() -> Result<()> {
    let df = df![
        "value" => [10, 20, 30, 40]
    ]?;

    let lf1 = df
        .clone()
        .lazy()
        .with_column(col("value") * lit(0.1f32))
        .filter(col("value").lt(lit(2.5f32)));

    let lf2 = df
        .lazy()
        .select([col("value") * lit(0.1f32)])
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
            |s: Series| Ok(s.gt(3).into_series()),
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
        |s: Series| Ok(s.gt(3).into_series()),
        GetOutput::from_type(DataType::Boolean),
    ));
    // the rename function should not interfere with the predicate pushdown
    assert!(predicate_at_scan(lf.clone()));

    assert_eq!(lf.collect().unwrap().get_column_names(), &["x", "b", "c"]);
}

#[test]
fn test_with_row_count_opts() -> Result<()> {
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
        "row_nr" => [5_u32, 6, 7, 8, 9],
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
            .u32()?
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
            .u32()?
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
            .u32()?
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
            .u32()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[0]
    );

    Ok(())
}

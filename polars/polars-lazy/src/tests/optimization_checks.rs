use super::*;
use crate::tests::queries::load_df;

fn get_arenas() -> (Arena<AExpr>, Arena<ALogicalPlan>) {
    let expr_arena = Arena::with_capacity(16);
    let lp_arena = Arena::with_capacity(8);
    (expr_arena, lp_arena)
}

pub(crate) fn predicate_at_scan(lp_arena: &Arena<ALogicalPlan>, lp: Node) -> bool {
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

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = df
        .clone()
        .lazy()
        .select([col("A"), col("B")])
        .filter(col("A").gt(lit(1)))
        .optimize(&mut lp_arena, &mut expr_arena)?;

    assert!(predicate_at_scan(&lp_arena, lp));

    // check if we understand that we can unwrap the alias
    let lp = df
        .clone()
        .lazy()
        .select([col("A").alias("C"), col("B")])
        .filter(col("C").gt(lit(1)))
        .optimize(&mut lp_arena, &mut expr_arena)?;

    assert!(predicate_at_scan(&lp_arena, lp));

    // check if we pass hstack
    let lp = df
        .clone()
        .lazy()
        .with_columns([col("A").alias("C"), col("B")])
        .filter(col("B").gt(lit(1)))
        .optimize(&mut lp_arena, &mut expr_arena)?;

    assert!(predicate_at_scan(&lp_arena, lp));

    // check if we do not pass slice
    let lp = df
        .lazy()
        .limit(10)
        .filter(col("B").gt(lit(1)))
        .optimize(&mut lp_arena, &mut expr_arena)?;

    assert!(!predicate_at_scan(&lp_arena, lp));

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

    dbg!(out);
    Ok(())
}

#[test]
pub fn test_simple_slice() -> Result<()> {
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
        // make sure that the predicate is not pushed down
        let (mut expr_arena, mut lp_arena) = get_arenas();
        let root = lf.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
        assert!(!predicate_at_scan(&mut lp_arena, root));

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
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let root = lf.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!(predicate_at_scan(&mut lp_arena, root));

    assert_eq!(lf.collect().unwrap().get_column_names(), &["x", "b", "c"]);
}

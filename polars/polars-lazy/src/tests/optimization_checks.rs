use super::*;

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

    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);
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
    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);
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

use super::*;

fn projection_at_scan(lp_arena: &Arena<ALogicalPlan>, lp: Node) -> bool {
    (&lp_arena).iter(lp).all(|(_, lp)| {
        if let ALogicalPlan::DataFrameScan { selection, .. } = lp {
            selection.is_some()
        } else {
            true
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

    assert!(projection_at_scan(&lp_arena, lp));

    // check if we understand that we can unwrap the alias
    let lp = df
        .lazy()
        .select([col("A").alias("C"), col("B")])
        .filter(col("C").gt(lit(1)))
        .optimize(&mut lp_arena, &mut expr_arena)?;

    assert!(projection_at_scan(&lp_arena, lp));

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

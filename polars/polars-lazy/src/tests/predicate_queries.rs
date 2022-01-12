use super::*;

#[test]
fn test_foo() -> Result<()> {
    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);

    let lf = scan_foods_parquet(false).select([col("calories").alias("bar")]);

    // this produces a predicate with two root columns, this test if we can
    // deal with multiple roots
    let lf = lf.filter(col("bar").gt(lit(45i32)));
    let lf = lf.filter(col("bar").lt(lit(110i32)));

    // also check if all predicates are combined and pushed down
    let root = lf.optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(predicate_at_scan(&mut lp_arena, root));
    // and that we don't have any filter node
    assert!(!(&lp_arena)
        .iter(root)
        .any(|(_, lp)| matches!(lp, ALogicalPlan::Selection { .. })));

    Ok(())
}

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
fn test_window_separation() -> Result<()> {
    let df = fruits_cars();

    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);
    let lp = df
        .clone()
        .lazy()
        .select([

            col("A").over([col("B")]).alias("1"),
            col("B").over([col("B")]).alias("2"),
            col("B").over([col("A")]).alias("3"),
            col("fruits")
        ])
        .optimize(&mut lp_arena, &mut expr_arena)?;

    let lp = node_to_lp(lp, &mut expr_arena, &mut lp_arena);
    dbg!(lp);

    Ok(())


}
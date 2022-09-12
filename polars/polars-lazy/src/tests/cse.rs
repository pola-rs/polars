use super::*;

fn cached_before_root(q: LazyFrame) {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    for input in lp_arena.get(lp).get_inputs() {
        assert!(matches!(lp_arena.get(input), ALogicalPlan::Cache { .. }));
    }
}

#[test]
fn test_cse_self_joins() -> Result<()> {
    let lf = scan_foods_ipc();

    let lf = lf.clone().with_column(col("category").str().to_uppercase());

    let lf = lf.clone().left_join(lf, col("fats_g"), col("fats_g"));
    cached_before_root(lf);

    Ok(())
}

#[test]
fn test_cse_unions() -> Result<()> {
    let lf = scan_foods_ipc();

    let lf1 = lf.clone().with_column(col("category").str().to_uppercase());

    let lf = concat(&[lf1.clone(), lf, lf1], false)?;
    cached_before_root(lf);

    Ok(())
}

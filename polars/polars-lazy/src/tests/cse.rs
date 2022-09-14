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

    let lf = lf
        .clone()
        .left_join(lf, col("fats_g"), col("fats_g"))
        .with_common_subplan_elimination(true);

    println!("{}", lf.to_dot(true).unwrap());
    cached_before_root(lf);

    Ok(())
}

#[test]
fn test_cse_unions() -> Result<()> {
    let lf = scan_foods_ipc();

    let lf1 = lf.clone().with_column(col("category").str().to_uppercase());

    let lf = concat(&[lf1.clone(), lf, lf1], false)?.with_common_subplan_elimination(true);
    cached_before_root(lf);

    Ok(())
}

#[test]
fn test_cse_cache_union_projection_pd() -> Result<()> {
    let q = df![
        "a" => [1],
        "b" => [2],
        "c" => [3],
    ]?
    .lazy();

    let q1 = q.clone().filter(col("a").eq(lit(1))).select([col("a")]);
    let q2 = q
        .clone()
        .filter(col("a").eq(lit(1)))
        .select([col("a"), col("b")]);
    let q = q1
        .left_join(q2, col("a"), col("a"))
        .with_common_subplan_elimination(true);

    // check that the projection of a is not done before the cache
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            DataFrameScan {
                projection: Some(projection),
                ..
            } => {
                let mut projection = (**projection).clone();
                projection.sort_unstable();
                projection == vec!["a".to_string(), "b".to_string()]
            },
            DataFrameScan { .. } => false,
            _ => true,
        }
    }));

    Ok(())
}

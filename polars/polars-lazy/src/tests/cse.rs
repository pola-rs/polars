use std::collections::BTreeSet;

use super::*;

fn cached_before_root(q: LazyFrame) {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    for input in lp_arena.get(lp).get_inputs() {
        assert!(matches!(lp_arena.get(input), ALogicalPlan::Cache { .. }));
    }
}

#[test]
fn test_cse_self_joins() -> PolarsResult<()> {
    let lf = scan_foods_ipc();

    let lf = lf.clone().with_column(col("category").str().to_uppercase());

    let lf = lf
        .clone()
        .left_join(lf, col("fats_g"), col("fats_g"))
        .with_common_subplan_elimination(true);

    cached_before_root(lf);

    Ok(())
}

#[test]
fn test_cse_unions() -> PolarsResult<()> {
    let lf = scan_foods_ipc();

    let lf1 = lf.clone().with_column(col("category").str().to_uppercase());

    let lf = concat(&[lf1.clone(), lf, lf1], false, false)?
        .select([col("category"), col("fats_g")])
        .with_common_subplan_elimination(true);

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = lf.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!((&lp_arena).iter(lp).all(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            IpcScan { options, .. } => {
                if let Some(columns) = &options.with_columns {
                    columns.len() == 2
                } else {
                    false
                }
            }
            _ => true,
        }
    }));
    let out = lf.collect()?;
    assert_eq!(out.get_column_names(), &["category", "fats_g"]);

    Ok(())
}

#[test]
fn test_cse_cache_union_projection_pd() -> PolarsResult<()> {
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
            } => projection.as_ref() == &vec!["a".to_string(), "b".to_string()],
            DataFrameScan { .. } => false,
            _ => true,
        }
    }));

    Ok(())
}

#[test]
fn test_cse_union2_4925() -> PolarsResult<()> {
    let lf1 = df![
        "ts" => [1],
        "sym" => ["a"],
        "c" => [true],
    ]?
    .lazy();

    let lf2 = df![
        "ts" => [1],
        "d" => [3],
    ]?
    .lazy();

    let lf1 = concat(&[lf1.clone(), lf1], false, false)?;
    let lf2 = concat(&[lf2.clone(), lf2], false, false)?;

    let q = lf1.inner_join(lf2, col("ts"), col("ts")).select([
        col("ts"),
        col("sym"),
        col("d") / col("c"),
    ]);

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    // ensure we get two different caches
    // and ensure that every cache only has 1 hit.
    let cache_ids = (&lp_arena)
        .iter(lp)
        .flat_map(|(_, lp)| {
            use ALogicalPlan::*;
            match lp {
                Cache { id, count, .. } => {
                    assert_eq!(*count, 1);
                    Some(*id)
                }
                _ => None,
            }
        })
        .collect::<BTreeSet<_>>();

    assert_eq!(cache_ids.len(), 2);

    Ok(())
}

#[test]
fn test_cse_joins_4954() -> PolarsResult<()> {
    let x = df![
        "a"=> [1],
        "b"=> [1],
        "c"=> [1],
    ]?
    .lazy();

    let y = df![
        "a"=> [1],
        "b"=> [1],
    ]?
    .lazy();

    let z = df![
        "a"=> [1],
    ]?
    .lazy();

    let a = x.left_join(z.clone(), col("a"), col("a"));
    let b = y.left_join(z.clone(), col("a"), col("a"));
    let c = a.join(
        b,
        &[col("a"), col("b")],
        &[col("a"), col("b")],
        JoinType::Left,
    );

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = c.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    // ensure we get only one cache and the it is not above the join
    // and ensure that every cache only has 1 hit.
    let cache_ids = (&lp_arena)
        .iter(lp)
        .flat_map(|(_, lp)| {
            use ALogicalPlan::*;
            match lp {
                Cache { id, count, input } => {
                    assert_eq!(*count, 1);
                    assert!(matches!(
                        lp_arena.get(*input),
                        ALogicalPlan::DataFrameScan { .. }
                    ));

                    Some(*id)
                }
                _ => None,
            }
        })
        .collect::<BTreeSet<_>>();

    assert_eq!(cache_ids.len(), 1);

    Ok(())
}
#[test]
#[cfg(feature = "semi_anti_join")]
fn test_cache_with_partial_projection() -> PolarsResult<()> {
    let lf1 = df![
        "id" => ["a"],
        "x" => [1],
        "freq" => [2]
    ]?
    .lazy();

    let lf2 = df![
        "id" => ["a"]
    ]?
    .lazy();

    let q = lf2
        .join(
            lf1.clone().select([col("id"), col("freq")]),
            [col("id")],
            [col("id")],
            JoinType::Semi,
        )
        .join(
            lf1.clone().filter(col("x").neq(lit(8))),
            [col("id")],
            [col("id")],
            JoinType::Semi,
        )
        .join(
            lf1.clone().filter(col("x").neq(lit(8))),
            [col("id")],
            [col("id")],
            JoinType::Semi,
        );

    let q = q.with_common_subplan_elimination(true);

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    // ensure we get two different caches
    // and ensure that every cache only has 1 hit.
    let cache_ids = (&lp_arena)
        .iter(lp)
        .flat_map(|(_, lp)| {
            use ALogicalPlan::*;
            match lp {
                Cache { id, .. } => Some(*id),
                _ => None,
            }
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(cache_ids.len(), 2);

    Ok(())
}

#[test]
#[cfg(feature = "cross_join")]
fn test_cse_columns_projections() -> PolarsResult<()> {
    let right = df![
        "A" => [1, 2],
        "B" => [3, 4],
        "D" => [5, 6]
    ]?
    .lazy();

    let left = df![
        "C" => [3, 4],
    ]?
    .lazy();

    let left = left.cross_join(right.clone().select([col("A")]));
    let q = left.join(
        right.rename(["B"], ["C"]),
        [col("A"), col("C")],
        [col("A"), col("C")],
        JoinType::Left,
    );

    let out = q.collect()?;

    assert_eq!(out.get_column_names(), &["C", "A", "D"]);

    Ok(())
}

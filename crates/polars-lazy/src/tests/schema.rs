use super::*;

#[test]
fn test_schema_update_after_projection_pd() -> PolarsResult<()> {
    let df = df![
        "a" => [1],
        "b" => [1],
        "c" => [1],
    ]?;

    let q = df
        .lazy()
        .with_column(col("a").implode(true))
        .explode(
            by_name(["a"], true, false),
            ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            },
        )
        .select([cols(["a", "b"]).as_expr()]);

    // run optimizations
    // Get the explode node
    let IRPlan {
        lp_top,
        lp_arena,
        expr_arena: _,
    } = q.to_alp_optimized()?;

    // assert the schema has been corrected with the projection pushdown run
    let lp = lp_arena.get(lp_top);
    assert!(matches!(
        lp,
        IR::MapFunction {
            function: FunctionIR::Explode { .. },
            ..
        }
    ));

    let schema = lp.schema(&lp_arena).into_owned();
    let mut expected = Schema::default();
    expected.with_column("a".into(), DataType::Int32);
    expected.with_column("b".into(), DataType::Int32);
    assert_eq!(schema.as_ref(), &expected);

    Ok(())
}

fn cached_arena_len(lf: &LazyFrame) -> usize {
    lf.cached_arena
        .lock()
        .unwrap()
        .as_ref()
        .map_or(0, |a| a.len())
}

#[test]
fn test_collect_schema_deep_chain_reuses_parent_ir() -> PolarsResult<()> {
    // Calling `collect_schema` after every step of a chain should only convert the new step,
    // not the whole plan again.
    let mut lf = df!["a" => [1, 2, 3]]?.lazy();
    lf.collect_schema()?;
    for i in 0..100 {
        let mut child = lf.with_columns([(col("a") + lit(i)).alias("a")]);
        // The child shares the arenas holding the IR of its parent.
        let before = cached_arena_len(&child);
        assert!(before > 0);
        let schema = child.collect_schema()?;
        assert_eq!(schema.get("a"), Some(&DataType::Int32));
        let added = cached_arena_len(&child).saturating_sub(before);
        assert!(
            added <= 16,
            "step {i} added {added} nodes, the input was converted again"
        );
        lf = child;
    }
    Ok(())
}

#[test]
fn test_collect_schema_shared_arena_garbage_is_bounded() -> PolarsResult<()> {
    let mut base = df!["a" => [1, 2, 3]]?.lazy();
    base.collect_schema()?;
    for i in 0..2000 {
        let mut child = base.clone().filter(col("a").gt(lit(i)));
        child.collect_schema()?;
    }
    assert!(cached_arena_len(&base) < 4096);
    assert_eq!(base.collect_schema()?.len(), 1);
    Ok(())
}

#[test]
fn test_collect_schema_shared_arena_error_and_empty_select() -> PolarsResult<()> {
    let mut lf = df!["a" => [1, 2, 3]]?.lazy().with_column(lit(1).alias("b"));
    assert_eq!(lf.collect_schema()?.len(), 2);

    // A failing child must not invalidate the cached IR of its parent.
    let mut failing = lf.clone().select([col("missing")]);
    assert!(failing.collect_schema().is_err());
    assert_eq!(lf.collect_schema()?.len(), 2);

    // An empty projection must not replace the (shared) input node.
    let mut empty = lf.clone().select(Vec::<Expr>::new());
    assert_eq!(empty.collect_schema()?.len(), 0);
    assert_eq!(lf.collect_schema()?.len(), 2);

    let mut child = lf.clone().with_column(lit(2).alias("c"));
    assert_eq!(child.collect_schema()?.len(), 3);
    assert_eq!(child.collect()?.shape(), (3, 3));
    Ok(())
}

#[test]
fn test_collect_schema_shared_arena_self_join() -> PolarsResult<()> {
    // Both join inputs have an IR node in the same cached arena; only one of them can be reused.
    let mut lf = scan_foods_csv().select([col("category"), col("calories")]);
    lf.collect_schema()?;
    let mut left = lf.clone().filter(col("calories").gt(lit(50)));
    left.collect_schema()?;
    let mut right = lf.clone().with_column(col("calories").alias("calories2"));
    right.collect_schema()?;

    let mut joined = left.clone().join(
        right.clone(),
        [col("category")],
        [col("category")],
        JoinArgs::new(JoinType::Inner),
    )?;
    let schema = joined.collect_schema()?;
    assert_eq!(schema.len(), 4);
    let out = joined.collect()?;
    let expected = left
        .collect()?
        .lazy()
        .join(
            right.collect()?.lazy(),
            [col("category")],
            [col("category")],
            JoinArgs::new(JoinType::Inner),
        )?
        .collect()?;
    assert_eq!(out.shape(), expected.shape());
    Ok(())
}

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
        .with_column(col("a").implode())
        .explode([col("a")])
        .select([cols(["a", "b"])]);

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
    let mut expected = Schema::new();
    expected.with_column("a".into(), DataType::Int32);
    expected.with_column("b".into(), DataType::Int32);
    assert_eq!(schema.as_ref(), &expected);

    Ok(())
}

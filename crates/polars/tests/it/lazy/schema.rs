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
    let (node, lp_arena, _expr_arena) = q.to_alp_optimized()?;
    // get the explode node
    let input = lp_arena.get(node).get_inputs()[0];
    // assert the schema has been corrected with the projection pushdown run
    lp_arena.get(input);
    let schema = lp_arena.get(input).schema(&lp_arena).into_owned();
    let mut expected = Schema::new();
    expected.with_column("a".into(), DataType::Int32);
    expected.with_column("b".into(), DataType::Int32);
    assert_eq!(schema.as_ref(), &expected);

    Ok(())
}

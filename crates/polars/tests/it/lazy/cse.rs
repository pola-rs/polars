use super::*;

#[test]
#[cfg(feature = "semi_anti_join")]
fn test_cse_union_schema_6504() -> PolarsResult<()> {
    use polars_core::df;
    let q1: LazyFrame = df![
        "a" => [1],
        "b" => [2],
    ]?
    .lazy();
    let q2: LazyFrame = df![
        "b" => [1],
    ]?
    .lazy();

    let q3 = q2
        .join(q1.clone(), [col("b")], [col("b")], JoinType::Anti.into())
        .with_column(lit(0).alias("a"))
        .select([col("a"), col("b")]);

    let out = concat(
        [q1, q3],
        UnionArgs {
            rechunk: false,
            parallel: false,
            ..Default::default()
        },
    )
    .unwrap()
    .with_comm_subplan_elim(true)
    .collect()?;
    let expected = df![
        "a" => [1, 0],
        "b" => [2, 1],
    ]?;
    assert!(out.equals(&expected));

    Ok(())
}

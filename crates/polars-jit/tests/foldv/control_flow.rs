use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_applies_the_selected_conditional_branch_for_each_row() -> PolarsResult<()> {
    // Negative values retain the current state; positive values are added.
    let value = col("l");
    let function = when(value.clone().gt(lit(0_i64)))
        .then(Expr::Element + value)
        .otherwise(Expr::Element);

    let result = df!("l" => [-5_i64, 2, -3, 7])?
        .lazy()
        .select([foldv(lit(10_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(10), Some(12), Some(12), Some(19)]
    );
    Ok(())
}

#[test]
fn foldv_supports_nested_state_and_value_dependent_conditions() -> PolarsResult<()> {
    // The inner branch is reached only after both the row and carried state qualify.
    let value = col("l");
    let function = when(
        value
            .clone()
            .gt(lit(0_i64))
            .and(Expr::Element.lt_eq(lit(5_i64))),
    )
    .then(
        when(value.clone().gt_eq(lit(3_i64)))
            .then(Expr::Element + value)
            .otherwise(Expr::Element + lit(1_i64)),
    )
    .otherwise(Expr::Element);

    let result = df!("l" => [1_i64, 4, 9])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(5), Some(14)]
    );
    Ok(())
}

#[test]
fn foldv_supports_or_in_predicates() -> PolarsResult<()> {
    // Either matching comparison selects the state-updating branch.
    let value = col("l");
    let predicate = value
        .clone()
        .eq(lit(2_i64))
        .or(value.clone().eq(lit(4_i64)));
    let function = when(predicate)
        .then(Expr::Element + value)
        .otherwise(Expr::Element);

    let result = df!("l" => [1_i64, 2, 3, 4])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(0), Some(2), Some(2), Some(6)]
    );
    Ok(())
}

#[test]
fn foldv_supports_not_equal_in_predicates() -> PolarsResult<()> {
    // Rows unequal to the excluded value update the state.
    let value = col("l");
    let function = when(value.clone().neq(lit(2_i64)))
        .then(Expr::Element + value)
        .otherwise(Expr::Element);

    let result = df!("l" => [1_i64, 2, 3])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(1), Some(4)]
    );
    Ok(())
}

#[test]
fn foldv_supports_not_in_predicates() -> PolarsResult<()> {
    // Negating the comparison reverses which rows update the state.
    let value = col("l");
    let function = when(value.clone().eq(lit(2_i64)).not())
        .then(Expr::Element + value)
        .otherwise(Expr::Element);

    let result = df!("l" => [1_i64, 2, 3])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(1), Some(4)]
    );
    Ok(())
}

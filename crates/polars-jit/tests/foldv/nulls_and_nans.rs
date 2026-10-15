use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_propagates_nan_values_through_arithmetic() -> PolarsResult<()> {
    // NaN is a value, so arithmetic carries it through subsequent rows.
    let result = df!("l" => [1.0_f64, f64::NAN, 2.0])?
        .lazy()
        .select([foldv(lit(0.0_f64), Expr::Element + col("l"))?.alias("result")])
        .collect()?;

    let values = result.column("result")?.f64()?.to_vec();
    assert_eq!(values[0], Some(1.0));
    assert!(values[1].unwrap().is_nan());
    assert!(values[2].unwrap().is_nan());
    Ok(())
}

#[test]
fn foldv_uses_polars_total_order_for_nan_comparisons() -> PolarsResult<()> {
    // Polars considers every NaN equal to NaN and greater than finite values.
    let function = when(col("l").gt(Expr::Element))
        .then(col("l"))
        .otherwise(Expr::Element);
    let result = df!("l" => [1.0_f64, f64::NAN, 2.0, f64::NAN])?
        .lazy()
        .select([foldv(lit(0.0_f64), function)?.alias("result")])
        .collect()?;

    let values = result.column("result")?.f64()?.to_vec();
    assert_eq!(values[0], Some(1.0));
    assert!(values[1].unwrap().is_nan());
    assert!(values[2].unwrap().is_nan());
    assert!(values[3].unwrap().is_nan());
    Ok(())
}

#[test]
fn foldv_propagates_null_state_through_arithmetic() -> PolarsResult<()> {
    // Once arithmetic receives a null operand, the carried state remains null.
    let result = df!("l" => [Some(1_i64), None, Some(3)])?
        .lazy()
        .select([
            foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("result"),
        ])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), None, None]
    );
    Ok(())
}

#[test]
fn foldv_can_recover_from_a_null_initial_state() -> PolarsResult<()> {
    // is_null lets a branch replace a null state with the current row value.
    let function = when(Expr::Element.is_null())
        .then(col("l"))
        .otherwise(Expr::Element + col("l"));
    let result = df!("l" => [1_i64, 2, 3])?
        .lazy()
        .select([foldv(lit(NULL).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(3), Some(6)]
    );
    Ok(())
}

#[test]
fn foldv_supports_missing_equality_for_null_values() -> PolarsResult<()> {
    // Missing equality produces a non-null Boolean even when both operands are null.
    let function = when(col("l").eq_missing(lit(NULL).cast(DataType::Int64)))
        .then(Expr::Element + lit(1_i64))
        .otherwise(Expr::Element);
    let result = df!("l" => [Some(1_i64), None, None])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(0), Some(1), Some(2)]
    );
    Ok(())
}

#[test]
fn foldv_fills_nulls_with_the_supertype_of_both_operands() -> PolarsResult<()> {
    // fill_null replaces only null values, and an Int64 value filled with a float is Float64.
    let result = df!("l" => [Some(1_i64), None, Some(3)])?
        .lazy()
        .select([
            foldv(
                lit(0_i64).cast(DataType::Int64),
                Expr::Element + col("l").fill_null(lit(10)),
            )?
            .alias("int"),
            foldv(lit(0.0_f64), Expr::Element + col("l").fill_null(lit(0.5)))?.alias("float"),
        ])
        .collect()?;

    assert_eq!(
        result.column("int")?.i64()?.to_vec(),
        [Some(1), Some(11), Some(14)]
    );
    assert_eq!(
        result.column("float")?.f64()?.to_vec(),
        [Some(1.0), Some(1.5), Some(4.5)]
    );
    Ok(())
}

#[test]
fn foldv_treats_an_untyped_null_literal_as_null() -> PolarsResult<()> {
    // is_null does not need a dtype to know that a null literal is null.
    let function = when(lit(NULL).is_null())
        .then(Expr::Element + col("l"))
        .otherwise(Expr::Element);
    let result = df!("l" => [1_i64, 2])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.i64()?.to_vec(), [Some(1), Some(3)]);
    Ok(())
}

#[test]
fn foldv_infers_an_untyped_null_initial_state_from_the_function() -> PolarsResult<()> {
    // Like `lit(NULL) + col("l")`, an untyped null state takes the type the body gives it,
    // both in the planned schema and in the result.
    let function = when(Expr::Element.is_null())
        .then(col("l"))
        .otherwise(Expr::Element + col("l"));
    let query = df!("l" => [1_i64, 2, 3])?
        .lazy()
        .select([foldv(lit(NULL), function)?.alias("result")]);

    assert_eq!(
        query.clone().collect_schema()?.get("result"),
        Some(&DataType::Int64)
    );
    let result = query.collect()?;
    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(3), Some(6)]
    );
    Ok(())
}

#[test]
fn foldv_asks_for_a_dtype_when_the_function_does_not_determine_the_state_type() -> PolarsResult<()>
{
    // A body that never combines the state with a typed value leaves it untyped.
    let function = when(col("l").gt(lit(0_i64)))
        .then(Expr::Element)
        .otherwise(Expr::Element);
    let error = df!("l" => [1_i64])?
        .lazy()
        .select([foldv(lit(NULL), function)?])
        .collect()
        .unwrap_err();

    assert!(error.to_string().contains("untyped null"), "{error}");
    Ok(())
}

#[test]
fn foldv_infers_a_null_then_branch_from_the_otherwise_branch() -> PolarsResult<()> {
    // The untyped null literal takes the dtype of the other branch.
    let function = when(col("l").gt(lit(1_i64)))
        .then(lit(NULL))
        .otherwise(col("l"));
    let result = df!("l" => [1_i64, 2])?
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.i64()?.to_vec(), [Some(1), None]);
    Ok(())
}

#[test]
fn foldv_treats_a_null_predicate_as_false() -> PolarsResult<()> {
    // Polars when-then-otherwise selects otherwise for a null predicate.
    let function = when(col("l").gt(lit(0_i64)))
        .then(Expr::Element + lit(1_i64))
        .otherwise(Expr::Element);
    let result = df!("l" => [Some(1_i64), None, Some(-1), Some(2)])?
        .lazy()
        .select([foldv(lit(0).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(1), Some(1), Some(2)]
    );
    Ok(())
}

#[test]
fn foldv_uses_kleene_validity_for_boolean_operators() -> PolarsResult<()> {
    // False AND null is false, while true AND null is null.
    let result = df!(
        "a" => [Some(false), Some(true), Some(true)],
        "b" => [None, None, Some(true)],
    )?
    .lazy()
    .select([foldv(lit(true), col("a").and(col("b")))?.alias("result")])
    .collect()?;

    assert_eq!(
        result.column("result")?.bool()?.iter().collect::<Vec<_>>(),
        [Some(false), None, Some(true)]
    );
    Ok(())
}

#[test]
fn foldv_propagates_nulls_through_supported_casts() -> PolarsResult<()> {
    // A widening cast changes the value representation without making null valid.
    let result = df!("l" => [Some(1_i32), None, Some(3)])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int64),
            Expr::Element + col("l").cast(DataType::Int64),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), None, None]
    );
    Ok(())
}

#[test]
fn foldv_skips_nan_values_with_fill_nan_and_is_nan() -> PolarsResult<()> {
    // fill_nan replaces NaN but keeps nulls; is_nan of a null is null, which selects otherwise.
    let result = df!("l" => [Some(1.0_f64), Some(f64::NAN), None, Some(2.0)])?
        .lazy()
        .select([
            foldv(
                lit(0.0_f64),
                Expr::Element + col("l").fill_nan(lit(0.0_f64)),
            )?
            .alias("filled"),
            foldv(
                lit(0_i64).cast(DataType::Int64),
                when(col("l").is_nan())
                    .then(Expr::Element + lit(1_i64))
                    .otherwise(Expr::Element),
            )?
            .alias("nan_count"),
        ])
        .collect()?;

    assert_eq!(
        result.column("filled")?.f64()?.to_vec(),
        [Some(1.0), Some(1.0), None, None]
    );
    assert_eq!(
        result.column("nan_count")?.i64()?.to_vec(),
        [Some(0), Some(1), Some(1), Some(1)]
    );
    Ok(())
}

#[test]
fn foldv_compiles_chained_fill_null_once_per_link() -> PolarsResult<()> {
    // Each fill_null evaluates its value once, so a deep chain stays small instead of doubling.
    let mut filled = col("a");
    for _ in 0..24 {
        filled = filled.fill_null(col("b"));
    }
    let result = df!(
        "a" => [Some(1_i64), None, None],
        "b" => [Some(10_i64), Some(20), None],
    )?
    .lazy()
    .select([foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + filled)?.alias("result")])
    .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(21), None]
    );
    Ok(())
}

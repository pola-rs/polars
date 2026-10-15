use polars::prelude::*;
use polars_jit::foldv;

fn assert_foldv_sum(
    input: Column,
    state_dtype: DataType,
    expected: &[AnyValue<'_>],
) -> PolarsResult<()> {
    let result = DataFrame::new(input.len(), vec![input])?
        .lazy()
        .select([foldv(lit(0).cast(state_dtype), Expr::Element + col("l"))?.alias("result")])
        .collect()?;

    let result = result.column("result")?;
    assert_eq!(result.len(), expected.len());
    for (index, expected) in expected.iter().enumerate() {
        assert_eq!(&result.get(index)?, expected);
    }
    Ok(())
}

#[test]
fn foldv_carries_an_int32_state() -> PolarsResult<()> {
    // The generated loop loads and stores four-byte signed values.
    assert_foldv_sum(
        Column::new("l".into(), [1_i32, 2, 3]),
        DataType::Int32,
        &[AnyValue::Int32(1), AnyValue::Int32(3), AnyValue::Int32(6)],
    )
}

#[test]
fn foldv_carries_an_int64_state() -> PolarsResult<()> {
    // The generated loop loads and stores eight-byte signed values.
    assert_foldv_sum(
        Column::new("l".into(), [1_i64, 2, 3]),
        DataType::Int64,
        &[AnyValue::Int64(1), AnyValue::Int64(3), AnyValue::Int64(6)],
    )
}

#[test]
fn foldv_carries_a_uint32_state() -> PolarsResult<()> {
    // The generated loop loads and stores four-byte unsigned values.
    assert_foldv_sum(
        Column::new("l".into(), [1_u32, 2, 3]),
        DataType::UInt32,
        &[
            AnyValue::UInt32(1),
            AnyValue::UInt32(3),
            AnyValue::UInt32(6),
        ],
    )
}

#[test]
fn foldv_carries_a_uint64_state() -> PolarsResult<()> {
    // The generated loop loads and stores eight-byte unsigned values.
    assert_foldv_sum(
        Column::new("l".into(), [1_u64, 2, 3]),
        DataType::UInt64,
        &[
            AnyValue::UInt64(1),
            AnyValue::UInt64(3),
            AnyValue::UInt64(6),
        ],
    )
}

#[test]
fn foldv_carries_a_float32_state() -> PolarsResult<()> {
    // The generated loop retains Float32 values across every iteration.
    let result = df!("l" => [1.0_f32, 2.5, 3.0])?
        .lazy()
        .select([
            foldv(lit(0.5).cast(DataType::Float32), Expr::Element + col("l"))?.alias("result"),
        ])
        .collect()?;

    assert_eq!(
        result.column("result")?.f32()?.to_vec(),
        [Some(1.5), Some(4.0), Some(7.0)]
    );
    Ok(())
}

#[test]
fn foldv_carries_a_float64_state() -> PolarsResult<()> {
    // The generated loop retains Float64 values across every iteration.
    let result = df!("l" => [1.0_f64, 2.5, 3.0])?
        .lazy()
        .select([foldv(lit(0.5_f64), Expr::Element + col("l"))?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.f64()?.to_vec(),
        [Some(1.5), Some(4.0), Some(7.0)]
    );
    Ok(())
}

#[test]
fn foldv_executes_boolean_state_updates() -> PolarsResult<()> {
    // Boolean state and row values use the same typed program as numeric values.
    let result = df!("l" => [true, true, false])?
        .lazy()
        .select([foldv(lit(true), Expr::Element.and(col("l")))?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.bool()?.iter().collect::<Vec<_>>(),
        [Some(true), Some(true), Some(false)]
    );
    Ok(())
}

#[test]
fn foldv_uses_polars_coercion_for_mixed_row_and_state_types() -> PolarsResult<()> {
    // Polars resolves Int32 row values to Float64 before the JIT updates its Float64 state.
    let result = df!("l" => [1_i32, 2_i32, 3_i32])?
        .lazy()
        .select([foldv(lit(0.5_f64), Expr::Element + col("l"))?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.f64()?.to_vec(),
        [Some(1.5), Some(3.5), Some(6.5)]
    );
    Ok(())
}

#[test]
fn foldv_preserves_wrapping_integer_arithmetic() -> PolarsResult<()> {
    // Cranelift integer addition follows Polars' wrapping arithmetic at the type boundary.
    let result = df!("l" => [1_i32])?
        .lazy()
        .select([foldv(
            lit(i32::MAX).cast(DataType::Int32),
            Expr::Element + col("l"),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.i32()?.get(0), Some(i32::MIN));
    Ok(())
}

#[test]
fn foldv_preserves_infinity() -> PolarsResult<()> {
    // Arithmetic with infinity follows native floating-point behavior.
    let result = df!("l" => [f64::INFINITY])?
        .lazy()
        .select([foldv(lit(1.0_f64), Expr::Element + col("l"))?.alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.f64()?.get(0), Some(f64::INFINITY));
    Ok(())
}

#[test]
fn foldv_preserves_signed_zero_in_the_carried_state() -> PolarsResult<()> {
    // A state-preserving branch retains negative zero's exact bit pattern.
    let result = df!("l" => [0.0_f64])?
        .lazy()
        .select([foldv(
            lit(-0.0_f64),
            when(col("l").eq(lit(0.0_f64)))
                .then(Expr::Element)
                .otherwise(Expr::Element),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.f64()?.get(0).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    Ok(())
}

#[test]
fn foldv_uses_unsigned_ordering_for_unsigned_values() -> PolarsResult<()> {
    // Values above i64::MAX must retain UInt64 ordering in conditional predicates.
    let value = col("l");
    let function = when(value.clone().gt(lit(i64::MAX as u64)))
        .then(value)
        .otherwise(Expr::Element);
    let result = df!("l" => [1_u64, u64::MAX])?
        .lazy()
        .select([foldv(lit(0).cast(DataType::UInt64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.u64()?.to_vec(),
        [Some(0), Some(u64::MAX)]
    );
    Ok(())
}

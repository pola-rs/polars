use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_widens_integer_row_values() -> PolarsResult<()> {
    // Widening Int32 row values to Int64 cannot introduce nulls.
    let result = df!("l" => [1_i32, 2_i32, 3_i32])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int64),
            Expr::Element + col("l").cast(DataType::Int64),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(3), Some(6)]
    );
    Ok(())
}

#[test]
fn foldv_converts_integer_row_values_to_float() -> PolarsResult<()> {
    // Converting Int32 row values to Float64 keeps the result non-null.
    let result = df!("l" => [1_i32, 2_i32, 3_i32])?
        .lazy()
        .select([foldv(
            lit(0.5_f64),
            Expr::Element + col("l").cast(DataType::Float64),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.f64()?.to_vec(),
        [Some(1.5), Some(3.5), Some(6.5)]
    );
    Ok(())
}

#[test]
fn foldv_demotes_float64_row_values_to_float32() -> PolarsResult<()> {
    // Changing floating-point width retains a non-null floating-point value.
    let result = df!("l" => [1.25_f64, 2.5])?
        .lazy()
        .select([foldv(
            lit(0.25).cast(DataType::Float32),
            Expr::Element + col("l").cast(DataType::Float32),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.f32()?.to_vec(),
        [Some(1.5), Some(4.0)]
    );
    Ok(())
}

#[test]
fn foldv_returns_null_when_a_narrowing_integer_cast_overflows() -> PolarsResult<()> {
    // Polars' non-strict cast turns an out-of-range integer into a null state.
    let result = df!("l" => [1_i64, i64::MAX])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int32),
            Expr::Element + col("l").cast(DataType::Int32),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.i32()?.to_vec(), [Some(1), None]);
    Ok(())
}

#[test]
fn foldv_returns_null_when_a_float_to_integer_cast_is_invalid() -> PolarsResult<()> {
    // Polars' non-strict cast truncates finite values and maps NaN to a null state.
    let result = df!("l" => [1.5_f64, f64::NAN])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int64),
            Expr::Element + col("l").cast(DataType::Int64),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.i64()?.to_vec(), [Some(1), None]);
    Ok(())
}

#[test]
fn foldv_returns_null_when_an_unsigned_value_exceeds_the_signed_maximum() -> PolarsResult<()> {
    // Same-width unsigned-to-signed casts keep the bits, so 2^63 must not wrap negative.
    let result = df!("l" => [1_u64, 1_u64 << 63])?
        .lazy()
        .select([
            foldv(lit(0).cast(DataType::Int64), col("l").cast(DataType::Int64))?.alias("result"),
        ])
        .collect()?;

    assert_eq!(result.column("result")?.i64()?.to_vec(), [Some(1), None]);
    Ok(())
}

#[test]
fn foldv_truncates_floats_before_checking_the_integer_range() -> PolarsResult<()> {
    // Polars casts -0.5 to 0_u32 because truncation happens before the range check.
    let result = df!("l" => [-0.5_f64, -1.0])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::UInt32),
            col("l").cast(DataType::UInt32),
        )?
        .alias("result")])
        .collect()?;

    assert_eq!(result.column("result")?.u32()?.to_vec(), [Some(0), None]);
    Ok(())
}

#[test]
fn foldv_casts_numbers_to_boolean_by_comparing_with_zero() -> PolarsResult<()> {
    // Every nonzero value, including NaN, is true; negative zero is false.
    let result = df!(
        "i" => [Some(0_i64), Some(-3), None, Some(0)],
        "f" => [Some(-0.0_f64), Some(0.0), Some(f64::NAN), Some(0.5)],
    )?
    .lazy()
    .select([
        foldv(lit(false), col("i").cast(DataType::Boolean))?.alias("int"),
        foldv(lit(false), col("f").cast(DataType::Boolean))?.alias("float"),
    ])
    .collect()?;

    assert_eq!(
        result.column("int")?.bool()?.iter().collect::<Vec<_>>(),
        [Some(false), Some(true), None, Some(false)]
    );
    assert_eq!(
        result.column("float")?.bool()?.iter().collect::<Vec<_>>(),
        [Some(false), Some(false), Some(true), Some(true)]
    );
    Ok(())
}

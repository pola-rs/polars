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
fn foldv_rejects_narrowing_integer_casts() -> PolarsResult<()> {
    // Narrowing an integer may create a null and awaits validity tracking.
    let error = df!("l" => [1_i64])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int32),
            Expr::Element + col("l").cast(DataType::Int32),
        )?])
        .collect()
        .unwrap_err();

    assert!(error.to_string().contains("may produce null"));
    Ok(())
}

#[test]
fn foldv_rejects_float_to_integer_casts() -> PolarsResult<()> {
    // Converting a float to an integer may create a null and awaits validity tracking.
    let error = df!("l" => [1.0_f64])?
        .lazy()
        .select([foldv(
            lit(0).cast(DataType::Int64),
            Expr::Element + col("l").cast(DataType::Int64),
        )?])
        .collect()
        .unwrap_err();

    assert!(error.to_string().contains("may produce null"), "{error}");
    Ok(())
}

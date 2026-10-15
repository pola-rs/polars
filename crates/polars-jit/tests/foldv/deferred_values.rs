use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_rejects_nan_inputs_until_nan_semantics_are_supported() -> PolarsResult<()> {
    // NaN behavior is deferred together with null validity semantics.
    let error = df!("l" => [1.0_f64, f64::NAN])?
        .lazy()
        .select([foldv(lit(0.0_f64), Expr::Element + col("l"))?])
        .collect()
        .unwrap_err();

    assert!(error.to_string().contains("NaN values"));
    Ok(())
}

#[test]
fn foldv_rejects_nan_literals_until_nan_semantics_are_supported() -> PolarsResult<()> {
    // A NaN embedded in the program is rejected by translation as well as column data.
    let error = df!("l" => [1.0_f64])?
        .lazy()
        .select([foldv(
            lit(0.0_f64),
            Expr::Element + lit(f64::NAN) * col("l"),
        )?])
        .collect()
        .unwrap_err();

    assert!(error.to_string().contains("NaN values"));
    Ok(())
}

use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_reports_the_location_of_an_unsupported_expression() -> PolarsResult<()> {
    // Division is outside the supported scalar expression subset.
    let error = df!("l" => [1_i64, 2])?
        .lazy()
        .select([foldv(
            lit(10_i64).cast(DataType::Int64),
            Expr::Element + (col("l") / lit(2_i64)),
        )?
        .alias("result")])
        .collect()
        .unwrap_err();

    assert!(error.to_string().contains("root.right"));
    Ok(())
}

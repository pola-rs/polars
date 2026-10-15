use polars::prelude::*;
use polars_jit::foldv;

#[test]
fn foldv_returns_an_empty_column_when_the_row_input_is_empty() -> PolarsResult<()> {
    // With no rows, there are no updated states to return.
    let result = df!("l" => Vec::<i64>::new())?
        .lazy()
        .select([
            foldv(lit(42_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("result"),
        ])
        .collect()?;

    assert!(result.column("result")?.is_empty());
    Ok(())
}

#[test]
fn foldv_reads_sliced_arrow_bitmaps_from_their_bit_offset() -> PolarsResult<()> {
    // A sliced Boolean array can begin in the middle of both its value and validity bytes.
    let values = Series::new(
        "l".into(),
        [Some(false), Some(true), None, Some(true), Some(false)],
    )
    .slice(1, 3);
    let frame = DataFrame::new(3, vec![values.into_column()])?;
    let function = when(col("l"))
        .then(Expr::Element + lit(1_i64))
        .otherwise(Expr::Element);

    let result = frame
        .lazy()
        .select([foldv(lit(0_i64).cast(DataType::Int64), function)?.alias("result")])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), Some(1), Some(2)]
    );
    Ok(())
}

#[test]
fn foldv_reads_rechunked_arrow_value_and_validity_buffers() -> PolarsResult<()> {
    // Multiple input chunks are combined by Polars before their Arrow buffers reach the JIT.
    let mut values = Series::new("l".into(), [Some(1_i64), None]);
    values.append(&Series::new("l".into(), [Some(3_i64), Some(4)]))?;
    assert_eq!(values.n_chunks(), 2);
    let frame = DataFrame::new(4, vec![values.into_column()])?;

    let result = frame
        .lazy()
        .select([
            foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("result"),
        ])
        .collect()?;

    assert_eq!(
        result.column("result")?.i64()?.to_vec(),
        [Some(1), None, None, None]
    );
    Ok(())
}

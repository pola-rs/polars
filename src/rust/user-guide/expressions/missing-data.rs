fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    use polars::prelude::*;
    let df = df! (
        "value" => &[Some(1), None],
    )?;

    println!("{df}");
    // --8<-- [end:dataframe]

    // --8<-- [start:count]
    let null_count_df = df.null_count();
    println!("{null_count_df}");
    // --8<-- [end:count]

    // --8<-- [start:isnull]
    let is_null_series = df.lazy().select([col("value").is_null()]).collect()?;
    println!("{is_null_series}");
    // --8<-- [end:isnull]

    // --8<-- [start:dataframe2]
    let df = df! (
        "col1" => [0.5, 1.0, 1.5, 2.0, 2.5],
        "col2" => [Some(1), None, Some(3), None, Some(5)],
    )?;

    println!("{df}");
    // --8<-- [end:dataframe2]

    // --8<-- [start:fill]
    let fill_literal_df = df
        .clone()
        .lazy()
        .with_column(col("col2").fill_null(3))
        .collect()?;

    println!("{fill_literal_df}");
    // --8<-- [end:fill]

    // --8<-- [start:fillstrategy]

    let fill_literal_df = df
        .clone()
        .lazy()
        .with_columns([
            col("col2")
                .fill_null_with_strategy(FillNullStrategy::Forward(None))
                .alias("forward"),
            col("col2")
                .fill_null_with_strategy(FillNullStrategy::Backward(None))
                .alias("backward"),
        ])
        .collect()?;

    println!("{fill_literal_df}");
    // --8<-- [end:fillstrategy]

    // --8<-- [start:fillexpr]
    let fill_expression_df = df
        .clone()
        .lazy()
        .with_column(col("col2").fill_null((lit(2) * col("col1")).cast(DataType::Int64)))
        .collect()?;

    println!("{fill_expression_df}");
    // --8<-- [end:fillexpr]

    // --8<-- [start:fillinterpolate]
    let fill_interpolation_df = df
        .lazy()
        .with_column(col("col2").interpolate(InterpolationMethod::Linear))
        .collect()?;

    println!("{fill_interpolation_df}");
    // --8<-- [end:fillinterpolate]

    // --8<-- [start:nan]
    let nan_df = df!(
        "value" => [1.0, f64::NAN, f64::NAN, 3.0],
    )?;
    println!("{nan_df}");
    // --8<-- [end:nan]

    // --8<-- [start:nan-computed]
    let df = df!(
        "dividend" => [1.0, 0.0, -1.0],
        "divisor" => [1.0, 0.0, -1.0],
    )?;

    let result = df
        .lazy()
        .select([col("dividend") / col("divisor")])
        .collect()?;

    println!("{result}");
    // --8<-- [end:nan-computed]

    // --8<-- [start:nanfill]
    let mean_nan_df = nan_df
        .lazy()
        .with_column(col("value").fill_nan(Null {}.lit()).alias("replaced"))
        .select([
            col("*").mean().name().suffix("_mean"),
            col("*").sum().name().suffix("_sum"),
        ])
        .collect()?;

    println!("{mean_nan_df}");
    // --8<-- [end:nanfill]
    Ok(())
}

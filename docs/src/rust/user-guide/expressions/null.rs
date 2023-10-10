use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]

    let df = df! (
        "value" => &[Some(1), None],
    )?;

    println!("{}", &df);
    // --8<-- [end:dataframe]

    // --8<-- [start:count]
    let null_count_df = df.null_count();
    println!("{}", &null_count_df);
    // --8<-- [end:count]

    // --8<-- [start:isnull]
    let is_null_series = df
        .clone()
        .lazy()
        .select([col("value").is_null()])
        .collect()?;
    println!("{}", &is_null_series);
    // --8<-- [end:isnull]

    // --8<-- [start:dataframe2]
    let df = df!(
            "col1" => &[Some(1), Some(2), Some(3)],
            "col2" => &[Some(1), None, Some(3)],

    )?;
    println!("{}", &df);
    // --8<-- [end:dataframe2]

    // --8<-- [start:fill]
    let fill_literal_df = df
        .clone()
        .lazy()
        .with_columns([col("col2").fill_null(lit(2))])
        .collect()?;
    println!("{}", &fill_literal_df);
    // --8<-- [end:fill]

    // --8<-- [start:fillstrategy]
    let fill_forward_df = df
        .clone()
        .lazy()
        .with_columns([col("col2").forward_fill(None)])
        .collect()?;
    println!("{}", &fill_forward_df);
    // --8<-- [end:fillstrategy]

    // --8<-- [start:fillexpr]
    let fill_median_df = df
        .clone()
        .lazy()
        .with_columns([col("col2").fill_null(median("col2"))])
        .collect()?;
    println!("{}", &fill_median_df);
    // --8<-- [end:fillexpr]

    // --8<-- [start:fillinterpolate]
    let fill_interpolation_df = df
        .clone()
        .lazy()
        .with_columns([col("col2").interpolate(InterpolationMethod::Linear)])
        .collect()?;
    println!("{}", &fill_interpolation_df);
    // --8<-- [end:fillinterpolate]

    // --8<-- [start:nan]
    let nan_df = df!(
            "value" => [1.0, f64::NAN, f64::NAN, 3.0],
    )?;
    println!("{}", &nan_df);
    // --8<-- [end:nan]

    // --8<-- [start:nanfill]
    let mean_nan_df = nan_df
        .clone()
        .lazy()
        .with_columns([col("value").fill_nan(lit(NULL)).alias("value")])
        .mean()
        .collect()?;
    println!("{}", &mean_nan_df);
    // --8<-- [end:nanfill]
    Ok(())
}

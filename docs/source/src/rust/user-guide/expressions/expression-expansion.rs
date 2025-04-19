fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    use polars::prelude::*;

    // Data as of 14th October 2024, ~3pm UTC
    let df = df!(
        "ticker" => ["AAPL", "NVDA", "MSFT", "GOOG", "AMZN"],
        "company_name" => ["Apple", "NVIDIA", "Microsoft", "Alphabet (Google)", "Amazon"],
        "price" => [229.9, 138.93, 420.56, 166.41, 188.4],
        "day_high" => [231.31, 139.6, 424.04, 167.62, 189.83],
        "day_low" => [228.6, 136.3, 417.52, 164.78, 188.44],
        "year_high" => [237.23, 140.76, 468.35, 193.31, 201.2],
        "year_low" => [164.08, 39.23, 324.39, 121.46, 118.35],
    )?;

    println!("{}", df);
    // --8<-- [end:df]

    // --8<-- [start:col-with-names]
    let eur_usd_rate = 1.09; // As of 14th October 2024

    let result = df
        .clone()
        .lazy()
        .with_column(
            (cols(["price", "day_high", "day_low", "year_high", "year_low"]) / lit(eur_usd_rate))
                .round(2),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:col-with-names]

    // --8<-- [start:expression-list]
    let exprs = [
        (col("price") / lit(eur_usd_rate)).round(2),
        (col("day_high") / lit(eur_usd_rate)).round(2),
        (col("day_low") / lit(eur_usd_rate)).round(2),
        (col("year_high") / lit(eur_usd_rate)).round(2),
        (col("year_low") / lit(eur_usd_rate)).round(2),
    ];

    let result2 = df.clone().lazy().with_columns(exprs).collect()?;
    println!("{}", result.equals(&result2));
    // --8<-- [end:expression-list]

    // --8<-- [start:col-with-dtype]
    let result = df
        .clone()
        .lazy()
        .with_column((dtype_col(&DataType::Float64) / lit(eur_usd_rate)).round(2))
        .collect()?;
    println!("{}", result);
    // --8<-- [end:col-with-dtype]

    // --8<-- [start:col-with-dtypes]
    let result2 = df
        .clone()
        .lazy()
        .with_column(
            (dtype_cols([DataType::Float32, DataType::Float64]) / lit(eur_usd_rate)).round(2),
        )
        .collect()?;
    println!("{}", result.equals(&result2));
    // --8<-- [end:col-with-dtypes]

    // --8<-- [start:col-with-regex]
    // NOTE: Using regex inside `col`/`cols` requires the feature flag `regex`.
    let result = df
        .clone()
        .lazy()
        .select([cols(["ticker", "^.*_high$", "^.*_low$"])])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:col-with-regex]

    // --8<-- [start:all]
    let result = df.clone().lazy().select([all()]).collect()?;
    println!("{}", result.equals(&df));
    // --8<-- [end:all]

    // --8<-- [start:all-exclude]
    let result = df
        .clone()
        .lazy()
        .select([all().exclude(["^day_.*$"])])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:all-exclude]

    // --8<-- [start:col-exclude]
    let result = df
        .clone()
        .lazy()
        .select([dtype_col(&DataType::Float64).exclude(["^day_.*$"])])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:col-exclude]

    // --8<-- [start:duplicate-error]
    let gbp_usd_rate = 1.31; // As of 14th October 2024

    let result = df
        .clone()
        .lazy()
        .select([
            col("price") / lit(gbp_usd_rate),
            col("price") / lit(eur_usd_rate),
        ])
        .collect();
    match result {
        Ok(df) => println!("{}", df),
        Err(e) => println!("{}", e),
    };
    // --8<-- [end:duplicate-error]

    // --8<-- [start:alias]
    let _result = df
        .clone()
        .lazy()
        .select([
            (col("price") / lit(gbp_usd_rate)).alias("price (GBP)"),
            (col("price") / lit(eur_usd_rate)).alias("price (EUR)"),
        ])
        .collect()?;
    // --8<-- [end:alias]

    // --8<-- [start:prefix-suffix]
    let result = df
        .clone()
        .lazy()
        .select([
            (col("^year_.*$") / lit(eur_usd_rate))
                .name()
                .prefix("in_eur_"),
            (cols(["day_high", "day_low"]) / lit(gbp_usd_rate))
                .name()
                .suffix("_gbp"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:prefix-suffix]

    // --8<-- [start:name-map]
    // There is also `name().to_uppercase()`, so this usage of `map` is moot.
    let result = df
        .clone()
        .lazy()
        .select([all()
            .name()
            .map(|name| Ok(PlSmallStr::from_string(name.to_ascii_uppercase())))])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:name-map]

    // --8<-- [start:for-with_columns]
    let mut result = df.clone().lazy();
    for tp in ["day", "year"] {
        let high = format!("{}_high", tp);
        let low = format!("{}_low", tp);
        let aliased = format!("{}_amplitude", tp);
        result = result.with_column((col(high) - col(low)).alias(aliased))
    }
    let result = result.collect()?;
    println!("{}", result);
    // --8<-- [end:for-with_columns]

    // --8<-- [start:yield-expressions]
    let mut exprs: Vec<Expr> = vec![];
    for tp in ["day", "year"] {
        let high = format!("{}_high", tp);
        let low = format!("{}_low", tp);
        let aliased = format!("{}_amplitude", tp);
        exprs.push((col(high) - col(low)).alias(aliased))
    }
    let result = df.clone().lazy().with_columns(exprs).collect()?;
    println!("{}", result);
    // --8<-- [end:yield-expressions]

    // --8<-- [start:selectors]
    // Selectors are not available in Rust yet.
    // Refer to https://github.com/pola-rs/polars/issues/10594
    // --8<-- [end:selectors]

    // --8<-- [start:selectors-set-operations]
    // Selectors are not available in Rust yet.
    // Refer to https://github.com/pola-rs/polars/issues/10594
    // --8<-- [end:selectors-set-operations]

    // --8<-- [start:selectors-expressions]
    // Selectors are not available in Rust yet.
    // Refer to https://github.com/pola-rs/polars/issues/10594
    // --8<-- [end:selectors-expressions]

    // --8<-- [start:selector-ambiguity]
    // Selectors are not available in Rust yet.
    // Refer to https://github.com/pola-rs/polars/issues/10594
    // --8<-- [end:selector-ambiguity]

    // --8<-- [start:as_expr]
    // Selectors are not available in Rust yet.
    // Refer to https://github.com/pola-rs/polars/issues/10594
    // --8<-- [end:as_expr]

    // --8<-- [start:is_selector]
    // Selectors are not available in Rust yet.
    // Refer to https://github.com/pola-rs/polars/issues/10594
    // --8<-- [end:is_selector]

    // --8<-- [start:expand_selector]
    // Selectors are not available in Rust yet.
    // Refer to https://github.com/pola-rs/polars/issues/10594
    // --8<-- [end:expand_selector]

    Ok(())
}

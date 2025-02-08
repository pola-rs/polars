// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // NOTE: This assumes the data has been downloaded and is available.
    // See the corresponding Python script for the remote location of the data.

    // --8<-- [start:props_groups]
    let props_groups = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("docs/assets/data/monopoly_props_groups.csv".into()))?
        .finish()?
        .head(Some(5));
    println!("{}", props_groups);
    // --8<-- [end:props_groups]

    // --8<-- [start:props_prices]
    let props_prices = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("docs/assets/data/monopoly_props_prices.csv".into()))?
        .finish()?
        .head(Some(5));
    println!("{}", props_prices);
    // --8<-- [end:props_prices]

    // --8<-- [start:equi-join]
    // In Rust, we cannot use the shorthand of specifying a common
    // column name just once.
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::default(),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:equi-join]

    // --8<-- [start:props_groups2]
    let props_groups2 = props_groups
        .clone()
        .lazy()
        .with_column(col("property_name").str().to_lowercase())
        .collect()?;
    println!("{}", props_groups2);
    // --8<-- [end:props_groups2]

    // --8<-- [start:props_prices2]
    let props_prices2 = props_prices
        .clone()
        .lazy()
        .select([col("property_name").alias("name"), col("cost")])
        .collect()?;
    println!("{}", props_prices2);
    // --8<-- [end:props_prices2]

    // --8<-- [start:join-key-expression]
    let result = props_groups2
        .clone()
        .lazy()
        .join(
            props_prices2.clone().lazy(),
            [col("property_name")],
            [col("name").str().to_lowercase()],
            JoinArgs::default(),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:join-key-expression]

    // --8<-- [start:inner-join]
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::new(JoinType::Inner),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:inner-join]

    // --8<-- [start:left-join]
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::new(JoinType::Left),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:left-join]

    // --8<-- [start:right-join]
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::new(JoinType::Right),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:right-join]

    // --8<-- [start:left-right-join-equals]
    // `equals_missing` is needed instead of `equals`
    // so that missing values compare as equal.
    let dfs_match = result.equals_missing(
        &props_prices
            .clone()
            .lazy()
            .join(
                props_groups.clone().lazy(),
                [col("property_name")],
                [col("property_name")],
                JoinArgs::new(JoinType::Left),
            )
            .select([
                // Reorder the columns to match the order of `result`.
                col("group"),
                col("property_name"),
                col("cost"),
            ])
            .collect()?,
    );
    println!("{}", dfs_match);
    // --8<-- [end:left-right-join-equals]

    // --8<-- [start:full-join]
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::new(JoinType::Full),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:full-join]

    // --8<-- [start:full-join-coalesce]
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::new(JoinType::Full).with_coalesce(JoinCoalesce::CoalesceColumns),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:full-join-coalesce]

    // --8<-- [start:semi-join]
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::new(JoinType::Semi),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:semi-join]

    // --8<-- [start:anti-join]
    let result = props_groups
        .clone()
        .lazy()
        .join(
            props_prices.clone().lazy(),
            [col("property_name")],
            [col("property_name")],
            JoinArgs::new(JoinType::Anti),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:anti-join]

    // --8<-- [start:players]
    let players = df!(
        "name" => ["Alice", "Bob"],
        "cash" => [78, 135],
    )?;
    println!("{}", players);
    // --8<-- [end:players]

    // --8<-- [start:non-equi]
    let result = players
        .clone()
        .lazy()
        .join_builder()
        .with(props_prices.clone().lazy())
        .join_where(vec![col("cash").cast(DataType::Int64).gt(col("cost"))])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:non-equi]

    // --8<-- [start:df_trades]
    use chrono::prelude::*;

    let df_trades = df!(
        "time" => [
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 1, 0).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 1, 0).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 3, 0).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 6, 0).unwrap(),
        ],
        "stock" => ["A", "B", "B", "C"],
        "trade" => [101, 299, 301, 500],
    )?;
    println!("{}", df_trades);
    // --8<-- [end:df_trades]

    // --8<-- [start:df_quotes]
    let df_quotes = df!(
        "time" => [
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 1, 0).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 2, 0).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 4, 0).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 6, 0).unwrap(),
        ],
        "stock" => ["A", "B", "C", "A"],
        "quote" => [100, 300, 501, 102],
    )?;
    println!("{}", df_quotes);
    // --8<-- [end:df_quotes]

    // --8<-- [start:asof]
    let result = df_trades.join_asof_by(
        &df_quotes,
        "time",
        "time",
        ["stock"],
        ["stock"],
        AsofStrategy::Backward,
        None,
        true,
        true,
    )?;
    println!("{}", result);
    // --8<-- [end:asof]

    // --8<-- [start:asof-tolerance]
    let result = df_trades.join_asof_by(
        &df_quotes,
        "time",
        "time",
        ["stock"],
        ["stock"],
        AsofStrategy::Backward,
        Some(AnyValue::Duration(60000, TimeUnit::Milliseconds)),
        true,
        true,
    )?;
    println!("{}", result);
    // --8<-- [end:asof-tolerance]

    // --8<-- [start:cartesian-product]
    let tokens = df!(
        "monopoly_token" => ["hat", "shoe", "boat"],
    )?;

    let result = players
        .clone()
        .lazy()
        .select([col("name")])
        .cross_join(tokens.clone().lazy(), None)
        .collect()?;
    println!("{}", result);
    // --8<-- [end:cartesian-product]

    Ok(())
}

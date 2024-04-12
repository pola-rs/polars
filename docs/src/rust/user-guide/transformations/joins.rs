// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:innerdf]
    let df_customers = df! (

        "customer_id" => &[1, 2, 3],
        "name" => &["Alice", "Bob", "Charlie"],
    )?;

    println!("{}", &df_customers);
    // --8<-- [end:innerdf]

    // --8<-- [start:innerdf2]
    let df_orders = df!(
            "order_id"=> &["a", "b", "c"],
            "customer_id"=> &[1, 2, 2],
            "amount"=> &[100, 200, 300],
    )?;
    println!("{}", &df_orders);
    // --8<-- [end:innerdf2]

    // --8<-- [start:inner]
    let df_inner_customer_join = df_customers
        .clone()
        .lazy()
        .join(
            df_orders.clone().lazy(),
            [col("customer_id")],
            [col("customer_id")],
            JoinArgs::new(JoinType::Inner),
        )
        .collect()?;
    println!("{}", &df_inner_customer_join);
    // --8<-- [end:inner]

    // --8<-- [start:left]
    let df_left_join = df_customers
        .clone()
        .lazy()
        .join(
            df_orders.clone().lazy(),
            [col("customer_id")],
            [col("customer_id")],
            JoinArgs::new(JoinType::Left),
        )
        .collect()?;
    println!("{}", &df_left_join);
    // --8<-- [end:left]

    // --8<-- [start:outer]
    let df_outer_join = df_customers
        .clone()
        .lazy()
        .join(
            df_orders.clone().lazy(),
            [col("customer_id")],
            [col("customer_id")],
            JoinArgs::new(JoinType::Outer { coalesce: false }),
        )
        .collect()?;
    println!("{}", &df_outer_join);
    // --8<-- [end:outer]

    // --8<-- [start:outer_coalesce]
    let df_outer_join = df_customers
        .clone()
        .lazy()
        .join(
            df_orders.clone().lazy(),
            [col("customer_id")],
            [col("customer_id")],
            JoinArgs::new(JoinType::Outer { coalesce: true }),
        )
        .collect()?;
    println!("{}", &df_outer_join);
    // --8<-- [end:outer_coalesce]

    // --8<-- [start:df3]
    let df_colors = df!(
            "color"=> &["red", "blue", "green"],
    )?;
    println!("{}", &df_colors);
    // --8<-- [end:df3]

    // --8<-- [start:df4]
    let df_sizes = df!(
            "size"=> &["S", "M", "L"],
    )?;
    println!("{}", &df_sizes);
    // --8<-- [end:df4]

    // --8<-- [start:cross]
    let df_cross_join = df_colors
        .clone()
        .lazy()
        .cross_join(df_sizes.clone().lazy())
        .collect()?;
    println!("{}", &df_cross_join);
    // --8<-- [end:cross]

    // --8<-- [start:df5]
    let df_cars = df!(
            "id"=> &["a", "b", "c"],
            "make"=> &["ford", "toyota", "bmw"],
    )?;
    println!("{}", &df_cars);
    // --8<-- [end:df5]

    // --8<-- [start:df6]
    let df_repairs = df!(
            "id"=> &["c", "c"],
            "cost"=> &[100, 200],
    )?;
    println!("{}", &df_repairs);
    // --8<-- [end:df6]

    // --8<-- [start:inner2]
    let df_inner_join = df_cars
        .clone()
        .lazy()
        .inner_join(df_repairs.clone().lazy(), col("id"), col("id"))
        .collect()?;
    println!("{}", &df_inner_join);
    // --8<-- [end:inner2]

    // --8<-- [start:semi]
    let df_semi_join = df_cars
        .clone()
        .lazy()
        .join(
            df_repairs.clone().lazy(),
            [col("id")],
            [col("id")],
            JoinArgs::new(JoinType::Semi),
        )
        .collect()?;
    println!("{}", &df_semi_join);
    // --8<-- [end:semi]

    // --8<-- [start:anti]
    let df_anti_join = df_cars
        .clone()
        .lazy()
        .join(
            df_repairs.clone().lazy(),
            [col("id")],
            [col("id")],
            JoinArgs::new(JoinType::Anti),
        )
        .collect()?;
    println!("{}", &df_anti_join);
    // --8<-- [end:anti]

    // --8<-- [start:df7]
    use chrono::prelude::*;
    let df_trades = df!(
        "time"=> &[
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 1, 0).unwrap(),
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 1, 0).unwrap(),
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 3, 0).unwrap(),
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 6, 0).unwrap(),
            ],
            "stock"=> &["A", "B", "B", "C"],
            "trade"=> &[101, 299, 301, 500],
    )?;
    println!("{}", &df_trades);
    // --8<-- [end:df7]

    // --8<-- [start:df8]
    let df_quotes = df!(
            "time"=> &[
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 0, 0).unwrap(),
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 2, 0).unwrap(),
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 4, 0).unwrap(),
        NaiveDate::from_ymd_opt(2020, 1, 1).unwrap().and_hms_opt(9, 6, 0).unwrap(),
            ],
            "stock"=> &["A", "B", "C", "A"],
            "quote"=> &[100, 300, 501, 102],
    )?;

    println!("{}", &df_quotes);
    // --8<-- [end:df8]

    // --8<-- [start:asofpre]
    let df_trades = df_trades
        .sort(
            ["time"],
            SortMultipleOptions::default().with_maintain_order(true),
        )
        .unwrap();
    let df_quotes = df_quotes
        .sort(
            ["time"],
            SortMultipleOptions::default().with_maintain_order(true),
        )
        .unwrap();
    // --8<-- [end:asofpre]

    // --8<-- [start:asof]
    let df_asof_join = df_trades.join_asof_by(
        &df_quotes,
        "time",
        "time",
        ["stock"],
        ["stock"],
        AsofStrategy::Backward,
        None,
    )?;
    println!("{}", &df_asof_join);
    // --8<-- [end:asof]

    // --8<-- [start:asof2]
    let df_asof_tolerance_join = df_trades.join_asof_by(
        &df_quotes,
        "time",
        "time",
        ["stock"],
        ["stock"],
        AsofStrategy::Backward,
        Some(AnyValue::Duration(60000, TimeUnit::Milliseconds)),
    )?;
    println!("{}", &df_asof_tolerance_join);
    // --8<-- [end:asof2]

    Ok(())
}

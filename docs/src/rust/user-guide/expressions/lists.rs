// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:weather_df]
    let stns: Vec<String> = (1..6).map(|i| format!("Station {i}")).collect();
    let weather = df!(
            "station"=> &stns,
            "temperatures"=> &[
                "20 5 5 E1 7 13 19 9 6 20",
                "18 8 16 11 23 E2 8 E2 E2 E2 90 70 40",
                "19 24 E9 16 6 12 10 22",
                "E2 E0 15 7 8 10 E1 24 17 13 6",
                "14 8 E0 16 22 24 E1",
            ],
    )?;
    println!("{}", &weather);
    // --8<-- [end:weather_df]

    // --8<-- [start:string_to_list]
    let out = weather
        .clone()
        .lazy()
        .with_columns([col("temperatures").str().split(lit(" "))])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:string_to_list]

    // --8<-- [start:explode_to_atomic]
    let out = weather
        .clone()
        .lazy()
        .with_columns([col("temperatures").str().split(lit(" "))])
        .explode(["temperatures"])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:explode_to_atomic]

    // --8<-- [start:list_ops]
    let out = weather
        .clone()
        .lazy()
        .with_columns([col("temperatures").str().split(lit(" "))])
        .with_columns([
            col("temperatures").list().head(lit(3)).alias("top3"),
            col("temperatures")
                .list()
                .slice(lit(-3), lit(3))
                .alias("bottom_3"),
            col("temperatures").list().len().alias("obs"),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:list_ops]

    // --8<-- [start:count_errors]
    let out = weather
        .clone()
        .lazy()
        .with_columns([col("temperatures")
            .str()
            .split(lit(" "))
            .list()
            .eval(col("").cast(DataType::Int64).is_null(), false)
            .list()
            .sum()
            .alias("errors")])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:count_errors]

    // --8<-- [start:count_errors_regex]
    let out = weather
        .clone()
        .lazy()
        .with_columns([col("temperatures")
            .str()
            .split(lit(" "))
            .list()
            .eval(col("").str().contains(lit("(?i)[a-z]"), false), false)
            .list()
            .sum()
            .alias("errors")])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:count_errors_regex]

    // --8<-- [start:weather_by_day]
    let stns: Vec<String> = (1..11).map(|i| format!("Station {i}")).collect();
    let weather_by_day = df!(
            "station" => &stns,
            "day_1" => &[17, 11, 8, 22, 9, 21, 20, 8, 8, 17],
            "day_2" => &[15, 11, 10, 8, 7, 14, 18, 21, 15, 13],
            "day_3" => &[16, 15, 24, 24, 8, 23, 19, 23, 16, 10],
    )?;
    println!("{}", &weather_by_day);
    // --8<-- [end:weather_by_day]

    // --8<-- [start:weather_by_day_rank]
    let rank_pct = (col("")
        .rank(
            RankOptions {
                method: RankMethod::Average,
                descending: true,
            },
            None,
        )
        .cast(DataType::Float32)
        / col("*").count().cast(DataType::Float32))
    .round(2);

    let out = weather_by_day
        .clone()
        .lazy()
        .with_columns(
            // create the list of homogeneous data
            [concat_list([all().exclude(["station"])])?.alias("all_temps")],
        )
        .select(
            // select all columns except the intermediate list
            [
                all().exclude(["all_temps"]),
                // compute the rank by calling `list.eval`
                col("all_temps")
                    .list()
                    .eval(rank_pct, true)
                    .alias("temps_rank"),
            ],
        )
        .collect()?;

    println!("{}", &out);
    // --8<-- [end:weather_by_day_rank]

    // --8<-- [start:array_df]
    let mut col1: ListPrimitiveChunkedBuilder<Int32Type> =
        ListPrimitiveChunkedBuilder::new("Array_1", 8, 8, DataType::Int32);
    col1.append_slice(&[1, 3]);
    col1.append_slice(&[2, 5]);
    let mut col2: ListPrimitiveChunkedBuilder<Int32Type> =
        ListPrimitiveChunkedBuilder::new("Array_2", 8, 8, DataType::Int32);
    col2.append_slice(&[1, 7, 3]);
    col2.append_slice(&[8, 1, 0]);
    let array_df = DataFrame::new([col1.finish(), col2.finish()].into())?;

    println!("{}", &array_df);
    // --8<-- [end:array_df]

    // --8<-- [start:array_ops]
    let out = array_df
        .clone()
        .lazy()
        .select([
            col("Array_1").list().min().name().suffix("_min"),
            col("Array_2").list().sum().name().suffix("_sum"),
        ])
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:array_ops]

    Ok(())
}

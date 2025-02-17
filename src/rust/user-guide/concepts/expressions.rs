use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    use chrono::prelude::*;
    use polars::prelude::*;

    let df: DataFrame = df!(
        "name" => ["Alice Archer", "Ben Brown", "Chloe Cooper", "Daniel Donovan"],
        "birthdate" => [
            NaiveDate::from_ymd_opt(1997, 1, 10).unwrap(),
            NaiveDate::from_ymd_opt(1985, 2, 15).unwrap(),
            NaiveDate::from_ymd_opt(1983, 3, 22).unwrap(),
            NaiveDate::from_ymd_opt(1981, 4, 30).unwrap(),
        ],
        "weight" => [57.9, 72.5, 53.6, 83.1],  // (kg)
        "height" => [1.56, 1.77, 1.65, 1.75],  // (m)
    )
    .unwrap();
    println!("{}", df);
    // --8<-- [end:df]

    // --8<-- [start:select-1]
    let bmi = col("weight") / col("height").pow(2);
    let result = df
        .clone()
        .lazy()
        .select([
            bmi.clone().alias("bmi"),
            bmi.clone().mean().alias("avg_bmi"),
            lit(25).alias("ideal_max_bmi"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:select-1]

    // --8<-- [start:select-2]
    let result = df
        .clone()
        .lazy()
        .select([((bmi.clone() - bmi.clone().mean()) / bmi.clone().std(1)).alias("deviation")])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:select-2]

    // --8<-- [start:with_columns-1]
    let result = df
        .clone()
        .lazy()
        .with_columns([
            bmi.clone().alias("bmi"),
            bmi.clone().mean().alias("avg_bmi"),
            lit(25).alias("ideal_max_bmi"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:with_columns-1]

    // --8<-- [start:filter-1]
    let result = df
        .clone()
        .lazy()
        .filter(
            col("birthdate")
                .is_between(
                    lit(NaiveDate::from_ymd_opt(1982, 12, 31).unwrap()),
                    lit(NaiveDate::from_ymd_opt(1996, 1, 1).unwrap()),
                    ClosedInterval::Both,
                )
                .and(col("height").gt(lit(1.7))),
        )
        .collect()?;
    println!("{}", result);
    // --8<-- [end:filter-1]

    // --8<-- [start:group_by-1]
    let result = df
        .clone()
        .lazy()
        .group_by([(col("birthdate").dt().year() / lit(10) * lit(10)).alias("decade")])
        .agg([col("name")])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:group_by-1]

    // --8<-- [start:group_by-2]
    let result = df
        .clone()
        .lazy()
        .group_by([
            (col("birthdate").dt().year() / lit(10) * lit(10)).alias("decade"),
            (col("height").lt(lit(1.7)).alias("short?")),
        ])
        .agg([col("name")])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:group_by-2]

    // --8<-- [start:group_by-3]
    let result = df
        .clone()
        .lazy()
        .group_by([
            (col("birthdate").dt().year() / lit(10) * lit(10)).alias("decade"),
            (col("height").lt(lit(1.7)).alias("short?")),
        ])
        .agg([
            len(),
            col("height").max().alias("tallest"),
            cols(["weight", "height"]).mean().name().prefix("avg_"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:group_by-3]

    // --8<-- [start:expression-expansion-1]
    let expr = (dtype_col(&DataType::Float64) * lit(1.1))
        .name()
        .suffix("*1.1");
    let result = df.clone().lazy().select([expr.clone()]).collect()?;
    println!("{}", result);
    // --8<-- [end:expression-expansion-1]

    // --8<-- [start:expression-expansion-2]
    let df2: DataFrame = df!(
        "ints" => [1, 2, 3, 4],
        "letters" => ["A", "B", "C", "D"],
    )
    .unwrap();
    let result = df2.clone().lazy().select([expr.clone()]).collect()?;
    println!("{}", result);
    // --8<-- [end:expression-expansion-2]

    Ok(())
}

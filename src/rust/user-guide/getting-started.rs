fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    use chrono::prelude::*;
    use polars::prelude::*;

    let mut df: DataFrame = df!(
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

    // --8<-- [start:csv]
    use std::fs::File;

    let mut file = File::create("../../../assets/data/output.csv").expect("could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df)?;
    let df_csv = CsvReadOptions::default()
        .with_infer_schema_length(None)
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some("../../../assets/data/output.csv".into()))?
        .finish()?;
    println!("{}", df_csv);
    // --8<-- [end:csv]

    // --8<-- [start:select]
    let result = df
        .clone()
        .lazy()
        .select([
            col("name"),
            col("birthdate").dt().year().alias("birth_year"),
            (col("weight") / col("height").pow(2)).alias("bmi"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:select]

    // --8<-- [start:expression-expansion]
    let result = df
        .clone()
        .lazy()
        .select([
            col("name"),
            (cols(["weight", "height"]) * lit(0.95))
                .round(2)
                .name()
                .suffix("-5%"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:expression-expansion]

    // --8<-- [start:with_columns]
    let result = df
        .clone()
        .lazy()
        .with_columns([
            col("birthdate").dt().year().alias("birth_year"),
            (col("weight") / col("height").pow(2)).alias("bmi"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:with_columns]

    // --8<-- [start:filter]
    let result = df
        .clone()
        .lazy()
        .filter(col("birthdate").dt().year().lt(lit(1990)))
        .collect()?;
    println!("{}", result);
    // --8<-- [end:filter]

    // --8<-- [start:filter-multiple]
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
    // --8<-- [end:filter-multiple]

    // --8<-- [start:group_by]
    // Use `group_by_stable` if you want the Python behaviour of `maintain_order=True`.
    let result = df
        .clone()
        .lazy()
        .group_by([(col("birthdate").dt().year() / lit(10) * lit(10)).alias("decade")])
        .agg([len()])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:group_by]

    // --8<-- [start:group_by-agg]
    let result = df
        .clone()
        .lazy()
        .group_by([(col("birthdate").dt().year() / lit(10) * lit(10)).alias("decade")])
        .agg([
            len().alias("sample_size"),
            col("weight").mean().round(2).alias("avg_weight"),
            col("height").max().alias("tallest"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:group_by-agg]

    // --8<-- [start:complex]
    let result = df
        .clone()
        .lazy()
        .with_columns([
            (col("birthdate").dt().year() / lit(10) * lit(10)).alias("decade"),
            col("name").str().split(lit(" ")).list().first(),
        ])
        .select([all().exclude(["birthdate"])])
        .group_by([col("decade")])
        .agg([
            col("name"),
            cols(["weight", "height"])
                .mean()
                .round(2)
                .name()
                .prefix("avg_"),
        ])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:complex]

    // --8<-- [start:join]
    let df2: DataFrame = df!(
        "name" => ["Ben Brown", "Daniel Donovan", "Alice Archer", "Chloe Cooper"],
        "parent" => [true, false, false, false],
        "siblings" => [1, 2, 3, 4],
    )
    .unwrap();

    let result = df
        .clone()
        .lazy()
        .join(
            df2.clone().lazy(),
            [col("name")],
            [col("name")],
            JoinArgs::new(JoinType::Left),
        )
        .collect()?;

    println!("{}", result);
    // --8<-- [end:join]

    // --8<-- [start:concat]
    let df3: DataFrame = df!(
        "name" => ["Ethan Edwards", "Fiona Foster", "Grace Gibson", "Henry Harris"],
        "birthdate" => [
            NaiveDate::from_ymd_opt(1977, 5, 10).unwrap(),
            NaiveDate::from_ymd_opt(1975, 6, 23).unwrap(),
            NaiveDate::from_ymd_opt(1973, 7, 22).unwrap(),
            NaiveDate::from_ymd_opt(1971, 8, 3).unwrap(),
        ],
        "weight" => [67.9, 72.5, 57.6, 93.1],  // (kg)
        "height" => [1.76, 1.6, 1.66, 1.8],  // (m)
    )
    .unwrap();

    let result = concat(
        [df.clone().lazy(), df3.clone().lazy()],
        UnionArgs::default(),
    )?
    .collect()?;
    println!("{}", result);
    // --8<-- [end:concat]

    Ok(())
}

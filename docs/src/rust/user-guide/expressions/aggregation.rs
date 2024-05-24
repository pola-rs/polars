use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    use std::io::Cursor;

    use reqwest::blocking::Client;

    let url = "https://theunitedstates.io/congress-legislators/legislators-historical.csv";

    let mut schema = Schema::new();
    schema.with_column(
        "first_name".into(),
        DataType::Categorical(None, Default::default()),
    );
    schema.with_column(
        "gender".into(),
        DataType::Categorical(None, Default::default()),
    );
    schema.with_column(
        "type".into(),
        DataType::Categorical(None, Default::default()),
    );
    schema.with_column(
        "state".into(),
        DataType::Categorical(None, Default::default()),
    );
    schema.with_column(
        "party".into(),
        DataType::Categorical(None, Default::default()),
    );
    schema.with_column("birthday".into(), DataType::Date);

    let data: Vec<u8> = Client::new().get(url).send()?.text()?.bytes().collect();

    let dataset = CsvReadOptions::default()
        .with_has_header(true)
        .with_schema(Some(Arc::new(schema)))
        .map_parse_options(|parse_options| parse_options.with_try_parse_dates(true))
        .into_reader_with_file_handle(Cursor::new(data))
        .finish()?;

    println!("{}", &dataset);
    // --8<-- [end:dataframe]

    // --8<-- [start:basic]
    let df = dataset
        .clone()
        .lazy()
        .group_by(["first_name"])
        .agg([len(), col("gender"), col("last_name").first()])
        .sort(
            ["len"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_nulls_last(true),
        )
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:basic]

    // --8<-- [start:conditional]
    let df = dataset
        .clone()
        .lazy()
        .group_by(["state"])
        .agg([
            (col("party").eq(lit("Anti-Administration")))
                .sum()
                .alias("anti"),
            (col("party").eq(lit("Pro-Administration")))
                .sum()
                .alias("pro"),
        ])
        .sort(
            ["pro"],
            SortMultipleOptions::default().with_order_descending(true),
        )
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:conditional]

    // --8<-- [start:nested]
    let df = dataset
        .clone()
        .lazy()
        .group_by(["state", "party"])
        .agg([col("party").count().alias("count")])
        .filter(
            col("party")
                .eq(lit("Anti-Administration"))
                .or(col("party").eq(lit("Pro-Administration"))),
        )
        .sort(
            ["count"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_nulls_last(true),
        )
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:nested]

    // --8<-- [start:filter]
    fn compute_age() -> Expr {
        lit(2022) - col("birthday").dt().year()
    }

    fn avg_birthday(gender: &str) -> Expr {
        compute_age()
            .filter(col("gender").eq(lit(gender)))
            .mean()
            .alias(&format!("avg {} birthday", gender))
    }

    let df = dataset
        .clone()
        .lazy()
        .group_by(["state"])
        .agg([
            avg_birthday("M"),
            avg_birthday("F"),
            (col("gender").eq(lit("M"))).sum().alias("# male"),
            (col("gender").eq(lit("F"))).sum().alias("# female"),
        ])
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:filter]

    // --8<-- [start:sort]
    fn get_person() -> Expr {
        col("first_name") + lit(" ") + col("last_name")
    }

    let df = dataset
        .clone()
        .lazy()
        .sort(
            ["birthday"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_nulls_last(true),
        )
        .group_by(["state"])
        .agg([
            get_person().first().alias("youngest"),
            get_person().last().alias("oldest"),
        ])
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:sort]

    // --8<-- [start:sort2]
    let df = dataset
        .clone()
        .lazy()
        .sort(
            ["birthday"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_nulls_last(true),
        )
        .group_by(["state"])
        .agg([
            get_person().first().alias("youngest"),
            get_person().last().alias("oldest"),
            get_person()
                .sort(Default::default())
                .first()
                .alias("alphabetical_first"),
        ])
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:sort2]

    // --8<-- [start:sort3]
    let df = dataset
        .clone()
        .lazy()
        .sort(
            ["birthday"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_nulls_last(true),
        )
        .group_by(["state"])
        .agg([
            get_person().first().alias("youngest"),
            get_person().last().alias("oldest"),
            get_person()
                .sort(Default::default())
                .first()
                .alias("alphabetical_first"),
            col("gender")
                .sort_by(["first_name"], SortMultipleOptions::default())
                .first()
                .alias("gender"),
        ])
        .sort(["state"], SortMultipleOptions::default())
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:sort3]

    Ok(())
}

use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dataframe]
    use std::io::Cursor;

    use reqwest::blocking::Client;

    let url = "https://theunitedstates.io/congress-legislators/legislators-historical.csv";

    let mut schema = Schema::new();
    schema.with_column("first_name".into(), DataType::Categorical(None));
    schema.with_column("gender".into(), DataType::Categorical(None));
    schema.with_column("type".into(), DataType::Categorical(None));
    schema.with_column("state".into(), DataType::Categorical(None));
    schema.with_column("party".into(), DataType::Categorical(None));
    schema.with_column("birthday".into(), DataType::Date);

    let data: Vec<u8> = Client::new().get(url).send()?.text()?.bytes().collect();

    let dataset = CsvReader::new(Cursor::new(data))
        .has_header(true)
        .with_dtypes(Some(Arc::new(schema)))
        .with_try_parse_dates(true)
        .finish()?;

    println!("{}", &dataset);
    // --8<-- [end:dataframe]

    // --8<-- [start:basic]
    let df = dataset
        .clone()
        .lazy()
        .group_by(["first_name"])
        .agg([count(), col("gender"), col("last_name").first()])
        .sort(
            "count",
            SortOptions {
                descending: true,
                nulls_last: true,
                ..Default::default()
            },
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
            "pro",
            SortOptions {
                descending: true,
                nulls_last: false,
                ..Default::default()
            },
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
            "count",
            SortOptions {
                descending: true,
                nulls_last: true,
                ..Default::default()
            },
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
            "birthday",
            SortOptions {
                descending: true,
                nulls_last: true,
                ..Default::default()
            },
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
            "birthday",
            SortOptions {
                descending: true,
                nulls_last: true,
                ..Default::default()
            },
        )
        .group_by(["state"])
        .agg([
            get_person().first().alias("youngest"),
            get_person().last().alias("oldest"),
            get_person().sort(false).first().alias("alphabetical_first"),
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
            "birthday",
            SortOptions {
                descending: true,
                nulls_last: true,
                ..Default::default()
            },
        )
        .group_by(["state"])
        .agg([
            get_person().first().alias("youngest"),
            get_person().last().alias("oldest"),
            get_person().sort(false).first().alias("alphabetical_first"),
            col("gender")
                .sort_by(["first_name"], [false])
                .first()
                .alias("gender"),
        ])
        .sort("state", SortOptions::default())
        .limit(5)
        .collect()?;

    println!("{}", df);
    // --8<-- [end:sort3]

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:pokemon]
    use polars::prelude::*;
    use reqwest::blocking::Client;

    let data: Vec<u8> = Client::new()
        .get("https://gist.githubusercontent.com/ritchie46/cac6b337ea52281aa23c049250a4ff03/raw/89a957ff3919d90e6ef2d34235e6bf22304f3366/pokemon.csv")
        .send()?
        .text()?
        .bytes()
        .collect();

    let file = std::io::Cursor::new(data);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()?;

    println!("{}", df);
    // --8<-- [end:pokemon]

    // --8<-- [start:rank]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:rank]

    // --8<-- [start:rank-multiple]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:rank-multiple]

    // --8<-- [start:rank-explode]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:rank-explode]

    // --8<-- [start:athletes]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:athletes]

    // --8<-- [start:athletes-sort-over-country]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:athletes-sort-over-country]

    // --8<-- [start:athletes-explode]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:athletes-explode]

    // --8<-- [start:athletes-join]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:athletes-join]

    // --8<-- [start:pokemon-mean]
    // Contribute the Rust translation of the Python example by opening a PR.
    // --8<-- [end:pokemon-mean]

    // --8<-- [start:group_by]
    let result = df
        .clone()
        .lazy()
        .select([
            col("Type 1"),
            col("Type 2"),
            col("Attack")
                .mean()
                .over(["Type 1"])
                .alias("avg_attack_by_type"),
            col("Defense")
                .mean()
                .over(["Type 1", "Type 2"])
                .alias("avg_defense_by_type_combination"),
            col("Attack").mean().alias("avg_attack"),
        ])
        .collect()?;

    println!("{}", result);
    // --8<-- [end:group_by]

    // --8<-- [start:operations]
    let filtered = df
        .clone()
        .lazy()
        .filter(col("Type 2").eq(lit("Psychic")))
        .select([col("Name"), col("Type 1"), col("Speed")])
        .collect()?;

    println!("{}", filtered);
    // --8<-- [end:operations]

    // --8<-- [start:sort]
    let result = filtered
        .lazy()
        .with_columns([cols(["Name", "Speed"])
            .sort_by(
                ["Speed"],
                SortMultipleOptions::default().with_order_descending(true),
            )
            .over(["Type 1"])])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:sort]

    // --8<-- [start:examples]
    let result = df
        .clone()
        .lazy()
        .select([
            col("Type 1").head(Some(3)).over(["Type 1"]).flatten(),
            col("Name")
                .sort_by(
                    ["Speed"],
                    SortMultipleOptions::default().with_order_descending(true),
                )
                .head(Some(3))
                .over(["Type 1"])
                .flatten()
                .alias("fastest/group"),
            col("Name")
                .sort_by(
                    ["Attack"],
                    SortMultipleOptions::default().with_order_descending(true),
                )
                .head(Some(3))
                .over(["Type 1"])
                .flatten()
                .alias("strongest/group"),
            col("Name")
                .sort(Default::default())
                .head(Some(3))
                .over(["Type 1"])
                .flatten()
                .alias("sorted_by_alphabet"),
        ])
        .collect()?;
    println!("{:?}", result);
    // --8<-- [end:examples]

    Ok(())
}

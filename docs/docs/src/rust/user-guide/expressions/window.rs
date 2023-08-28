use polars::prelude::*;

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

    let df = CsvReader::new(std::io::Cursor::new(data))
        .has_header(true)
        .finish()?;

    println!("{}", df);
    // --8<-- [end:pokemon]

    // --8<-- [start:groupby]
    let out = df
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

    println!("{}", out);
    // --8<-- [end:groupby]

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
    let out = filtered
        .lazy()
        .with_columns([cols(["Name", "Speed"]).sort_by(["Speed"],[true]).over(["Type 1"])])
        .collect()?;
    println!("{}", out);
    // --8<-- [end:sort]

    // --8<-- [start:rules]
    // aggregate and broadcast within a group
    // output type: -> i32
    sum("foo").over([col("groups")])
    // sum within a group and multiply with group elements
    // output type: -> i32
    (col("x").sum() * col("y"))
        .over([col("groups")])
        .alias("x1")
    // sum within a group and multiply with group elements
    // and aggregate the group to a list
    // output type: -> ChunkedArray<i32>
    (col("x").sum() * col("y"))
        .list()
        .over([col("groups")])
        .alias("x2")
    // note that it will require an explicit `list()` call
    // sum within a group and multiply with group elements
    // and aggregate the group to a list
    // the flatten call explodes that list

    // This is the fastest method to do things over groups when the groups are sorted
    (col("x").sum() * col("y"))
        .list()
        .over([col("groups")])
        .flatten()
        .alias("x3");
    // --8<-- [end:rules]

    // --8<-- [start:examples]
    let out = df
        .clone()
        .lazy()
        .select([
            col("Type 1")
                .head(Some(3))
                .list()
                .over(["Type 1"])
                .flatten(),
            col("Name")
                .sort_by(["Speed"], [true])
                .head(Some(3))
                .list()
                .over(["Type 1"])
                .flatten()
                .alias("fastest/group"),
            col("Name")
                .sort_by(["Attack"], [true])
                .head(Some(3))
                .list()
                .over(["Type 1"])
                .flatten()
                .alias("strongest/group"),
            col("Name")
                .sort(false)
                .head(Some(3))
                .list()
                .over(["Type 1"])
                .flatten()
                .alias("sorted_by_alphabet"),
        ])
        .collect()?;
    println!("{:?}", out);
    // --8<-- [end:examples]

    Ok(())
}
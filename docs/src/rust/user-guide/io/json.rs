use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:read]
    use polars::prelude::*;

    let mut file = std::fs::File::open("docs/data/path.json").unwrap();
    let df = JsonReader::new(&mut file).finish().unwrap();
    // --8<-- [end:read]
    println!("{}", df);

    // --8<-- [start:readnd]
    let mut file = std::fs::File::open("docs/data/path.json").unwrap();
    let df = JsonLineReader::new(&mut file).finish().unwrap();
    // --8<-- [end:readnd]
    println!("{}", df);

    // --8<-- [start:write]
    let mut df = df!(
        "foo" => &[1, 2, 3],
        "bar" => &[None, Some("bak"), Some("baz")],
    )
    .unwrap();

    let mut file = std::fs::File::create("docs/data/path.json").unwrap();

    // json
    JsonWriter::new(&mut file)
        .with_json_format(JsonFormat::Json)
        .finish(&mut df)
        .unwrap();

    // ndjson
    JsonWriter::new(&mut file)
        .with_json_format(JsonFormat::JsonLines)
        .finish(&mut df)
        .unwrap();
    // --8<-- [end:write]

    // --8<-- [start:scan]
    let lf = LazyJsonLineReader::new("docs/data/path.json")
        .finish()
        .unwrap();
    // --8<-- [end:scan]
    println!("{}", lf.collect()?);

    Ok(())
}

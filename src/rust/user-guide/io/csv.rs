use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:read]
    use polars::prelude::*;

    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("docs/data/path.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    // --8<-- [end:read]
    println!("{}", df);

    // --8<-- [start:write]
    let mut df = df!(
        "foo" => &[1, 2, 3],
        "bar" => &[None, Some("bak"), Some("baz")],
    )
    .unwrap();

    let mut file = std::fs::File::create("docs/data/path.csv").unwrap();
    CsvWriter::new(&mut file).finish(&mut df).unwrap();
    // --8<-- [end:write]

    // --8<-- [start:scan]
    let lf = LazyCsvReader::new("./test.csv").finish().unwrap();
    // --8<-- [end:scan]
    println!("{}", lf.collect()?);

    Ok(())
}

use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>>{

    """
    // --8<-- [start:read]
    use polars::prelude::*;

    let df = CsvReader::from_path("path.csv").unwrap().finish().unwrap();
    // --8<-- [end:read]
    """

    // --8<-- [start:write]
    let mut df = df!(
        "foo" => &[1, 2, 3],
        "bar" => &[None, Some("bak"), Some("baz")],
    )
    .unwrap();
    
    let mut file = std::fs::File::create("path.csv").unwrap();
    CsvWriter::new(&mut file).finish(&mut df).unwrap();
    // --8<-- [end:write]

    // --8<-- [start:scan]
    let df = LazyCsvReader::new("./test.csv").finish().unwrap();
    // --8<-- [end:scan]

    Ok(())
}
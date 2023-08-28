use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>>{

    """
    // --8<-- [start:read]
    let mut file = std::fs::File::open("path.parquet").unwrap();

    let df = ParquetReader::new(&mut file).finish().unwrap();
    // --8<-- [end:read]
    """

    // --8<-- [start:write]
    let mut df = df!(
        "foo" => &[1, 2, 3],
        "bar" => &[None, Some("bak"), Some("baz")],
    )
    .unwrap();
    
    let mut file = std::fs::File::create("path.parquet").unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
    // --8<-- [end:write]

    // --8<-- [start:scan]
    let args = ScanArgsParquet::default();
    let df = LazyFrame::scan_parquet("./file.parquet",args).unwrap();
    // --8<-- [end:scan]

    Ok(())
}
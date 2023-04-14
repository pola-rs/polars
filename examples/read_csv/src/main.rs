use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let mut buf = br#"foo,bar\n1,2"#;
    let mut cursor = std::io::Cursor::new(buf);
    let _ = polars::read_parquet!(&mut cursor);
    let path = std::path::Path::new("foo.parquet");
    let mut cursor = std::io::Cursor::new(buf);
    let _ = polars::read_parquet!(&mut cursor);
    
    let _ = polars::read_parquet!(&mut buf);
    let _ = polars::read_parquet!(&path);
    let _ = polars::read_parquet!(path);
    let _ = polars::read_parquet!("foo.parquet");

    let _ = polars::read_csv!("foo.csv");
    let _ = polars::read_csv!("foo.csv", delimiter = b',');
    let _ = polars::scan_csv!("foo.csv");
    let _ = polars::scan_csv!("foo.csv", has_header = true);
    let _ = polars::scan_csv!("foo.csv", has_header = true, delimiter = b',');
    let _ = polars::scan_csv!(
        "foo.csv",
        has_header = true,
        delimiter = b',',
        low_memory = true
    );

    Ok(())
}

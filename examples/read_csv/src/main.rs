use polars::prelude::*;

fn main() -> PolarsResult<()> {
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

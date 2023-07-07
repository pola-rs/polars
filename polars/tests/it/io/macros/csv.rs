use std::io::Cursor;

use super::super::*;

const FOODS_CSV: &str = "../examples/datasets/foods1.csv";

#[test]
fn read_csv_macro_1() {
    let df = polars::read_csv!(&FOODS_CSV).unwrap();
    assert_eq!(df.shape(), (27, 4));
}

#[test]
fn read_csv_macro_2() {
    let csv = std::fs::read_to_string(FOODS_CSV).unwrap();
    let csv = Cursor::new(csv);
    let df = polars::read_csv!(csv).unwrap();
    assert_eq!(df.shape(), (27, 4));
}

#[test]
fn read_csv_macro_3() {
    let csv = std::fs::read_to_string(FOODS_CSV).unwrap();
    let csv = Cursor::new(csv);
    let df = polars::read_csv!(
        csv,
        n_rows = 2,
        rechunk = true,
        n_threads = 1,
        has_header = false,
        row_count = ("rc", 0)
    )
    .unwrap();
    assert_eq!(df.shape(), (2, 5));
}

#[test]
fn scan_csv_macro_1() -> PolarsResult<()> {
    let df = polars::scan_csv!(&FOODS_CSV)?.collect()?;
    assert_eq!(df.shape(), (27, 4));
    Ok(())
}

#[test]
fn read_csv_macro_null_values() -> PolarsResult<()> {
    let _ = polars::read_csv!(&FOODS_CSV, null_values = "foo")?;
    let _ = polars::read_csv!(&FOODS_CSV, null_values = ("foo",))?;
    let _ = polars::read_csv!(&FOODS_CSV, null_values = ("foo", "bar"))?;
    let _ = polars::read_csv!(&FOODS_CSV, null_values = &["foo", "bar"])?;
    let _ = polars::read_csv!(&FOODS_CSV, null_values = ("foo", "bar"))?;
    let _ = polars::read_csv!(&FOODS_CSV, null_values = vec!["foo", "bar"])?;
    Ok(())
}

#[test]
fn scan_csv_macro_null_values() -> PolarsResult<()> {
    let _ = polars::scan_csv!(&FOODS_CSV, null_values = "foo")?.collect()?;
    let _ = polars::scan_csv!(&FOODS_CSV, null_values = ("foo",))?.collect()?;
    let _ = polars::scan_csv!(&FOODS_CSV, null_values = ("foo", "bar"))?.collect()?;
    let _ = polars::scan_csv!(&FOODS_CSV, null_values = &["foo", "bar"])?.collect()?;
    let _ = polars::scan_csv!(&FOODS_CSV, null_values = ("foo", "bar"))?.collect()?;
    let _ = polars::scan_csv!(&FOODS_CSV, null_values = vec!["foo", "bar"])?.collect()?;
    Ok(())
}

#[test]
fn scan_csv_macro_row_cound() -> PolarsResult<()> {
    let _ = polars::scan_csv!(&FOODS_CSV, row_count = ("rc", 0))?.collect()?;
    let _ = polars::scan_csv!(&FOODS_CSV, row_count = "rc", encoding = "utf8")?.collect()?;

    Ok(())
}

#[test]
fn scan_csv_macro_2() -> PolarsResult<()> {
    let df = polars::scan_csv!(
        &FOODS_CSV,
        cache = true,
        comment_char = b'/',
        delimiter = b',',
        encoding = "utf-8",
        eol_char = b'\n',
        has_header = true,
        ignore_errors = false,
        infer_schema_length = 100,
        low_memory = true,
        missing_is_null = true,
        n_rows = 100,
        null_values = "null"
    )?
    .collect()?;
    assert_eq!(df.shape(), (27, 4));
    Ok(())
}

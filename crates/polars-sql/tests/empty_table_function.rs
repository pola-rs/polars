// https://github.com/pola-rs/polars-cli/issues/51
#[cfg(any(
    feature = "csv",
    feature = "parquet",
    feature = "ipc",
    feature = "json"
))]
use polars_sql::*;

#[test]
#[cfg(feature = "csv")]
fn test_empty_table_csv_function() {
    let mut ctx = SQLContext::new();
    let actual = ctx.execute("SELECT * FROM read_csv()");
    assert!(actual.is_err());
}

#[test]
#[cfg(feature = "parquet")]
fn test_empty_table_parquet_function() {
    let mut ctx = SQLContext::new();
    let actual = ctx.execute("SELECT * FROM read_parquet()");
    assert!(actual.is_err());
}

#[test]
#[cfg(feature = "ipc")]
fn test_empty_table_ipc_function() {
    let mut ctx = SQLContext::new();
    let actual = ctx.execute("SELECT * FROM read_ipc()");
    assert!(actual.is_err());
}

#[test]
#[cfg(feature = "json")]
fn test_empty_table_json_function() {
    let mut ctx = SQLContext::new();
    let actual = ctx.execute("SELECT * FROM read_json()");
    assert!(actual.is_err());
}

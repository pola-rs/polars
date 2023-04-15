use std::io::Cursor;

use polars::prelude::*;

#[test]
fn test_vstack_empty_3220() -> PolarsResult<()> {
    let df1 = df! {
        "a" => ["1", "2"],
        "b" => [1, 2]
    }?;
    let empty_df = df1.head(Some(0));
    let mut stacked = df1.clone();
    stacked.vstack_mut(&empty_df)?;
    stacked.vstack_mut(&df1)?;
    let mut buf = Cursor::new(Vec::new());
    ParquetWriter::new(&mut buf).finish(&mut stacked)?;
    let read_df = ParquetReader::new(buf).finish()?;
    assert!(stacked.frame_equal(&read_df));
    Ok(())
}

#[test]
fn test_scan_parquet_files() -> PolarsResult<()> {
    let files_to_load_set = vec![
        "../examples/datasets/foods1.parquet".to_string(),
        "../examples/datasets/foods2.parquet".to_string(),
    ];

    // Use of deprecated scan_parquet_files() for testing purposes
    #[allow(deprecated)]
    let df = LazyFrame::scan_parquet_files(files_to_load_set, Default::default())?.collect()?;
    assert_eq!(df.shape(), (54, 4));
    Ok(())
}

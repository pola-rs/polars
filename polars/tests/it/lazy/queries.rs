use super::*;

#[test]
fn test_with_duplicate_column_empty_df() {
    let a = Int32Chunked::from_slice("a", &[]);

    assert_eq!(
        DataFrame::new(vec![a.into_series()])
            .unwrap()
            .lazy()
            .with_columns([lit(true).alias("a")])
            .collect()
            .unwrap()
            .get_column_names(),
        &["a"]
    );
}

#[test]
fn test_drop() -> Result<()> {
    // dropping all columns is a special case. It may fail because a projection
    // that projects nothing could be misinterpreted as select all.
    let out = df![
        "a" => [1],
    ]?
    .lazy()
    .drop_columns(["a"])
    .collect()?;
    assert_eq!(out.width(), 0);
    Ok(())
}

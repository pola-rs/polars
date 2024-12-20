use crate::prelude::*;

#[test]
fn test_initial_empty_sort() -> PolarsResult<()> {
    // https://github.com/pola-rs/polars/issues/1396
    let data = vec![1.3; 42];
    let mut series = Column::new("data".into(), Vec::<f64>::new());
    let series2 = Column::new("data2".into(), data.clone());
    let series3 = Column::new("data3".into(), data);
    let df = DataFrame::new(vec![series2, series3])?;

    for column in df.get_columns().iter() {
        series.append(column)?;
    }
    series.f64()?.sort(false);
    Ok(())
}

#[test]
fn test_unique_non_existent_subset_column() {
    let df = DataFrame::new(vec![
        Column::new("ID".into(), vec![1, 2, 1, 2]),
        Column::new("Name".into(), vec!["foo", "bar", "baz", "baa"]),
    ])
    .unwrap();

    let result = df.unique(Some(&["id"]), None, None);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().to_string(),
        "polars.exceptions.ColumnNotFoundError: \"id\" not found"
    );
}

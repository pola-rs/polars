use crate::prelude::*;

#[test]
fn test_initial_empty_sort() -> PolarsResult<()> {
    // https://github.com/pola-rs/polars/issues/1396
    let data = vec![1.3; 42];
    let mut series = Series::new("data", Vec::<f64>::new());
    let series2 = Series::new("data2", data.clone());
    let series3 = Series::new("data3", data);
    let df = DataFrame::new(vec![series2, series3])?;

    for column in df.get_columns().iter() {
        series.append(column)?;
    }
    series.f64()?.sort(false);
    Ok(())
}

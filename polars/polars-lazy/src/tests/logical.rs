use polars_core::utils::arrow::temporal_conversions::MILLISECONDS_IN_DAY;

use super::*;

#[test]
#[cfg(all(feature = "strings", feature = "temporal", feature = "dtype-duration"))]
fn test_duration() -> PolarsResult<()> {
    let df = df![
        "date" => ["2021-01-01", "2021-01-02", "2021-01-03"],
        "groups" => [1, 1, 1]
    ]?;

    let out = df
        .lazy()
        .with_columns(&[col("date").str().strptime(StrpTimeOptions {
            date_dtype: DataType::Date,
            ..Default::default()
        })])
        .with_column(
            col("date")
                .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                .alias("datetime"),
        )
        .groupby([col("groups")])
        .agg([
            (col("date") - col("date").first()).alias("date"),
            (col("datetime") - col("datetime").first()).alias("datetime"),
        ])
        .explode([col("date"), col("datetime")])
        .collect()?;

    for c in ["date", "datetime"] {
        let column = out.column(c)?;
        assert!(matches!(
            column.dtype(),
            DataType::Duration(TimeUnit::Milliseconds)
        ));

        assert_eq!(
            column.get(0)?,
            AnyValue::Duration(0, TimeUnit::Milliseconds)
        );
        assert_eq!(
            column.get(1)?,
            AnyValue::Duration(MILLISECONDS_IN_DAY, TimeUnit::Milliseconds)
        );
        assert_eq!(
            column.get(2)?,
            AnyValue::Duration(2 * MILLISECONDS_IN_DAY, TimeUnit::Milliseconds)
        );
    }
    Ok(())
}

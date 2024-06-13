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
        .with_columns(&[col("date").str().to_date(StrptimeOptions {
            ..Default::default()
        })])
        .with_column(
            col("date")
                .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                .alias("datetime"),
        )
        .group_by([col("groups")])
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

fn print_plans(lf: &LazyFrame) {
    println!("LOGICAL PLAN\n\n{}\n", lf.describe_plan().unwrap());
    println!(
        "OPTIMIZED LOGICAL PLAN\n\n{}\n",
        lf.describe_optimized_plan().unwrap()
    );
}

#[test]
fn test_lazy_arithmetic() {
    let df = get_df();
    let lf = df
        .lazy()
        .select(&[((col("sepal_width") * lit(100)).alias("super_wide"))])
        .sort(["super_wide"], SortMultipleOptions::default());

    print_plans(&lf);

    let new = lf.collect().unwrap();
    println!("{:?}", new);
    assert_eq!(new.height(), 7);
    assert_eq!(
        new.column("super_wide").unwrap().f64().unwrap().get(0),
        Some(300.0)
    );
}

#[test]
fn test_lazy_logical_plan_filter_and_alias_combined() {
    let df = get_df();
    let lf = df
        .lazy()
        .filter(col("sepal_width").lt(lit(3.5)))
        .select(&[col("variety").alias("foo")]);

    print_plans(&lf);
    let df = lf.collect().unwrap();
    println!("{:?}", df);
}

#[test]
fn test_lazy_logical_plan_schema() {
    let df = get_df();
    let lp = df
        .clone()
        .lazy()
        .select(&[col("variety").alias("foo")])
        .logical_plan;

    assert!(lp.compute_schema().unwrap().get("foo").is_some());

    let lp = df
        .lazy()
        .group_by([col("variety")])
        .agg([col("sepal_width").min()])
        .logical_plan;
    assert!(lp.compute_schema().unwrap().get("sepal_width").is_some());
}

#[test]
fn test_lazy_logical_plan_join() {
    let left = df!("days" => &[0, 1, 2, 3, 4],
    "temp" => [22.1, 19.9, 7., 2., 3.],
    "rain" => &[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    .unwrap();

    let right = df!(
    "days" => &[1, 2],
    "rain" => &[0.1, 0.2]
    )
    .unwrap();

    // check if optimizations succeeds without selection
    {
        let lf = left
            .clone()
            .lazy()
            .left_join(right.clone().lazy(), col("days"), col("days"));

        print_plans(&lf);
        // implicitly checks logical plan == optimized logical plan
        let _df = lf.collect().unwrap();
    }

    // check if optimization succeeds with selection
    {
        let lf = left
            .clone()
            .lazy()
            .left_join(right.clone().lazy(), col("days"), col("days"))
            .select(&[col("temp")]);

        let _df = lf.collect().unwrap();
    }

    // check if optimization succeeds with selection of a renamed column due to the join
    {
        let lf = left
            .lazy()
            .left_join(right.lazy(), col("days"), col("days"))
            .select(&[col("temp"), col("rain_right")]);

        print_plans(&lf);
        let _df = lf.collect().unwrap();
    }
}

use super::*;

#[test]
#[cfg(feature = "temporal")]
fn test_lazy_agg() {
    let s0 = DateChunked::parse_from_str_slice(
        "date".into(),
        &[
            "2020-08-21",
            "2020-08-21",
            "2020-08-22",
            "2020-08-23",
            "2020-08-22",
        ],
        "%Y-%m-%d",
    )
    .into_column();
    let s1 = Column::new("temp".into(), [20, 10, 7, 9, 1].as_ref());
    let s2 = Column::new("rain".into(), [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
    let df = DataFrame::new_infer_height(vec![s0, s1, s2]).unwrap();

    let lf = df
        .lazy()
        .group_by([col("date")])
        .agg([
            col("rain").min().alias("min"),
            col("rain").sum().alias("sum"),
            col("rain")
                .quantile(lit(0.5), QuantileMethod::default())
                .alias("median_rain"),
        ])
        .sort(["date"], Default::default());

    let new = lf.collect().unwrap();
    let min = new.column("min").unwrap();
    assert_eq!(min, &Column::new("min".into(), [0.1f64, 0.01, 0.1]));
}

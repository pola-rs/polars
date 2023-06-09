use super::*;

#[test]
fn test_lazy_df_aggregations() {
    let df = load_df();

    assert!(df
        .clone()
        .lazy()
        .min()
        .collect()
        .unwrap()
        .frame_equal_missing(&df.min()));
    assert!(df
        .clone()
        .lazy()
        .median()
        .collect()
        .unwrap()
        .frame_equal_missing(&df.median()));
    assert!(df
        .clone()
        .lazy()
        .quantile(lit(0.5), QuantileInterpolOptions::default())
        .collect()
        .unwrap()
        .frame_equal_missing(
            &df.quantile(0.5, QuantileInterpolOptions::default())
                .unwrap()
        ));
}

#[test]
#[cfg(feature = "temporal")]
fn test_lazy_agg() {
    let s0 = DateChunked::parse_from_str_slice(
        "date",
        &[
            "2020-08-21",
            "2020-08-21",
            "2020-08-22",
            "2020-08-23",
            "2020-08-22",
        ],
        "%Y-%m-%d",
    )
    .into_series();
    let s1 = Series::new("temp", [20, 10, 7, 9, 1].as_ref());
    let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
    let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

    let lf = df
        .lazy()
        .groupby([col("date")])
        .agg([
            col("rain").min().alias("min"),
            col("rain").sum().alias("sum"),
            col("rain")
                .quantile(lit(0.5), QuantileInterpolOptions::default())
                .alias("median_rain"),
        ])
        .sort("date", Default::default());

    let new = lf.collect().unwrap();
    let min = new.column("min").unwrap();
    assert_eq!(min, &Series::new("min", [0.1f64, 0.01, 0.1]));
}

#[test]
#[should_panic(expected = "hardcoded error")]
/// Test where apply_multiple returns an error
fn test_apply_multiple_error() {
    fn issue() -> Expr {
        apply_multiple(
            move |_| polars_bail!(ComputeError: "hardcoded error"),
            &[col("x"), col("y")],
            GetOutput::from_type(DataType::Float64),
            true,
        )
    }

    let df = df![
        "rf" => ["App", "App", "Gg", "App"],
        "x" => ["Hey", "There", "Ante", "R"],
        "y" => [Some(-1.11), Some(2.),None, Some(3.4)],
        "z" => [Some(-1.11), Some(2.),None, Some(3.4)],
    ]
    .unwrap();

    let _res = df
        .lazy()
        .with_streaming(false)
        .groupby_stable([col("rf")])
        .agg([issue()])
        .collect()
        .unwrap();
}

#[test]
/// Tests a complex aggregation, where groupby is called inside apply multiple
fn test_apply_multiple_groupby() {
    ///  apply_multiple which calls groupby inside
    fn inner() -> Expr {
        apply_multiple(
            move |columns| {
                let df = df![
                    "F" => &columns[0],
                    "B" => &columns[1],
                ]
                .unwrap();

                let df = df
                    .lazy()
                    .groupby([col("F")])
                    .agg([(col("B") * col("F").cast(DataType::Float64))
                        .sum()
                        .alias("res")])
                    .collect()
                    .unwrap();

                let res = df.column("res").unwrap().sum().unwrap_or(0.);

                return Ok(Some(Series::new("res", [res])));
            },
            &[col("F"), col("B")],
            GetOutput::from_type(DataType::Float64),
            true,
        )
    }

    let s0 = Series::new("B", [None, Some(2), None, None]);
    let s1 = Series::new("C", [1, 1, 3, 2]);
    let s2 = Series::new("F", ["1", "2", "1", "1"]);
    let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

    let res = df
        .lazy()
        .groupby_stable([col("C")])
        .agg([inner()])
        .collect()
        .unwrap();

    // expected
    let e0 = Series::new("F", [4., 0., 0.]);
    let e1 = Series::new("C", [1, 3, 2]);
    let expected = DataFrame::new(vec![e1, e0]).unwrap();

    assert!(res.frame_equal(&expected));
}

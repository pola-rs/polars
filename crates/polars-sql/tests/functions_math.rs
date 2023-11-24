use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
fn test_math_functions() {
    let df = df! {
        "a" => [1.0]
    }
    .unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            a,
            ABS(a) AS abs,
            ACOS(a) AS acos,
            ASIN(a) AS asin,
            ATAN(a) AS atan,
            PI() AS pi,
            CEIL(a) AS ceil,
            EXP(a) AS exp,
            FLOOR(a) AS floor,
            LN(a) AS ln,
            LOG2(a) AS log2,
            LOG10(a) AS log10,
            LOG(a, 5) AS log5,
            LOG1P(a) AS log1p,
            POW(a, 2) AS pow,
            SQRT(a) AS sqrt,
            CBRT(a) AS cbrt
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[
            col("a"),
            col("a").abs().alias("abs"),
            col("a").arccos().alias("acos"),
            col("a").arcsin().alias("asin"),
            col("a").arctan().alias("atan"),
            lit(std::f64::consts::PI).alias("pi"),
            col("a").ceil().alias("ceil"),
            col("a").exp().alias("exp"),
            col("a").floor().alias("floor"),
            col("a").log(std::f64::consts::E).alias("ln"),
            col("a").log(2.0).alias("log2"),
            col("a").log(10.0).alias("log10"),
            col("a").log(5.0).alias("log5"),
            col("a").log1p().alias("log1p"),
            col("a").pow(2.0).alias("pow"),
            col("a").sqrt().alias("sqrt"),
            col("a").cbrt().alias("cbrt"),
        ])
        .collect()
        .unwrap();
    println!("{}", df_pl.head(Some(10)));
    println!("{}", df_sql.head(Some(10)));
    assert!(df_sql.equals_missing(&df_pl));
}

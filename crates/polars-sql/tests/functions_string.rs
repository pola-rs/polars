use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::prelude::LiteralValue::Null;
use polars_sql::*;

#[test]
fn test_string_functions() {
    let df = df! {
        "a" => &["foo", "xxxbarxxx", "---bazyyy"]
    }
    .unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = r#"
        SELECT
            a,
            lower('LITERAL') as lower_literal,
            lower(a) as lower_a,
            lower("a") as lower_a2,
            lower(df.a) as lower_a_df,
            lower("df".a) as lower_a_df2,
            lower("df"."a") as lower_a_df3,
            upper(a) as upper_a,
            upper(df.a) as upper_a_df,
            upper("df".a) as upper_a_df2,
            upper("df"."a") as upper_a_df3,
            trim(both 'x' from a) as trim_a,
            trim(leading 'x' from a) as trim_a_leading,
            trim(trailing 'x' from a) as trim_a_trailing,
            ltrim(a) as ltrim_a,
            rtrim(a) as rtrim_a,
            ltrim(a, '-') as ltrim_a_dash,
            rtrim(a, '-') as rtrim_a_dash,
            ltrim(a, 'xyz') as ltrim_a_xyz,
            rtrim(a, 'xyz') as rtrim_a_xyz
        FROM df"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();
    let df_pl = df
        .lazy()
        .select(&[
            col("a"),
            lit("LITERAL").str().to_lowercase().alias("lower_literal"),
            col("a").str().to_lowercase().alias("lower_a"),
            col("a").str().to_lowercase().alias("lower_a2"),
            col("a").str().to_lowercase().alias("lower_a_df"),
            col("a").str().to_lowercase().alias("lower_a_df2"),
            col("a").str().to_lowercase().alias("lower_a_df3"),
            col("a").str().to_uppercase().alias("upper_a"),
            col("a").str().to_uppercase().alias("upper_a_df"),
            col("a").str().to_uppercase().alias("upper_a_df2"),
            col("a").str().to_uppercase().alias("upper_a_df3"),
            col("a").str().strip_chars(lit("x")).alias("trim_a"),
            col("a")
                .str()
                .strip_chars_start(lit("x"))
                .alias("trim_a_leading"),
            col("a")
                .str()
                .strip_chars_end(lit("x"))
                .alias("trim_a_trailing"),
            col("a").str().strip_chars_start(lit(Null)).alias("ltrim_a"),
            col("a").str().strip_chars_end(lit(Null)).alias("rtrim_a"),
            col("a")
                .str()
                .strip_chars_start(lit("-"))
                .alias("ltrim_a_dash"),
            col("a")
                .str()
                .strip_chars_end(lit("-"))
                .alias("rtrim_a_dash"),
            col("a")
                .str()
                .strip_chars_start(lit("xyz"))
                .alias("ltrim_a_xyz"),
            col("a")
                .str()
                .strip_chars_end(lit("xyz"))
                .alias("rtrim_a_xyz"),
        ])
        .collect()
        .unwrap();
    assert!(df_sql.equals_missing(&df_pl));
}

#[test]
fn array_to_string() {
    let df = df! {
        "a" => &["first", "first", "third"],
        "b" => &[1, 1, 42],
    }
    .unwrap();
    let mut context = SQLContext::new();
    context.register("df", df.clone().lazy());
    let sql = context
        .execute(
            r#"
        SELECT
            b,
            a
        FROM df
        GROUP BY
            b"#,
        )
        .unwrap();
    context.register("df_1", sql.clone());
    let sql = r#"
        SELECT
            b,
            array_to_string(a, ', ') as as,
        FROM df_1
        ORDER BY
            b,
            as"#;
    let df_sql = context.execute(sql).unwrap().collect().unwrap();

    let df_pl = df
        .lazy()
        .group_by([col("b")])
        .agg([col("a")])
        .select(&[col("b"), col("a").list().join(lit(", ")).alias("as")])
        .sort_by_exprs(vec![col("b"), col("as")], vec![false, false], false, true)
        .collect()
        .unwrap();

    assert!(df_sql.equals_missing(&df_pl));
}

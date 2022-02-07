use super::*;

#[test]
fn test_multiple_roots() -> Result<()> {
    let mut expr_arena = Arena::with_capacity(16);
    let mut lp_arena = Arena::with_capacity(8);

    let lf = scan_foods_parquet(false).select([col("calories").alias("bar")]);

    // this produces a predicate with two root columns, this test if we can
    // deal with multiple roots
    let lf = lf.filter(col("bar").gt(lit(45i32)));
    let lf = lf.filter(col("bar").lt(lit(110i32)));

    // also check if all predicates are combined and pushed down
    let root = lf.clone().optimize(&mut lp_arena, &mut expr_arena)?;
    assert!(predicate_at_scan(lf));
    // and that we don't have any filter node
    assert!(!(&lp_arena)
        .iter(root)
        .any(|(_, lp)| matches!(lp, ALogicalPlan::Selection { .. })));

    Ok(())
}

#[test]
#[cfg(all(feature = "is_in", feature = "strings"))]
fn test_issue_2472() -> Result<()> {
    let df = df![
        "group" => ["54360-2001-0-20020312-4-1"
    ,"39444-2020-0-20210418-4-1"
    ,"68398-2020-0-20201216-4-1"
    ,"30910-2020-0-20210223-4-1"
    ,"71060-2020-0-20210315-4-1"
    ,"47959-2020-0-20210305-4-1"
    ,"63212-2018-0-20181007-2-2"
    ,"61465-2018-0-20181018-2-2"
             ]
    ]?;
    let base = df
        .lazy()
        .with_column(col("group").cast(DataType::Categorical));

    let extract = col("group")
        .cast(DataType::Utf8)
        .str()
        .extract(r#"(\d+-){4}(\w+)-"#, 2)
        .cast(DataType::Int32)
        .alias("age");
    let predicate = col("age").is_in(lit(Series::new("", [2i32])));

    let out = base
        .clone()
        .with_column(extract.clone())
        .filter(predicate.clone())
        .collect()?;

    assert_eq!(out.shape(), (2, 2));

    let out = base.clone().select([extract]).filter(predicate).collect()?;
    assert_eq!(out.shape(), (2, 1));

    Ok(())
}

#[test]
fn test_pass_unrelated_apply() -> Result<()> {
    // maps should not influence a predicate of a different column as maps should not depend on previous values
    let df = fruits_cars();

    let q = df
        .lazy()
        .with_column(col("A").map(
            |s| Ok(s.is_null().into_series()),
            GetOutput::from_type(DataType::Boolean),
        ))
        .filter(col("B").gt(lit(10i32)));

    assert!(predicate_at_scan(q));

    Ok(())
}

#[test]
fn filter_added_column_issue_2470() -> Result<()> {
    let df = fruits_cars();

    // the binary expression in the predicate lead to an incorrect pushdown because the rhs
    // was not checked on the schema.
    let out = df
        .lazy()
        .select([col("A"), lit(NULL).alias("foo")])
        .filter(col("A").gt(lit(2i32)).and(col("foo").is_null()))
        .collect()?;
    assert_eq!(out.shape(), (3, 2));

    Ok(())
}

#[test]
fn filter_blocked_by_map() -> Result<()> {
    let df = fruits_cars();

    let allowed = AllowedOptimizations {
        predicate_pushdown: false,
        ..Default::default()
    };
    let q = df
        .lazy()
        .map(|df| Ok(df), Some(allowed), None, None)
        .filter(col("A").gt(lit(2i32)));

    assert!(!predicate_at_scan(q.clone()));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 4));

    Ok(())
}

#[test]
#[cfg(all(feature = "temporal", feature = "strings"))]
fn test_strptime_block_predicate() -> Result<()> {
    let df = df![
        "date" => ["2021-01-01", "2021-01-02"]
    ]?;

    let q = df
        .lazy()
        .with_column(col("date").str().strptime(StrpTimeOptions {
            date_dtype: DataType::Date,
            ..Default::default()
        }))
        .filter(col("date").gt(Expr::Literal(LiteralValue::DateTime(
            NaiveDate::from_ymd(2021, 1, 1).and_hms(0, 0, 0),
            TimeUnit::Milliseconds,
        ))));

    assert!(!predicate_at_scan(q.clone()));
    let df = q.collect()?;
    assert_eq!(df.shape(), (1, 1));

    Ok(())
}

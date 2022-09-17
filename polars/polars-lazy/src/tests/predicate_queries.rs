use super::*;

#[test]
fn test_multiple_roots() -> PolarsResult<()> {
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
fn test_issue_2472() -> PolarsResult<()> {
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
        .with_column(col("group").cast(DataType::Categorical(None)));

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
fn test_pass_unrelated_apply() -> PolarsResult<()> {
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
fn filter_added_column_issue_2470() -> PolarsResult<()> {
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
fn filter_blocked_by_map() -> PolarsResult<()> {
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
fn test_strptime_block_predicate() -> PolarsResult<()> {
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

#[test]
fn test_strict_cast_predicate_pushdown() -> PolarsResult<()> {
    let df = df![
        "a" => ["a", "b", "c"]
    ]?;

    let lf = df
        .lazy()
        .with_column(col("a").cast(DataType::Int32))
        .filter(col("a").is_null());

    assert!(!predicate_at_scan(lf.clone()));
    let out = lf.collect()?;
    assert_eq!(out.shape(), (3, 1));
    Ok(())
}

#[test]
fn test_filter_nulls_created_by_join() -> PolarsResult<()> {
    // #2602
    let a = df![
        "key" => ["foo", "bar"],
        "bar" => [1, 2]
    ]?;

    let b = df![
        "key"=> ["bar"]
    ]?
    .lazy()
    .with_column(lit(true).alias("flag"));

    let out = a
        .clone()
        .lazy()
        .join(b.clone(), [col("key")], [col("key")], JoinType::Left)
        .filter(col("flag").is_null())
        .collect()?;
    let expected = df![
        "key" => ["foo"],
        "bar" => [1],
        "flag" => &[None, Some(true)][0..1]
    ]?;
    assert!(out.frame_equal_missing(&expected));

    let out = a
        .lazy()
        .join(b.clone(), [col("key")], [col("key")], JoinType::Left)
        .filter(col("flag").eq(lit(NULL)))
        .with_predicate_pushdown(false)
        .collect()?;
    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

#[test]
fn test_filter_null_creation_by_cast() -> PolarsResult<()> {
    let df = df![
        "int" => [1, 2, 3],
        "empty" => ["", "", ""]
    ]?;

    let out = df
        .lazy()
        .with_column(col("empty").cast(DataType::Int32).alias("empty"))
        .filter(col("empty").is_null().and(col("int").eq(lit(3i32))))
        .collect()?;

    let expected = df![
        "int" => [3],
        "empty" => &[None, Some(1i32)][..1]
    ]?;
    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

#[test]
fn test_predicate_pd_apply() -> PolarsResult<()> {
    let q = df![
        "a" => [1, 2, 3],
    ]?
    .lazy()
    .select([
        // map_list is use in python `col().apply`
        col("a"),
        col("a")
            .map_list(|s| Ok(s), GetOutput::same_type())
            .alias("a_applied"),
    ])
    .filter(col("a").lt(lit(3)));

    assert!(predicate_at_scan(q.clone()));
    Ok(())
}
#[test]
#[cfg(feature = "cse")]
fn test_predicate_on_join_suffix_4788() -> PolarsResult<()> {
    let lf = df![
      "x" => [1, 2],
      "y" => [1, 1],
    ]?
    .lazy();

    let q = (lf.clone().join_builder().with(lf))
        .left_on([col("y")])
        .right_on([col("y")])
        .suffix("_")
        .finish()
        .filter(col("x").eq(1))
        .with_common_subplan_elimination(false);

    // the left hand side should have a predicate
    assert!(predicate_at_scan(q.clone()));

    let expected = df![
        "x" => [1, 1],
        "y" => [1, 1],
        "x_" => [1, 2],
    ]?;
    assert_eq!(q.collect()?, expected);

    Ok(())
}

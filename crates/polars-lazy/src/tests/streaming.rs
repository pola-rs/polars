use polars_ops::frame::JoinCoalesce;

use super::*;

fn get_csv_file() -> LazyFrame {
    let file = "../../examples/datasets/foods1.csv";
    LazyCsvReader::new(file).finish().unwrap()
}

fn get_parquet_file() -> LazyFrame {
    let file = "../../examples/datasets/foods1.parquet";
    LazyFrame::scan_parquet(file, Default::default()).unwrap()
}

fn get_csv_glob() -> LazyFrame {
    let file = "../../examples/datasets/foods*.csv";
    LazyCsvReader::new(file).finish().unwrap()
}

fn assert_streaming_with_default(q: LazyFrame, total: bool, check_shape_only: bool) {
    let q_streaming = q.clone().with_streaming(true);
    if total {
        assert!(optimization_checks::is_pipeline(q_streaming.clone()));
    } else {
        assert!(optimization_checks::has_pipeline(q_streaming.clone()));
    }
    let q_expected = q.with_streaming(false);
    let out = q_streaming.collect().unwrap();
    let expected = q_expected.collect().unwrap();
    if check_shape_only {
        assert_eq!(out.shape(), expected.shape())
    } else {
        assert_eq!(out, expected);
    }
}

#[test]
fn test_streaming_parquet() -> PolarsResult<()> {
    let q = get_parquet_file();

    let q = q
        .group_by([col("sugars_g")])
        .agg([((lit(1) - col("fats_g")) + col("calories")).sum()])
        .sort(["sugars_g"], Default::default());

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_streaming_csv() -> PolarsResult<()> {
    let q = get_csv_file();

    let q = q
        .select([col("sugars_g"), col("calories")])
        .group_by([col("sugars_g")])
        .agg([col("calories").sum()])
        .sort(["sugars_g"], Default::default());

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_streaming_glob() -> PolarsResult<()> {
    let q = get_csv_glob();
    let q = q.sort(["sugars_g"], Default::default());

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_streaming_union_order() -> PolarsResult<()> {
    let q = get_csv_glob();
    let q = concat([q.clone(), q], Default::default())?;
    let q = q.select([col("sugars_g"), col("calories")]);

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
#[cfg(feature = "cross_join")]
fn test_streaming_union_join() -> PolarsResult<()> {
    let q = get_csv_glob();
    let q = q.select([col("sugars_g"), col("calories")]);
    let q = q.clone().cross_join(q, None);

    assert_streaming_with_default(q, true, true);
    Ok(())
}

#[test]
fn test_streaming_multiple_keys_aggregate() -> PolarsResult<()> {
    let q = get_csv_glob();

    let q = q
        .filter(col("sugars_g").gt(lit(10)))
        .group_by([col("sugars_g"), col("calories")])
        .agg([
            (col("fats_g") * lit(10)).sum(),
            col("calories").mean().alias("cal_mean"),
        ])
        .sort_by_exprs(
            [col("sugars_g"), col("calories")],
            SortMultipleOptions::default().with_order_descending_multi([false, false]),
        );

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_streaming_first_sum() -> PolarsResult<()> {
    let q = get_csv_file();

    let q = q
        .select([col("sugars_g"), col("calories")])
        .group_by([col("sugars_g")])
        .agg([
            col("calories").sum(),
            col("calories").first().alias("calories_first"),
        ])
        .sort(["sugars_g"], Default::default());

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_streaming_unique() -> PolarsResult<()> {
    let q = get_csv_file();

    let q = q
        .select([col("sugars_g"), col("calories")])
        .unique(None, Default::default())
        .sort_by_exprs(
            [cols(["sugars_g", "calories"])],
            SortMultipleOptions::default(),
        );

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_streaming_aggregate_slice() -> PolarsResult<()> {
    let q = get_parquet_file();

    let q = q
        .group_by([col("sugars_g")])
        .agg([((lit(1) - col("fats_g")) + col("calories")).sum()])
        .slice(3, 3);

    let q1 = q.with_streaming(true);
    let out_streaming = q1.collect()?;
    assert_eq!(out_streaming.shape(), (3, 2));
    Ok(())
}

#[test]
#[cfg(feature = "cross_join")]
fn test_streaming_cross_join() -> PolarsResult<()> {
    let df = df![
        "a" => [1 ,2, 3]
    ]?;
    let q = df.lazy();
    let out = q
        .clone()
        .cross_join(q, None)
        .with_streaming(true)
        .collect()?;
    assert_eq!(out.shape(), (9, 2));

    let q = get_parquet_file().with_projection_pushdown(false);
    let q1 = q
        .clone()
        .select([col("calories")])
        .cross_join(q.clone(), None)
        .filter(col("calories").gt(col("calories_right")));
    let q2 = q1
        .select([all().name().suffix("_second")])
        .cross_join(q, None)
        .filter(col("calories_right_second").lt(col("calories")))
        .select([
            col("calories"),
            col("calories_right_second").alias("calories_right"),
        ]);

    let q2 = q2.with_streaming(true);
    let out_streaming = q2.collect()?;

    assert_eq!(
        out_streaming.get_column_names(),
        &["calories", "calories_right"]
    );
    assert_eq!(out_streaming.shape(), (5753, 2));
    Ok(())
}

#[test]
fn test_streaming_inner_join3() -> PolarsResult<()> {
    let lf_left = df![
        "col1" => [1, 1, 1],
        "col2" => ["a", "a", "b"],
        "int_col" => [1, 2, 3]
    ]?
    .lazy();

    let lf_right = df![
        "col1" => [1, 1, 1, 1, 1, 2],
        "col2" => ["a", "a", "a", "a", "a", "c"],
        "floats" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ]?
    .lazy();

    let q = lf_left.inner_join(lf_right, col("col1"), col("col1"));

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_streaming_inner_join2() -> PolarsResult<()> {
    let lf_left = df![
           "a"=> [0, 0, 0, 3, 0, 1, 3, 3, 3, 1, 4, 4, 2, 1, 1, 3, 1, 4, 2, 2],
    "b"=> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
       ]?
    .lazy();

    let lf_right = df![
           "a"=> [10, 18, 13, 9, 1, 13, 14, 12, 15, 11],
    "b"=> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
       ]?
    .lazy();

    let q = lf_left.inner_join(lf_right, col("a"), col("a"));

    assert_streaming_with_default(q, true, false);
    Ok(())
}
#[test]
fn test_streaming_left_join() -> PolarsResult<()> {
    let lf_left = df![
           "a"=> [0, 0, 0, 3, 0, 1, 3, 3, 3, 1, 4, 4, 2, 1, 1, 3, 1, 4, 2, 2],
    "b"=> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
       ]?
    .lazy();

    let lf_right = df![
           "a"=> [10, 18, 13, 9, 1, 13, 14, 12, 15, 11],
    "b"=> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
       ]?
    .lazy();

    let q = lf_left.left_join(lf_right, col("a"), col("a"));

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
#[cfg(feature = "cross_join")]
fn test_streaming_slice() -> PolarsResult<()> {
    let vals = (0..100).collect::<Vec<_>>();
    let s = Series::new("", vals);
    let lf_a = df![
        "a" => s
    ]?
    .lazy();

    let q = lf_a.clone().cross_join(lf_a, None).slice(10, 20);
    let a = q.with_streaming(true).collect().unwrap();
    assert_eq!(a.shape(), (20, 2));

    Ok(())
}

#[test]
fn test_streaming_partial() -> PolarsResult<()> {
    let lf_left = df![
        "a"=> [0],
         "b"=> [0],
    ]?
    .lazy();

    let lf_right = df![
        "a"=> [0],
         "b"=> [0],
    ]?
    .lazy();

    let q = lf_left.clone().left_join(lf_right, col("a"), col("a"));

    // we add a join that is not supported streaming (for now)
    // so we can test if the partial query is executed without panics
    let q = q
        .join_builder()
        .with(lf_left.clone())
        .left_on([col("a")])
        .right_on([col("a")])
        .suffix("_foo")
        .how(JoinType::Full)
        .coalesce(JoinCoalesce::CoalesceColumns)
        .finish();

    let q = q.left_join(
        lf_left.select([all().name().suffix("_foo")]),
        col("a"),
        col("a_foo"),
    );
    assert_streaming_with_default(q, false, true);

    Ok(())
}

#[test]
fn test_streaming_aggregate_join() -> PolarsResult<()> {
    let q = get_parquet_file();

    let q = q
        .group_by([col("sugars_g")])
        .agg([((lit(1) - col("fats_g")) + col("calories")).sum()])
        .slice(0, 3);

    let q = q.clone().left_join(q, col("sugars_g"), col("sugars_g"));
    let q1 = q.with_streaming(true);
    let out_streaming = q1.collect()?;
    assert_eq!(out_streaming.shape(), (3, 3));
    Ok(())
}

#[test]
fn test_streaming_double_left_join() -> PolarsResult<()> {
    // A left join swaps the tables, so that checks the swapping of the branches
    let q1 = df![
        "id" => ["1a"],
        "p_id" => ["1b"],
        "m_id" => ["1c"],
    ]?
    .lazy();

    let q2 = df![
        "p_id2" => ["2a"],
        "p_code" => ["2b"],
    ]?
    .lazy();

    let q3 = df![
        "m_id3" => ["3a"],
        "m_code" => ["3b"],
    ]?
    .lazy();

    let q = q1
        .clone()
        .left_join(q2.clone(), col("p_id"), col("p_id2"))
        .left_join(q3.clone(), col("m_id"), col("m_id3"));

    assert_streaming_with_default(q.clone(), true, false);

    // more joins
    let q = q.left_join(q1.clone(), col("p_id"), col("p_id")).left_join(
        q3.clone(),
        col("m_id"),
        col("m_id3"),
    );

    assert_streaming_with_default(q, true, false);
    // empty tables
    let q = q1
        .slice(0, 0)
        .left_join(q2.slice(0, 0), col("p_id"), col("p_id2"))
        .left_join(q3.slice(0, 0), col("m_id"), col("m_id3"));

    assert_streaming_with_default(q, true, false);
    Ok(())
}

#[test]
fn test_sort_maintain_order_streaming() -> PolarsResult<()> {
    let q = df![
        "A" => [1, 1, 1, 1],
        "B" => ["A", "B", "C", "D"],
    ]?
    .lazy();

    let res = q
        .sort_by_exprs(
            [col("A")],
            SortMultipleOptions::default()
                .with_nulls_last(true)
                .with_maintain_order(true),
        )
        .slice(0, 3)
        .with_streaming(true)
        .collect()?;
    assert!(res.equals(&df![
        "A" => [1, 1, 1],
        "B" => ["A", "B", "C"],
    ]?));
    Ok(())
}

#[test]
fn test_streaming_full_outer_join() -> PolarsResult<()> {
    let lf_left = df![
         "a"=> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "b"=> [0, 0, 0, 3, 0, 1, 3, 3, 3, 1, 4, 4, 2, 1, 1, 3, 1, 4, 2, 2],
    ]?
    .lazy();

    let lf_right = df![
           "a"=> [10, 18, 13, 9, 1, 13, 14, 12, 15, 11],
    "b"=> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
       ]?
    .lazy();

    let q = lf_left
        .full_join(lf_right, col("a"), col("a"))
        .sort_by_exprs([all()], SortMultipleOptions::default());

    // Toggle so that the join order is swapped.
    for toggle in [true, true] {
        assert_streaming_with_default(q.clone().with_streaming(toggle), true, false);
    }

    Ok(())
}

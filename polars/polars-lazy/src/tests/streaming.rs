use polars_core::prelude::JoinType::AsOf;

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

#[test]
fn test_streaming_parquet() -> PolarsResult<()> {
    let q = get_parquet_file();

    let q = q
        .groupby([col("sugars_g")])
        .agg([((lit(1) - col("fats_g")) + col("calories")).sum()])
        .sort("sugars_g", Default::default());

    let q1 = q.clone().with_streaming(true);
    let q2 = q.clone();
    let out_streaming = q1.collect()?;
    let out_one_shot = q2.collect()?;

    assert!(out_streaming.frame_equal(&out_one_shot));
    Ok(())
}

#[test]
fn test_streaming_csv() -> PolarsResult<()> {
    let q = get_csv_file();

    let q = q
        .select([col("sugars_g"), col("calories")])
        .groupby([col("sugars_g")])
        .agg([col("calories").sum()])
        .sort("sugars_g", Default::default());

    let q1 = q.clone().with_streaming(true);
    let q2 = q.clone();
    let out_streaming = q1.collect()?;
    let out_one_shot = q2.collect()?;

    assert!(out_streaming.frame_equal(&out_one_shot));
    Ok(())
}

#[test]
fn test_streaming_glob() -> PolarsResult<()> {
    let q = get_csv_glob();

    let q = q
        .select([col("sugars_g"), col("calories")])
        .filter(col("sugars_g").gt(lit(10)))
        .groupby([col("sugars_g")])
        .agg([col("calories").sum() * lit(10)])
        .sort("sugars_g", Default::default());

    let q1 = q.clone().with_streaming(true);
    let q2 = q.clone();
    let out_streaming = q1.collect()?;
    let out_one_shot = q2.collect()?;

    assert!(out_streaming.frame_equal(&out_one_shot));
    Ok(())
}

#[test]
fn test_streaming_multiple_keys_aggregate() -> PolarsResult<()> {
    let q = get_csv_glob();

    let q = q
        .filter(col("sugars_g").gt(lit(10)))
        .groupby([col("sugars_g"), col("calories")])
        .agg([
            (col("fats_g") * lit(10)).sum(),
            col("calories").mean().alias("cal_mean"),
        ])
        .sort_by_exprs([col("sugars_g"), col("calories")], [false, false], false);

    let q1 = q.clone().with_streaming(true);
    let q2 = q.clone();
    let out_streaming = q1.collect()?;
    let out_one_shot = q2.collect()?;

    assert!(out_streaming.frame_equal(&out_one_shot));
    Ok(())
}

#[test]
fn test_streaming_first_sum() -> PolarsResult<()> {
    let q = get_csv_file();

    let q = q
        .select([col("sugars_g"), col("calories")])
        .groupby([col("sugars_g")])
        .agg([
            col("calories").sum(),
            col("calories").first().alias("calories_first"),
        ])
        .sort("sugars_g", Default::default());

    let q1 = q.clone().with_streaming(true);
    let q2 = q.clone();
    let out_streaming = q1.collect()?;
    let out_one_shot = q2.collect()?;

    assert!(out_streaming.frame_equal(&out_one_shot));
    Ok(())
}

#[test]
fn test_streaming_aggregate_slice() -> PolarsResult<()> {
    let q = get_parquet_file();

    let q = q
        .groupby([col("sugars_g")])
        .agg([((lit(1) - col("fats_g")) + col("calories")).sum()])
        .slice(3, 3);

    let q1 = q.clone().with_streaming(true);
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
    let out = q.clone().cross_join(q).with_streaming(true).collect()?;
    assert_eq!(out.shape(), (9, 2));

    let q = get_parquet_file().with_projection_pushdown(false);
    let q1 = q
        .clone()
        .select([col("calories")])
        .clone()
        .cross_join(q.clone())
        .filter(col("calories").gt(col("calories_right")));
    let q2 = q1
        .select([all().suffix("_second")])
        .cross_join(q)
        .filter(col("calories_right_second").lt(col("calories")))
        .select([
            col("calories"),
            col("calories_right_second").alias("calories_right"),
        ]);

    let q2 = q2.clone().with_streaming(true);
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

    let out1 = q.clone().with_streaming(true).collect()?;
    let out2 = q.clone().with_streaming(false).collect()?;
    assert!(out1.frame_equal(&out2));

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

    let out1 = q.clone().with_streaming(true).collect()?;
    let out2 = q.clone().with_streaming(false).collect()?;
    assert!(out1.frame_equal(&out2));

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

    let out1 = q.clone().with_streaming(false).collect()?;
    let out2 = q.clone().with_streaming(false).collect()?;
    assert!(out1.frame_equal_missing(&out2));

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

    let q = lf_a.clone().cross_join(lf_a).slice(10, 20);
    let a = q.clone().with_streaming(true).collect().unwrap();
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
    // so we can test if the partial query is executed with out panics
    let q = q
        .join_builder()
        .with(lf_left.clone())
        .left_on([col("a")])
        .right_on([col("a")])
        .suffix("_foo")
        .how(JoinType::Outer)
        .finish();

    let q = q.left_join(
        lf_left.clone().select([all().suffix("_foo")]),
        col("a"),
        col("a_foo"),
    );
    let out = q.with_streaming(true).collect()?;
    // simply check if it runs panic free
    assert_eq!(out.shape(), (1, 5));

    Ok(())
}

#[test]
fn test_streaming_aggregate_join() -> PolarsResult<()> {
    let q = get_parquet_file();

    let q = q
        .groupby([col("sugars_g")])
        .agg([((lit(1) - col("fats_g")) + col("calories")).sum()])
        .slice(0, 3);

    let q = q.clone().left_join(q, col("sugars_g"), col("sugars_g"));
    let q1 = q.clone().with_streaming(true);
    let out_streaming = q1.collect()?;
    assert_eq!(out_streaming.shape(), (3, 3));
    Ok(())
}

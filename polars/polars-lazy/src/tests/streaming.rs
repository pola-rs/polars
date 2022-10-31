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
fn test_streaming_slice() -> PolarsResult<()> {
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
fn test_streaming_cross_join() -> PolarsResult<()> {
    let df = df![
        "a" => [1 ,2, 3]
    ]?;
    let q = df.lazy();
    let out = q.clone().cross_join(q).with_streaming(true).collect()?;
    assert_eq!(out.shape(), (9, 2));

    let q = get_parquet_file().with_projection_pushdown(false); // ;.slice(3, 3);
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

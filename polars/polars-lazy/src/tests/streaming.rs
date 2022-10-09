use super::*;

fn get_file() -> LazyFrame {
    let file = "../../examples/datasets/foods1.csv";
    LazyCsvReader::new(file).finish().unwrap()
}

#[test]
fn test_streaming_csv() -> PolarsResult<()> {
    let q = get_file();

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
fn test_streaming_first_sum() -> PolarsResult<()> {
    let q = get_file();

    let q = q
        .select([col("sugars_g"), col("calories")])
        .groupby([col("sugars_g")])
        .agg([col("calories").sum(), col("calories").first().alias("calories_first")])
        .sort("sugars_g", Default::default());

    let q1 = q.clone().with_streaming(true);
    let q2 = q.clone();
    let out_streaming = q1.collect()?;
    let out_one_shot = q2.collect()?;

    assert!(out_streaming.frame_equal(&out_one_shot));
    Ok(())
}

#[test]
fn test_streaming2() -> PolarsResult<()> {

    let out = LazyCsvReader::new("/home/ritchie46/Downloads/csv-benchmark/yellow_tripdata_2010-01.csv")
        .finish().unwrap()
        .groupby([col("passenger_count")])
        .agg([
            col("rate_code").sum(), col("rate_code").first().alias("first")
        ]).with_streaming(true).collect()?;

    dbg!(out);

    Ok(())
}
use super::*;

#[test]
fn test_streaming_csv() -> PolarsResult<()> {
    let file = "../../examples/datasets/foods1.csv";
    let q = LazyCsvReader::new(file).finish()?;

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

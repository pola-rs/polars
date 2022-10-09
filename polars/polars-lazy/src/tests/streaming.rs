use super::*;

#[test]
fn test_streaming_csv() -> PolarsResult<()> {
    let file = "../../examples/datasets/foods1.csv";
    let q = LazyCsvReader::new(file).finish()?;

    // let out = q
    //     .select([col("category"), col("calories")])
    //     .collect_streaming()?;

    let out = q
        .select([col("sugars_g"), col("calories")])
        .groupby([col("sugars_g")])
        .agg([col("calories").sum()])
        .collect_streaming()?;

    dbg!(out);
    Ok(())
}

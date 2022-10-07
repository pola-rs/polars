use super::*;

#[test]
fn test_streaming_csv() -> PolarsResult<()> {
    let file = "../../examples/datasets/foods1.csv";
    let q = LazyCsvReader::new(file).finish()?;

    let out = q
        .select([col("category"), col("calories")])
        .collect_streaming()?;

    dbg!(out);
    Ok(())
}

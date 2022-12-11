// used only if feature="strings"
#[allow(unused_imports)]
use super::*;

#[cfg(feature = "strings")]
#[test]
fn test_explode_row_numbers() -> PolarsResult<()> {
    let df = df![
        "text" => ["one two three four", "uno dos tres cuatro"]
    ]?
    .lazy()
    .select([col("text").str().split(" ").alias("tokens")])
    .with_row_count("row_nr", None)
    .explode([col("tokens")])
    .select([col("row_nr"), col("tokens")])
    .collect()?;

    assert_eq!(df.shape(), (8, 2));
    Ok(())
}

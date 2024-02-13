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
    .select([col("text").str().split(lit(" ")).alias("tokens")])
    .with_row_index("index", None)
    .explode([col("tokens")])
    .select([col("index"), col("tokens")])
    .collect()?;

    assert_eq!(df.shape(), (8, 2));
    Ok(())
}

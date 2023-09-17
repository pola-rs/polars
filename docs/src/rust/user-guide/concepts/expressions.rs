use polars::prelude::*;
use rand::Rng;
use chrono::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>>{

    let df = df! (
        "foo" => &[Some(1), Some(2), Some(3), None, Some(5)],
        "bar" => &[Some("foo"), Some("ham"), Some("spam"), Some("egg"), None],
    )?;

    // --8<-- [start:example1]
    df.column("foo")?.sort(false).head(Some(2));
    // --8<-- [end:example1]

    // --8<-- [start:example2]
    df.clone().lazy().select([
        col("foo").sort(Default::default()).head(Some(2)),
        col("bar").filter(col("foo").eq(lit(1))).sum(),
     ]).collect()?;
    // --8<-- [end:example2]

    Ok(())
}
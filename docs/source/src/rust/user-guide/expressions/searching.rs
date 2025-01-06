fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:dfnum]
    use polars::prelude::*;

    let df = df! (
        "integers" => &[Some(17), Some(42), None, Some(35)],
    )?;
    // --8<-- [end:dfnum]

    // --8<-- [start:index_of]
    let result = df
        .clone()
        .lazy()
        .select([col("integers").index_of(35)])
        .collect()?;
    println!("{}", result);

    let result = df
        .clone()
        .lazy()
        .select([col("integers").index_of(Null {}.lit())])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:index_of]

    // --8<-- [start:index_of_not_found]
    let result = df
        .clone()
        .lazy()
        .select([col("integers").index_of(1233)])
        .collect()?;
    println!("{}", result);
    // --8<-- [end:index_of_not_found]

    Ok(())
}
